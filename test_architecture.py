import os
os.makedirs("output", exist_ok=True)

import pandas as pd
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
import os
import itertools

from models import ClassificadorV2, GOTYModelV2
from data_handler import GOTYDataModule
import config as cfg

ACCELERATOR = "gpu"
DEVICES = 1
SWEEP_BATCH_SIZE = 512
SWEEP_EPOCHS = 30
SEED = 42


def carregar_dados():
    games = pd.read_csv("data/games.csv")
    trends = pd.read_csv("data/yearly_trends.csv")
    
    return games, trends


def contar_parametros(model):
    """Retorna (total_params, trainable_params) do modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    if ACCELERATOR == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU solicitada, mas torch.cuda.is_available() retornou False.")

    L.seed_everything(SEED, workers=True)

    games, trends = carregar_dados()

    X_baseline = games.drop(columns=cfg.COLUNAS_DROP, errors="ignore")
    y = games["goty_nominated"]

    os.makedirs("logs", exist_ok=True)

    # Usamos apenas 1 fold fixo (o primeiro) para o sweep de capacidade,
    # já que o objetivo aqui é comparar arquiteturas, não validar generalização final.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X_baseline, y))

    data_module = GOTYDataModule(
        X=X_baseline,
        y=y,
        trends_df=trends,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=SWEEP_BATCH_SIZE,
    )
    data_module.setup()

    # Grade de busca: ajuste conforme necessário
    grid_n_hidden = [0, 1, 2, 3, 4]
    grid_n_neurons = [16, 32, 64, 128, 256]

    resultados = []
    combinacoes = list(itertools.product(grid_n_hidden, grid_n_neurons))

    for i, (n_hidden, n_neurons) in enumerate(combinacoes, start=1):
        print(f"\\n{'='*50}")
        print(f"🔍 SWEEP {i}/{len(combinacoes)} | n_hidden={n_hidden} | n_neurons={n_neurons}")
        print(f"{'='*50}")

        modelo_base = ClassificadorV2(
            input_size=data_module.input_size,
            n_neurons=n_neurons,
            n_hidden=n_hidden,
        )

        total_params, trainable_params = contar_parametros(modelo_base)

        l_model = GOTYModelV2(
            model=modelo_base,
            learning_rate=cfg.LR,
            pos_weight_val=cfg.POS_WEIGHT_VAL,
            scheduler_t_max=SWEEP_EPOCHS,
        )

        sweep_logger = CSVLogger(
            save_dir="logs",
            name="capacity_sweep",
            version=f"nh{n_hidden}_nn{n_neurons}",
        )

        trainer = L.Trainer(
            max_epochs=SWEEP_EPOCHS,
            accelerator=ACCELERATOR,
            devices=DEVICES,
            logger=sweep_logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        trainer.fit(l_model, datamodule=data_module)
        val_metrics = trainer.validate(l_model, datamodule=data_module, verbose=False)[0]

        resultados.append({
            "n_hidden": n_hidden,
            "n_neurons": n_neurons,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "val_loss": val_metrics.get("val_loss"),
            "val_auroc": val_metrics.get("val_auroc"),
            "val_f1": val_metrics.get("val_f1"),
        })

    df_resultados = pd.DataFrame(resultados).sort_values("val_auroc", ascending=False)

    os.makedirs("output", exist_ok=True)
    df_resultados.to_csv("output/capacity_sweep_resultados.csv", index=False)

    print("\\n📊 Resultados do sweep de capacidade (top 10 por val_auroc):")
    print(df_resultados.head(10).to_string(index=False))

    return df_resultados


if __name__ == "__main__":
    df_resultados = main()
