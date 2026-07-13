import os

os.makedirs("output", exist_ok=True)

import pandas as pd
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
import os
import itertools

from models import ClassificadorV2, ClassificadorV3, GOTYModelV2, GOTYModelV3
from data_handler import GOTYDataModule
import config as cfg


SWEEP_BATCH_SIZE = 256
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

    L.seed_everything(SEED, workers=True)

    games, trends = carregar_dados()

    X_baseline = games.drop(columns=cfg.COLUNAS_DROP, errors="ignore")
    y = games["goty_nominated"]

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
    pos_weight_val = 20
    pos_weight = [0.25*pos_weight_val,0.5*pos_weight_val,0.75*pos_weight_val,1*pos_weight_val,1.25*pos_weight_val]
    dropout = [0.1, 0.3, 0.5]

    resultados = []
    combinacoes = list(itertools.product(pos_weight, dropout))

    for i, (pos_weight, dropout) in enumerate(combinacoes, start=1):
        print(f"\\n{'='*50}")
        print(f"🔍 SWEEP {i}/{len(combinacoes)} | pos_weight={pos_weight} | dropout={dropout}")
        print(f"{'='*50}")

        modelo_base = ClassificadorV3(
            input_size=data_module.input_size,
            n_neurons=32,
            n_hidden=1,
            dropout_rate=dropout
        )

        total_params, trainable_params = contar_parametros(modelo_base)

        l_model = GOTYModelV3(
            model=modelo_base,
            learning_rate=cfg.LR,
            pos_weight_val=pos_weight,
            scheduler_t_max=SWEEP_EPOCHS,
        )

        trainer = L.Trainer(
            max_epochs=SWEEP_EPOCHS,
            accelerator='cpu',
            devices=1,
            enable_checkpointing=False,
            enable_model_summary=True,
            enable_progress_bar=True,
        )

        trainer.fit(l_model, datamodule=data_module)
        val_metrics = trainer.validate(l_model, datamodule=data_module, verbose=False)[0]

        resultados.append({
            "pos_weight": pos_weight,
            "dropout": dropout,
            "val_loss": val_metrics.get("val_loss"),
            "val_auroc": val_metrics.get("val_auroc"),
            "val_f1": val_metrics.get("val_f1"),
        })

    df_resultados = pd.DataFrame(resultados).sort_values("val_auroc", ascending=False)

    os.makedirs("output", exist_ok=True)
    df_resultados.to_csv("output/pos_weight_sweep_resultados.csv", index=False)
    return df_resultados


if __name__ == "__main__":
    df_resultados = main()
