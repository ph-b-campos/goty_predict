import pandas as pd
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import ClassificadorV4, ClassificadorV5, GOTYModelV3, GOTYModelV4
from data_handler import GOTYDataModule
import config as cfg
RUN_NAME = "cv_resultados_v5_final"
def carregar_dados():
    games = pd.read_csv("data/games.csv")
    trends = pd.read_csv("data/yearly_trends.csv")
    return games, trends

def main():
    games, trends = carregar_dados()

    X_baseline = games.drop(columns=cfg.COLUNAS_DROP, errors='ignore')
    y = games['goty_nominated']

    os.makedirs("logs", exist_ok=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_baseline, y)):
        print(f"\n{'='*40}")
        print(f" INICIANDO FOLD {fold + 1}/10")
        print(f"{'='*40}")

        data_module = GOTYDataModule(
            X=X_baseline,
            y=y,
            trends_df=trends,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=cfg.BATCH_SIZE
        )

        data_module.setup()

        modelo_base = ClassificadorV5(
            input_size=data_module.input_size,
            n_neurons=cfg.N_NEURONS,
            n_hidden=cfg.N_HIDDEN,
            dropout_rate=0.5
        )

        l_model = GOTYModelV4(
            model=modelo_base,
            learning_rate=cfg.LR,
            pos_weight_val=5
        )

        fold_logger = CSVLogger(
            save_dir="logs",
            name=RUN_NAME,
            version=f"fold_{fold + 1}"
        )

        # Mudança: interrompe quando a val_loss deixa de melhorar
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=7,
            min_delta=0.0001,
            verbose=True
        )

        # Mudança: salva apenas o checkpoint com menor val_loss
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{fold_logger.log_dir}/checkpoints",
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=False,
            filename='best-{epoch:02d}-{val_loss:.4f}'
        )

        trainer = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger=fold_logger,
            callbacks=[early_stopping, checkpoint_callback],
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )

        trainer.fit(l_model, datamodule=data_module)

        print(f"Melhor checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Melhor val_loss: {checkpoint_callback.best_model_score.item():.4f}")

if __name__ == "__main__":
    main()