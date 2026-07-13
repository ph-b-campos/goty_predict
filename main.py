import pandas as pd
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from models import ClassificadorV5, ClassificadorV2, ClassificadorV3, GOTYModel, GOTYModelV2, GOTYModelV3, GOTYModelV5
from data_handler import GOTYDataModule
import config as cfg

SAVE_DIR = f"logs/{cfg.SAMPLING_STRATEGY}"

def carregar_dados():

    games = pd.read_csv("data/games.csv")
    trends = pd.read_csv("data/yearly_trends.csv")
    return games, trends

def main():
    games, trends = carregar_dados()

    X_baseline = games.drop(columns=cfg.COLUNAS_DROP, errors='ignore')
    y = games['goty_nominated']

    os.makedirs(SAVE_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_baseline, y)):
        print(f"\n{'='*40}")
        print(f"🚀 INICIANDO FOLD {fold + 1}/10")
        print(f"{'='*40}")

        data_module = GOTYDataModule(
            X=X_baseline,
            y=y,
            trends_df=trends,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=cfg.BATCH_SIZE,
            sampling_strategy=cfg.SAMPLING_STRATEGY
        )

        data_module.setup()

        modelo_base = ClassificadorV5(
            input_size=data_module.input_size, 
            n_neurons=cfg.N_NEURONS, 
            n_hidden=2
        )
        
        l_model = GOTYModelV5(
            model=modelo_base, 
            learning_rate=cfg.LR,
            pos_weight_val=cfg.POS_WEIGHT_VAL,
            threshold=cfg.TRESHOLD
        )

        fold_logger = CSVLogger(
            save_dir=SAVE_DIR, 
            name="cv_resultados_v5", 
            version=f"fold_{fold + 1}"
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='max',
            patience=7,
            min_delta=0.0001,
            verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/cv_resultados_5/fold_{fold + 1}",
            monitor='val_loss',
            mode='max',
            save_top_k=1,
            filename='best-{epoch:02d}-{val_loss:.4f}'
        )
        
        trainer = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger=fold_logger,
            callbacks=[early_stopping, checkpoint_callback], # Mudança V5
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )

        trainer.fit(l_model, datamodule=data_module)




if __name__ == "__main__":
    main()