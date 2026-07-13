import pandas as pd
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
import os

from models import Classificador, ClassificadorV2, ClassificadorV3, GOTYModel, GOTYModelV2, GOTYModelV3, GOTYModelV4
from data_handler import GOTYDataModule
import config as cfg

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
        print(f"🚀 INICIANDO FOLD {fold + 1}/10")
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

        modelo_base = Classificador(
            input_size=data_module.input_size, 
            n_neurons=3, 
            n_hidden=32
        )
        
        l_model= GOTYModel(
            model=modelo_base, 
            learning_rate=cfg.LR 
        )

        fold_logger = CSVLogger(
            save_dir="logs", 
            name="cv_resultados", 
            version=f"fold_{fold + 1}"
        )
        
        trainer = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger= fold_logger,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )

        trainer.fit(l_model, datamodule=data_module)
        # V2
        modelo_v2 = ClassificadorV2(
            input_size=data_module.input_size, 
            n_neurons=3, 
            n_hidden=32
        )
        
        l_model_v2= GOTYModelV2(
            model=modelo_v2, 
            learning_rate=cfg.LR, 
            pos_weight_val=20
        )

        fold_logger_v2 = CSVLogger(
            save_dir="logs", 
            name="cv_resultados_v2", 
            version=f"fold_{fold + 1}"
        )
        
        trainer_v2 = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger= fold_logger_v2,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )

        trainer_v2.fit(l_model_v2, datamodule=data_module)
        # V3
        modelo_v3 = ClassificadorV3(
            input_size=data_module.input_size, 
            n_neurons=cfg.N_NEURONS, 
            n_hidden=cfg.N_HIDDEN
        )
        
        l_model_v3= GOTYModelV3(
            model=modelo_v3, 
            learning_rate=cfg.LR, 
            pos_weight_val=20
        )

        fold_logger_v3 = CSVLogger(
            save_dir="logs", 
            name="cv_resultados_v3", 
            version=f"fold_{fold + 1}"
        )
        
        trainer_v3 = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger= fold_logger_v3,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )
        trainer_v3.fit(l_model_v3, datamodule=data_module)
        # V4
        modelo_v4 = ClassificadorV3(
            input_size=data_module.input_size, 
            n_neurons=cfg.N_NEURONS, 
            n_hidden=cfg.N_HIDDEN
        )
        l_model_v4 = GOTYModelV4(
            model=modelo_v4,
            learning_rate=cfg.LR,
            pos_weight_val=cfg.POS_WEIGHT_VAL
        )
        fold_logger_v4 = CSVLogger(
            save_dir="logs", 
            name="cv_resultados_v4", 
            version=f"fold_{fold + 1}"
        )
        trainer_v4 = L.Trainer(
            max_epochs=cfg.MAX_EPOCHS,
            accelerator="auto",
            devices=1,
            logger= fold_logger_v4,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True
        )
        trainer_v4.fit(l_model_v4, datamodule=data_module)
        

if __name__ == "__main__":
    main()