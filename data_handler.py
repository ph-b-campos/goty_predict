import pandas as pd
import torch
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

class Trend_Feature_Eng(BaseEstimator, TransformerMixin):
    def __init__(self, trends_df):
        # A tabela trends é um parâmetro de inicialização, pois atua como um dicionário histórico
        self.trends_df = trends_df

        # Dicionário para armazenar as estatísticas extraídas do conjunto de treino
        self.stats = {}

        self.tipos_avaliacao = [('metacritic', 'critic'), ('user', 'user')]
        self.colunas_vendas_regionais = ['na_sales_million', 'eu_sales_million', 'jp_sales_million', 'other_sales_million']
        self.colunas_para_padronizar = [
            'metacritic_score', 'user_score', 'launch_price_usd',
            'how_long_to_beat_main_hrs', 'global_sales_million', 'estimated_revenue_million_usd'
        ]
        self.esrb_mapping = {'E': 1, 'E10+': 2, 'T': 3, 'M': 4, 'AO': 5, 'RP': 3}

    def fit(self, X, y=None):
        """
        O método fit calcula e armazena os parâmetros estatísticos usando APENAS os dados de treino.
        """
        # 1. Estatísticas para Suavização Bayesiana por ano
        self.stats['bayes'] = {}
        for tipo_nota, tipo_count in self.tipos_avaliacao:
            col_score = f'{tipo_nota}_score'
            col_count = f'{tipo_count}_review_count'

            self.stats['bayes'][col_score] = {
                'C': X.groupby('year')[col_score].mean().to_dict(),
                'm': X.groupby('year')[col_count].median().to_dict(),
                'C_global': X[col_score].mean(),
                'm_global': X[col_count].median()
            }

        # 2. Estatísticas para Z-Score por ano
        self.stats['zscore'] = {}
        for col in self.colunas_para_padronizar:
            self.stats['zscore'][col] = {
                'mean': X.groupby('year')[col].mean().to_dict(),
                'std': X.groupby('year')[col].std().replace(0, 1).to_dict(),
                'mean_global': X[col].mean(),
                'std_global': X[col].std() if X[col].std() != 0 else 1
            }

        # 3. Estatísticas para Saturação de Mercado (Global)
        # O merge temporário é necessário apenas para extrair a estatística global
        X_temp = X.merge(self.trends_df[['year', 'titles_released']], on='year', how='left')
        self.stats['market'] = {
            'mean_titulos': X_temp['titles_released'].mean(),
            'std_titulos': X_temp['titles_released'].std()
        }

        return self

    def transform(self, X):
        """
        O método transform aplica a engenharia aos dados (treino, teste ou validação).
        """
        X_out = X.copy()

        # 1. Suavização Bayesiana
        for tipo_nota, tipo_count in self.tipos_avaliacao:
            col_score = f'{tipo_nota}_score'
            col_count = f'{tipo_count}_review_count'

            # Mapeia os valores do treino. Se o ano não existir (unseen data), usa a média global
            C = X_out['year'].map(self.stats['bayes'][col_score]['C']).fillna(self.stats['bayes'][col_score]['C_global'])
            m = X_out['year'].map(self.stats['bayes'][col_score]['m']).fillna(self.stats['bayes'][col_score]['m_global'])

            v = X_out[col_count].fillna(0)
            R = X_out[col_score].fillna(C)

            peso_jogo = v / (v + m)
            peso_media = m / (v + m)
            X_out[col_score] = (peso_jogo * R) + (peso_media * C)

        # 2. Z-Score Temporal
        for col in self.colunas_para_padronizar:
            media_ano = X_out['year'].map(self.stats['zscore'][col]['mean']).fillna(self.stats['zscore'][col]['mean_global'])
            desvio_ano = X_out['year'].map(self.stats['zscore'][col]['std']).fillna(self.stats['zscore'][col]['std_global'])

            X_out[f'{col}_zscore'] = (X_out[col] - media_ano) / desvio_ano

        # 3. Merge com Trends e Desvios
        trends_cols = ['year', 'pct_microtransactions', 'pct_online', 'pct_dlc', 'titles_released', 'goty_games']
        X_out = X_out.merge(self.trends_df[trends_cols], on='year', how='left')

        X_out['dev_microtransactions'] = X_out['microtransactions'] - X_out['pct_microtransactions']
        X_out['dev_online'] = X_out['online_multiplayer'] - X_out['pct_online']
        X_out['dev_dlc'] = X_out['dlc_released'] - X_out['pct_dlc']

        # 4. Variáveis Macro
        X_out['goty_hit_rate'] = (X_out['goty_games'] / X_out['titles_released']).fillna(0)
        X_out['market_saturation_zscore'] = (X_out['titles_released'] - self.stats['market']['mean_titulos']) / self.stats['market']['std_titulos']

        # 5. Ordinal Encoding ESRB
        X_out['esrb_rating_num'] = X_out['esrb_rating'].map(self.esrb_mapping).fillna(0)

        # 6. Limpeza e Remoção de Colunas Redundantes
        colunas_avaliacao = ['critic_review_count', 'user_review_count']
        colunas_tendencias_brutas = ['microtransactions', 'pct_microtransactions', 'online_multiplayer', 'pct_online', 'dlc_released', 'pct_dlc']
        colunas_macro_brutas = ['year', 'titles_released', 'goty_games']

        colunas_redundantes = (
            self.colunas_vendas_regionais +
            self.colunas_para_padronizar +
            colunas_avaliacao +
            colunas_tendencias_brutas +
            colunas_macro_brutas +
            ['esrb_rating']
        )

        # O uso do errors='ignore' garante que a remoção não falhe caso a coluna já não exista
        X_out = X_out.drop(columns=colunas_redundantes, errors='ignore')

        return X_out


class GOTYDataModule(L.LightningDataModule):
    """
    DataModule para orquestrar datasets, preprocessamento e DataLoaders do GOTY.
    Projetado para receber índices de Cross-Validation (K-Fold).
    """
    def __init__(self, X, y, trends_df, train_idx, val_idx, batch_size=32, num_workers=0):
        super().__init__()
        # O reset_index garante que o fatiamento por .iloc nos folds funcione sem erros
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.trends_df = trends_df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.pipeline = Trend_Feature_Eng(trends_df=self.trends_df)

    def setup(self, stage=None):
        """
        Executado em cada GPU/Máquina (se houver paralelismo).
        Faz o split, o fit_transform no treino, e o transform na validação.
        """
        # 1. Separação dos dados do fold atual
        X_train = self.X.iloc[self.train_idx]
        y_train = self.y.iloc[self.train_idx]
        
        X_val = self.X.iloc[self.val_idx]
        y_val = self.y.iloc[self.val_idx]

        # 2. Prevenção de Data Leakage
        X_train_proc = self.pipeline.fit_transform(X_train)
        X_val_proc = self.pipeline.transform(X_val)

        # 3. Conversão para Tensores PyTorch
        X_train_tensor = torch.tensor(X_train_proc.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        
        X_val_tensor = torch.tensor(X_val_proc.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        # Salva o input_size (número de colunas finais) para instanciar a rede neural depois
        self.input_size = X_train_tensor.shape[1]

        # 4. Empacotamento em Datasets
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )