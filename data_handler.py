import pandas as pd
import numpy as np
import torch
import pytorch_lightning as L
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, TransformerMixin


class Feature_Eng(BaseEstimator, TransformerMixin):
    def __init__(self, trends_df):
        self.trends_df = trends_df
        self.stats = {}
        self.tipos_avaliacao = [('metacritic', 'critic'), ('user', 'user')]
        self.colunas_vendas_regionais = ['na_sales_million', 'eu_sales_million', 'jp_sales_million', 'other_sales_million']
        self.colunas_para_padronizar = [
            'metacritic_score', 'user_score', 'launch_price_usd',
            'how_long_to_beat_main_hrs', 'global_sales_million',
            'estimated_revenue_million_usd', 'how_long_to_beat_completionist_hrs'
        ]
        self.colunas_publisher_zscore = [
            'publisher_log_titles',
            'publisher_avg_metacritic',
            'publisher_avg_user_score',
            'publisher_log_avg_revenue',
            'metacritic_vs_publisher_avg',
            'user_score_vs_publisher_avg'
        ]
        self.esrb_mapping = {'E': 1, 'E10+': 2, 'T': 3, 'M': 4, 'AO': 5, 'RP': 3}
        self.publisher_tier_mapping = {'Indie': 0.0, 'AA': 0.5, 'AAA': 1.0}
        self.smoothing_alpha = 50.0

    def fit(self, X, y=None):
        """
        Aprende estatísticas usando apenas o fold de treino para evitar data leakage.
        """
        if y is None:
            raise ValueError("Feature_Eng precisa de y no fit para calcular o target encoding por gênero.")

        df = X.copy()
        df['_target'] = pd.Series(y, index=df.index)

        self.stats['bayes'] = self._fit_bayes_stats(df)
        self.stats['year_zscore'] = self._fit_group_zscore_stats(df, ['year'], self.colunas_para_padronizar)
        self.stats['genre_year_zscore'] = self._fit_group_zscore_stats(df, ['year', 'genre'], self.colunas_para_padronizar)
        self.stats['genre_counts'] = self._fit_genre_counts(df)
        self.stats['genre_target_rate'] = df.groupby('genre')['_target'].mean().to_dict()
        self.stats['global_target_rate'] = df['_target'].mean()
        self.stats['global_win_rate'] = df['goty_won'].mean()
        self.stats['market'] = self._fit_market_stats(df)
        self.stats['publisher'] = self._fit_publisher_stats(df)

        df_publisher_features = self._build_publisher_features(df)
        self.stats['publisher_zscore'] = self._fit_global_zscore_stats(
            df_publisher_features,
            self.colunas_publisher_zscore
        )

        return self

    def transform(self, X):
        """
        Aplica as features aprendidas no fit em treino, validação ou teste.
        """
        X_out = X.copy()

        X_out = self._apply_bayesian_smoothing(X_out)
        X_out = self._add_group_zscores(X_out, 'year_zscore', self.colunas_para_padronizar, 'zscore')
        X_out = self._add_trend_features(X_out)
        X_out = self._add_genre_features(X_out)
        X_out = self._add_publisher_features(X_out)

        X_out['esrb_rating_num'] = X_out['esrb_rating'].map(self.esrb_mapping).fillna(0)

        return self._drop_redundant_columns(X_out)

    def _fit_bayes_stats(self, df):
        stats = {}
        por_ano = df.groupby('year')

        for tipo_nota, tipo_count in self.tipos_avaliacao:
            col_score = f'{tipo_nota}_score'
            col_count = f'{tipo_count}_review_count'

            stats[col_score] = {
                'C': por_ano[col_score].mean().to_dict(),
                'm': por_ano[col_count].median().to_dict(),
                'C_global': df[col_score].mean(),
                'm_global': df[col_count].median()
            }

        return stats

    def _fit_group_zscore_stats(self, df, group_cols, columns):
        stats_df = df.groupby(group_cols)[columns].agg(['mean', 'std'])
        stats_df.columns = [f'{col}_{stat}' for col, stat in stats_df.columns]
        stats_df = stats_df.reset_index()

        for col in columns:
            stats_df[f'{col}_std'] = stats_df[f'{col}_std'].fillna(1).replace(0, 1)

        return {
            'group_cols': group_cols,
            'stats_df': stats_df,
            'global': self._fit_global_zscore_stats(df, columns)
        }

    def _fit_global_zscore_stats(self, df, columns):
        stats = {}
        for col in columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': self._safe_std(df[col].std())
            }
        return stats

    def _fit_genre_counts(self, df):
        return (
            df.groupby(['year', 'genre'])
            .size()
            .reset_index(name='qtd_jogos')
        )

    def _fit_market_stats(self, df):
        df_trends = df.merge(self.trends_df[['year', 'titles_released']], on='year', how='left')

        return {
            'mean_titles': df_trends['titles_released'].mean(),
            'std_titles': self._safe_std(df_trends['titles_released'].std())
        }

    def _fit_publisher_stats(self, df):
        grouped = df.groupby('publisher')
        stats_df = grouped.agg(
            publisher_titles=('publisher', 'size'),
            publisher_avg_metacritic=('metacritic_score', 'mean'),
            publisher_avg_user_score=('user_score', 'mean'),
            publisher_avg_revenue=('estimated_revenue_million_usd', 'mean'),
            publisher_pct_sequels=('is_sequel', 'mean'),
            publisher_goty_nom_count=('_target', 'sum'),
            publisher_goty_win_count=('goty_won', 'sum')
        ).reset_index()

        stats_df['publisher_log_titles'] = np.log1p(stats_df['publisher_titles'])
        stats_df['publisher_log_avg_revenue'] = np.log1p(stats_df['publisher_avg_revenue'].clip(lower=0))
        stats_df['publisher_smoothed_goty_nom_rate'] = (
            stats_df['publisher_goty_nom_count'] + self.smoothing_alpha * self.stats['global_target_rate']
        ) / (stats_df['publisher_titles'] + self.smoothing_alpha)
        stats_df['publisher_smoothed_goty_win_rate'] = (
            stats_df['publisher_goty_win_count'] + self.smoothing_alpha * self.stats['global_win_rate']
        ) / (stats_df['publisher_titles'] + self.smoothing_alpha)

        defaults = {
            'publisher_log_titles': np.log1p(df.groupby('publisher').size().mean()),
            'publisher_avg_metacritic': df['metacritic_score'].mean(),
            'publisher_avg_user_score': df['user_score'].mean(),
            'publisher_avg_revenue': df['estimated_revenue_million_usd'].mean(),
            'publisher_log_avg_revenue': np.log1p(max(df['estimated_revenue_million_usd'].mean(), 0)),
            'publisher_pct_sequels': df['is_sequel'].mean(),
            'publisher_smoothed_goty_nom_rate': self.stats['global_target_rate'],
            'publisher_smoothed_goty_win_rate': self.stats['global_win_rate']
        }

        keep_cols = ['publisher'] + list(defaults.keys())
        return {
            'stats_df': stats_df[keep_cols],
            'defaults': defaults
        }

    def _safe_std(self, std):
        if pd.isna(std) or std == 0:
            return 1
        return std

    def _zscore(self, values, mean, std):
        return (values - mean) / std

    def _apply_bayesian_smoothing(self, X_out):
        for tipo_nota, tipo_count in self.tipos_avaliacao:
            col_score = f'{tipo_nota}_score'
            col_count = f'{tipo_count}_review_count'

            C = X_out['year'].map(self.stats['bayes'][col_score]['C']).fillna(self.stats['bayes'][col_score]['C_global'])
            m = X_out['year'].map(self.stats['bayes'][col_score]['m']).fillna(self.stats['bayes'][col_score]['m_global'])

            v = X_out[col_count].fillna(0)
            R = X_out[col_score].fillna(C)

            denominador = (v + m).replace(0, 1)
            peso_jogo = v / denominador
            peso_media = m / denominador
            X_out[col_score] = (peso_jogo * R) + (peso_media * C)

        return X_out

    def _add_group_zscores(self, X_out, stats_key, columns, suffix):
        stats = self.stats[stats_key]
        X_out = X_out.merge(stats['stats_df'], on=stats['group_cols'], how='left')
        temp_cols = []

        for col in columns:
            col_mean = f'{col}_mean'
            col_std = f'{col}_std'
            temp_cols.extend([col_mean, col_std])

            media = X_out[col_mean].fillna(stats['global'][col]['mean'])
            desvio = X_out[col_std].fillna(stats['global'][col]['std']).replace(0, 1)
            X_out[f'{col}_{suffix}'] = self._zscore(X_out[col], media, desvio)

        return X_out.drop(columns=temp_cols, errors='ignore')

    def _add_global_zscores(self, X_out, stats_key, columns):
        stats = self.stats[stats_key]
        for col in columns:
            X_out[f'{col}_zscore'] = self._zscore(X_out[col], stats[col]['mean'], stats[col]['std'])

        return X_out

    def _add_trend_features(self, X_out):
        trends_cols = ['year', 'pct_microtransactions', 'pct_online', 'pct_dlc', 'titles_released', 'goty_games']
        X_out = X_out.merge(self.trends_df[trends_cols], on='year', how='left')

        X_out['dev_microtransactions'] = X_out['microtransactions'] - X_out['pct_microtransactions']
        X_out['dev_online'] = X_out['online_multiplayer'] - X_out['pct_online']
        X_out['dev_dlc'] = X_out['dlc_released'] - X_out['pct_dlc']

        X_out['goty_hit_rate'] = (X_out['goty_games'] / X_out['titles_released']).fillna(0)
        X_out['market_saturation_zscore'] = (
            self._zscore(
                X_out['titles_released'],
                self.stats['market']['mean_titles'],
                self.stats['market']['std_titles']
            )
        )

        return X_out

    def _add_genre_features(self, X_out):
        X_out['pct_goty_nominated'] = (
            X_out['genre']
            .map(self.stats['genre_target_rate'])
            .fillna(self.stats['global_target_rate'])
        )

        X_out = X_out.merge(self.stats['genre_counts'], on=['year', 'genre'], how='left')
        X_out['qtd_jogos'] = X_out['qtd_jogos'].fillna(1)
        X_out['genre_scarcity'] = 1.0 / X_out['qtd_jogos']
        X_out = self._add_group_zscores(X_out, 'genre_year_zscore', self.colunas_para_padronizar, 'zscore_gy')

        return X_out

    def _build_publisher_features(self, X_out):
        publisher_stats = self.stats['publisher']
        X_out = X_out.merge(publisher_stats['stats_df'], on='publisher', how='left')

        for col, default in publisher_stats['defaults'].items():
            X_out[col] = X_out[col].fillna(default)

        X_out['metacritic_vs_publisher_avg'] = X_out['metacritic_score'] - X_out['publisher_avg_metacritic']
        X_out['user_score_vs_publisher_avg'] = X_out['user_score'] - X_out['publisher_avg_user_score']
        X_out['publisher_tier_encoded'] = X_out['publisher_tier'].map(self.publisher_tier_mapping).fillna(0.0)

        return X_out

    def _add_publisher_features(self, X_out):
        X_out = self._build_publisher_features(X_out)
        X_out = self._add_global_zscores(X_out, 'publisher_zscore', self.colunas_publisher_zscore)

        return X_out

    def _drop_redundant_columns(self, X_out):
        colunas_redundantes = (
            self.colunas_vendas_regionais +
            self.colunas_para_padronizar +
            ['goty_won', 'critic_review_count', 'user_review_count'] +
            ['microtransactions', 'pct_microtransactions',
             'online_multiplayer', 'pct_online', 'dlc_released', 'pct_dlc'] +
            ['year', 'titles_released', 'goty_games', 'genre', 'qtd_jogos', 'esrb_rating'] +
            ['publisher', 'publisher_tier', 'publisher_region'] +
            self.colunas_publisher_zscore +
            ['publisher_avg_revenue']
        )

        return X_out.drop(columns=colunas_redundantes, errors='ignore')


class GOTYDataModule(L.LightningDataModule):
    """
    DataModule para orquestrar datasets, preprocessamento e DataLoaders do GOTY.
    Projetado para receber índices de Cross-Validation (K-Fold).
    """
    def __init__(self, X, y, trends_df, train_idx, val_idx, batch_size=32, num_workers=9):
        super().__init__()
        # O reset_index garante que o fatiamento por .iloc nos folds funcione sem erros
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.trends_df = trends_df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.pipeline = Feature_Eng(trends_df=self.trends_df)

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
        X_train_proc = self.pipeline.fit_transform(X_train, y_train)
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