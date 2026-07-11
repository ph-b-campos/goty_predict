
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def imprimir_metricas_finais(df_stats):
    ultima = df_stats.iloc[-1]
    epoca = int(ultima['epoch'])

    print("="*55)
    print(f"Relatório - 10 Folds")
    print("="*55)

    if 'val_auroc_mean' in df_stats.columns:
        print(f"AUROC            : {ultima['val_auroc_mean']:.4f} ± {ultima['val_auroc_std']:.4f}")
    if 'val_f1_mean' in df_stats.columns:
        print(f"F1-Score         : {ultima['val_f1_mean']:.4f} ± {ultima['val_f1_std']:.4f}")
    print("="*55)


def plotar_loss_cv(df_stats):
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))

    epocas = df_stats['epoch']

    plt.fill_between(epocas,
                     df_stats['train_loss_mean'] - df_stats['train_loss_std'],
                     df_stats['train_loss_mean'] + df_stats['train_loss_std'],
                     alpha=0.2, color='#1f77b4')
    plt.plot(epocas, df_stats['train_loss_mean'], label='Treino (Média)', linewidth=2.5, color='#1f77b4')

    plt.fill_between(epocas,
                     df_stats['val_loss_mean'] - df_stats['val_loss_std'],
                     df_stats['val_loss_mean'] + df_stats['val_loss_std'],
                     alpha=0.2, color='#d62728')
    plt.plot(epocas, df_stats['val_loss_mean'], label='Validação (Média)', linewidth=2.5, color='#d62728', linestyle='--')

    plt.title('Evolução do Erro (Loss) - Média ± Desvio Padrão', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Binary Cross Entropy Loss', fontsize=12)
    plt.legend(fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.show()

def plotar_metricas_cv(df_stats):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epocas = df_stats['epoch']

    axes[0].fill_between(epocas,
                         df_stats['val_auroc_mean'] - df_stats['val_auroc_std'],
                         df_stats['val_auroc_mean'] + df_stats['val_auroc_std'],
                         alpha=0.2, color='#9467bd')
    axes[0].plot(epocas, df_stats['val_auroc_mean'], label='AUROC (Média)', linewidth=2.5, color='#9467bd')
    axes[0].set_title('Desempenho: AUROC', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Épocas', fontsize=12)
    axes[0].set_ylabel('Score (0.0 a 1.0)', fontsize=12)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].legend(fontsize=11)

    axes[1].fill_between(epocas,
                         df_stats['val_f1_mean'] - df_stats['val_f1_std'],
                         df_stats['val_f1_mean'] + df_stats['val_f1_std'],
                         alpha=0.2, color='#2ca02c')
    axes[1].plot(epocas, df_stats['val_f1_mean'], label='F1-Score (Média)', linewidth=2.5, color='#2ca02c')
    axes[1].set_title('Desempenho: F1-Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Épocas', fontsize=12)
    axes[1].set_ylabel('Score (0.0 a 1.0)', fontsize=12)
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend(fontsize=11)

    sns.despine()
    plt.tight_layout()
    plt.show()

