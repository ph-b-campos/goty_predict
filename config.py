# Hiperparâmetros da Rede Neural
N_NEURONS = 32
N_HIDDEN = 1
LR = 1e-3
POS_WEIGHT_VAL = 5.0
TRESHOLD = 0.83

# Configurações de Treinamento
BATCH_SIZE = 512
MAX_EPOCHS = 50

# Configurações de Dados
COLUNAS_DROP = ['goty_nominated','game_id','title',
    'platform', 'platform_type', 'platform_maker',
    'platform_generation', 'developer'
]

#Sampling
SAMPLING_STRATEGY = 'upsample_positive'