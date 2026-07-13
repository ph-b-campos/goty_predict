import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics.classification as tm_class
from torchmetrics.functional.classification import binary_auroc, binary_average_precision, binary_f1_score, binary_precision_recall_curve
import config as cfg

#############################################################################################
# --------------------Primeira Iteração do modelo------------------------------
#############################################################################################
class Classificador(nn.Module):
    def __init__(self,input_size ,n_neurons = cfg.N_NEURONS, n_hidden = cfg.N_HIDDEN):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden

        self.first_layer = self.dense_layer(input_size, n_neurons)

        hidden_layers_list = [self.dense_layer(n_neurons, n_neurons) for _ in range(n_hidden)]
        self.hidden_layers = nn.Sequential(*hidden_layers_list)

        self.output_layer = nn.Sequential( nn.Linear(n_neurons, 1),nn.Sigmoid())

    def dense_layer(self, n_input, n_output):
        layer = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        out = self.first_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out

class GOTYModel(L.LightningModule):
    def __init__(self, model, learning_rate=cfg.LR):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.auroc = tm_class.BinaryAUROC()
        self.f1 = tm_class.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # squeeze para alinhar as dimensões [batch_size, 1] -> [batch_size]
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # squeeze para alinhar as dimensões [batch_size, 1] -> [batch_size]
        y_pred = self.forward(x).squeeze(-1)

        loss = self.criterion(y_pred, y.float())

        self.auroc(y_pred, y)
        self.f1(y_pred, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_f1', self.f1, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optim
#############################################################################################
# --------------------Segunda Iteração do modelo------------------------------
# Adicionando o peso da classe positiva na função de perda para lidar com o desbalanceamento de classes
#############################################################################################
class ClassificadorV2(nn.Module):
    def __init__(self,input_size ,n_neurons = cfg.N_NEURONS, n_hidden = cfg.N_HIDDEN, dropout_rate=0.5):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden

        self.first_layer = self.dense_layer(input_size, n_neurons)

        hidden_layers_list = [self.dense_layer(n_neurons, n_neurons) for _ in range(n_hidden)]
        self.hidden_layers = nn.Sequential(*hidden_layers_list)

        self.output_layer = nn.Linear(n_neurons, 1) # não usar sigmoide na camada de saída

    def dense_layer(self, n_input, n_output):
        layer = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        out = self.first_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out
    
class GOTYModelV2(L.LightningModule):
    def __init__(self, model, learning_rate=cfg.LR, pos_weight_val=cfg.POS_WEIGHT_VAL, scheduler_t_max=cfg.MAX_EPOCHS):
        super().__init__()
        # Peso da classe positiva 
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler_t_max = scheduler_t_max
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.auroc = tm_class.BinaryAUROC()
        self.f1 = tm_class.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # squeeze para alinhar as dimensões [batch_size, 1] -> [batch_size]
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_logits = self.forward(x).squeeze(-1)
        
        loss = self.criterion(y_pred_logits, y.float())
        
        preds_prob = torch.sigmoid(y_pred_logits)
        
        self.auroc(preds_prob, y)
        self.f1(preds_prob, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_f1', self.f1, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        return loss
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optim
#############################################################################################
# --------------------Terceira Iteração do modelo------------------------------
# Adicionando dropout para regularização e evitar overfitting e removendo o scheduler de aprendizado
#############################################################################################
class ClassificadorV3(nn.Module):
    def __init__(self,input_size ,n_neurons = cfg.N_NEURONS, n_hidden = cfg.N_HIDDEN,dropout_rate=0.5):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.first_layer = self.dense_layer(input_size, n_neurons)

        hidden_layers_list = [self.dense_layer(n_neurons, n_neurons) for _ in range(n_hidden)]
        self.hidden_layers = nn.Sequential(*hidden_layers_list)

        self.output_layer = nn.Linear(n_neurons, 1) # não usar sigmoide na camada de saída

    def dense_layer(self, n_input, n_output):
        layer = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.Dropout(p=self.dropout_rate),  # Adicionando dropout para regularização
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        out = self.first_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out




class GOTYModelV3(L.LightningModule):
    def __init__(self, model, learning_rate=cfg.LR, pos_weight_val=cfg.POS_WEIGHT_VAL, scheduler_t_max=cfg.MAX_EPOCHS):
        super().__init__()
        # Peso da classe positiva 
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.model = model
        self.learning_rate = learning_rate
        self.scheduler_t_max = scheduler_t_max
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.auroc = tm_class.BinaryAUROC()
        self.f1 = tm_class.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # squeeze para alinhar as dimensões [batch_size, 1] -> [batch_size]
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_logits = self.forward(x).squeeze(-1)
        
        loss = self.criterion(y_pred_logits, y.float())
        
        preds_prob = torch.sigmoid(y_pred_logits)
        
        self.auroc(preds_prob, y)
        self.f1(preds_prob, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        self.log('val_f1', self.f1, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        return loss
         
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optim
    
#############################################################################################
# --------------------Quarta Iteração do modelo------------------------------
# Ajustando o threshold com base no F1 de validação e adicionando novas métricas
#############################################################################################

class GOTYModelV4(L.LightningModule):
    def __init__(self, model, learning_rate=cfg.LR, pos_weight_val=cfg.POS_WEIGHT_VAL):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val], dtype=torch.float32))
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Mudança V4: armazenamento das probabilidades e labels de toda a validação
        self.validation_probabilities = []
        self.validation_targets = []
        self.best_validation_threshold = 0.5

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred, y.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=cfg.BATCH_SIZE)
        return loss

    def on_validation_epoch_start(self):
        # Mudança V4: limpar as previsões armazenadas no início de cada época
        self.validation_probabilities.clear()
        self.validation_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_logits = self.forward(x).squeeze(-1)
        loss = self.criterion(y_pred_logits, y.float())
        preds_prob = torch.sigmoid(y_pred_logits)

        # Mudança V4: armazenar previsões para calcular métricas sobre toda a validação
        self.validation_probabilities.append(preds_prob.detach())
        self.validation_targets.append(y.int().detach())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cfg.BATCH_SIZE)
        return loss

    def on_validation_epoch_end(self):
        probabilities = torch.cat(self.validation_probabilities)
        targets = torch.cat(self.validation_targets)

        auroc = binary_auroc(probabilities, targets)
        average_precision = binary_average_precision(probabilities, targets)

        f1_threshold_05 = binary_f1_score(probabilities, targets, threshold=0.5)

        precision_curve, recall_curve, thresholds = binary_precision_recall_curve(probabilities, targets, thresholds=201)
        precision_curve = torch.nan_to_num(precision_curve[:-1], nan=0.0)
        recall_curve = torch.nan_to_num(recall_curve[:-1], nan=0.0)
        f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve).clamp_min(1e-8)

        best_index = torch.argmax(f1_curve)
        best_threshold = thresholds[best_index]
        self.best_validation_threshold = best_threshold.item()

        predicted_classes = (probabilities >= best_threshold).int()

        tp = ((predicted_classes == 1) & (targets == 1)).sum().float()
        tn = ((predicted_classes == 0) & (targets == 0)).sum().float()
        fp = ((predicted_classes == 1) & (targets == 0)).sum().float()
        fn = ((predicted_classes == 0) & (targets == 1)).sum().float()

        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        specificity = tn / (tn + fp).clamp_min(1.0)
        accuracy = (tp + tn) / (tp + tn + fp + fn).clamp_min(1.0)
        balanced_accuracy = (recall + specificity) / 2
        f1_optimized = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        self.log_dict({
            'val_auroc': auroc,
            'val_average_precision': average_precision,
            'val_f1_threshold_05': f1_threshold_05,
            'val_best_threshold': best_threshold,
            'val_f1_optimized': f1_optimized,
            'val_precision_optimized': precision,
            'val_recall_optimized': recall,
            'val_specificity_optimized': specificity,
            'val_accuracy_optimized': accuracy,
            'val_balanced_accuracy_optimized': balanced_accuracy,
            'val_true_positives': tp,
            'val_true_negatives': tn,
            'val_false_positives': fp,
            'val_false_negatives': fn
        }, logger=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optim
#############################################################################################
# --------------------Quinta Iteração do modelo------------------------------
# Aumentando a complexidade do modelo, adicionando mais camadas e neurônios, e ajustando o peso da classe positiva
#############################################################################################
class ClassificadorV5(nn.Module):
    def __init__(self, input_size, n_neurons=cfg.N_NEURONS, n_hidden=cfg.N_HIDDEN, dropout_rate=0.5):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate

        self.first_layer = self.dense_layer(input_size, n_neurons)

        hidden_layers_list = [self.dense_layer(n_neurons, n_neurons) for _ in range(n_hidden)]
        self.hidden_layers = nn.Sequential(*hidden_layers_list)

        self.output_layer = nn.Linear(n_neurons, 1)

    def dense_layer(self, n_input, n_output):
        layer = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.BatchNorm1d(n_output),
            nn.Dropout(p=self.dropout_rate),
            nn.GELU() # substituindo ReLU por GELU
        )
        return layer

    def forward(self, x):
        out = self.first_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out
    