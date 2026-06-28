import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import torchmetrics.classification as tm_class

class Classificador(nn.Module):
    def __init__(self,input_size ,n_neurons, n_hidden):
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

class ClassificadorV2(nn.Module):
    def __init__(self,input_size ,n_neurons, n_hidden):
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

class GOTYModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
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

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optim

class GOTYModelV2(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3,pos_weight_val=19):
        super().__init__()
        # Peso da classe positiva 
        self.register_buffer('pos_weight', torch.tensor([pos_weight_val]))
        self.model = model
        self.learning_rate = learning_rate
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
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_auroc', self.auroc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.f1, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=0.5,patience=5,min_lr=1e-6)
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", 
                "frequency": 1
            }
        }