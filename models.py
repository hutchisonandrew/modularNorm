import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import os
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from muon import Muon

class MLP(pl.LightningModule):
    def __init__(
        self, 
        input_dim=3072,  # CIFAR images are 32x32x3 = 3072 when flattened
        hidden_dims=[512, 256],
        num_classes=10,
        learning_rate=1e-3,
        weight_decay=1e-5,
        momentum=0.95,
        optimizer='AdamW'
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer # Renamed to avoid conflict, using self.optimizer from LightningModule for the actual optimizer object(s)
        self.momentum = momentum
        
        if self.optimizer_name == 'Muon':
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes, bias=True))
        
        self.layers = nn.Sequential(*layers)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x):
        # Flatten the input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        if not self.automatic_optimization:
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
            for opt in optimizers:
                opt.zero_grad()
            self.manual_backward(loss)
            for opt in optimizers:
                opt.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.val_acc(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        self.test_acc(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)
        
        return loss
    
    def configure_optimizers(self):
        if self.optimizer_name == 'AdamW':
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'Muon':
            muon_params = []
            adamw_params = []
            
            # Identify the head layer (last nn.Linear)
            head_layer = None
            for layer in reversed(list(self.layers.children())): # self.layers is nn.Sequential
                if isinstance(layer, nn.Linear):
                    head_layer = layer
                    break
            
            if head_layer is None:
                raise ValueError("No Linear layer found in the model")
    
            adamw_params.extend(list(head_layer.parameters()))

            # Separate body parameters
            for layer in self.layers.children():
                if isinstance(layer, nn.Linear) and layer is not head_layer:
                    for p in layer.parameters():
                        if p.ndim >= 2:
                            print(f"Adding parameter to Muon: {p.shape}")
                            muon_params.append(p)
                        else:
                            adamw_params.append(p)
                elif not isinstance(layer, nn.Linear) and not isinstance(layer, nn.ReLU): # Handle other layer types like BatchNorm, etc.
                    adamw_params.extend(list(layer.parameters()))

            optimizers = []
            for p in adamw_params:
                print(f"AdamW param: {p.shape}")
            if muon_params:
                # Muon doesn't directly use weight_decay in its constructor
                optimizers.append(Muon(muon_params, lr=self.learning_rate, momentum=self.momentum)) 
                
            if adamw_params:
                optimizers.append(torch.optim.AdamW(adamw_params, lr=self.learning_rate, betas=(0.90, 0.95), weight_decay=self.weight_decay))
            
            if not optimizers: # If no parameters were assigned (e.g. model with no Linear layers or only a head)
                # Fallback to AdamW for all parameters
                raise ValueError("No parameters were assigned to the optimizer")
            
            return optimizers



