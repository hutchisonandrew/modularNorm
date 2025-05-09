import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import os
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from residuals import WeightPerturbationCallback
from muon import Muon

class MetricsCallback(Callback):
    """Callback to track and save metrics at the end of each epoch."""
    def __init__(self, save_dir='metrics'):
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the metrics
        train_loss = trainer.callback_metrics.get('train_loss')
        train_acc = trainer.callback_metrics.get('train_acc')
        
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_acc is not None:
            self.train_accs.append(train_acc.item())
        
        # Save metrics to file every epoch
        self._save_metrics()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get metrics
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_acc is not None:
            self.val_accs.append(val_acc.item())
        
        # Save metrics to file
        self._save_metrics()
    
    def on_test_epoch_end(self, trainer, pl_module):
        # Get metrics
        test_loss = trainer.callback_metrics.get('test_loss')
        test_acc = trainer.callback_metrics.get('test_acc')
        
        if test_loss is not None:
            self.test_losses.append(test_loss.item())
        if test_acc is not None:
            self.test_accs.append(test_acc.item())
        
        # Save metrics to file
        self._save_metrics()
    
    def _save_metrics(self):
        # Create a combined metrics DataFrame with all available data
        max_len = max(
            len(self.train_losses), len(self.train_accs),
            len(self.val_losses), len(self.val_accs),
            len(self.test_losses), len(self.test_accs)
        )
        
        # Create a DataFrame with epochs and metrics
        metrics_data = {
            'epoch': list(range(max_len)),
            'train_loss': self.train_losses + [None] * (max_len - len(self.train_losses)),
            'train_acc': self.train_accs + [None] * (max_len - len(self.train_accs)),
            'val_loss': self.val_losses + [None] * (max_len - len(self.val_losses)),
            'val_acc': self.val_accs + [None] * (max_len - len(self.val_accs)),
            'test_loss': self.test_losses + [None] * (max_len - len(self.test_losses)),
            'test_acc': self.test_accs + [None] * (max_len - len(self.test_accs))
        }
        
        # Save combined metrics to a single CSV file
        pd.DataFrame(metrics_data).to_csv(
            os.path.join(self.save_dir, 'metrics.csv'), index=False)


class SaveFinalModelCallback(Callback):
    """Callback to save the final model at the end of training."""
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def on_train_end(self, trainer, pl_module):
        # Save the final model when training is complete
        final_model_path = os.path.join(self.save_dir, 'final_model.pt')
        torch.save(pl_module.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")


    
            

class MLP(pl.LightningModule):
    def __init__(
        self, 
        input_dim=3072,  # CIFAR images are 32x32x3 = 3072 when flattened
        hidden_dims=[512, 256],
        num_classes=10,
        learning_rate=1e-3,
        weight_decay=1e-5,
        optimizer='AdamW'
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer # Renamed to avoid conflict, using self.optimizer from LightningModule for the actual optimizer object(s)
        
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
                            muon_params.append(p)
                        else:
                            adamw_params.append(p)
                elif not isinstance(layer, nn.Linear) and not isinstance(layer, nn.ReLU): # Handle other layer types like BatchNorm, etc.
                    adamw_params.extend(list(layer.parameters()))

            optimizers = []
            if muon_params:
                optimizers.append(Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=self.weight_decay, rank=0, world_size=1)) # Added weight_decay from self
            if adamw_params:
                optimizers.append(torch.optim.AdamW(adamw_params, lr=self.learning_rate, betas=(0.90, 0.95), weight_decay=self.weight_decay))
            
            if not optimizers: # If no parameters were assigned (e.g. model with no Linear layers or only a head)
                # Fallback to AdamW for all parameters
                raise ValueError("No parameters were assigned to the optimizer")
            
            return optimizers

def get_callbacks(checkpoint_dir='checkpoints', metrics_dir='metrics', num_magnitudes=10, magnitude_type='frobenius'):
    """Helper function to set up callbacks for training."""
    # Create checkpoint callback to save model weights every 5 epochs
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_dir,
    #     filename='model-{epoch:02d}',
    #     save_top_k=-1,  # Save all checkpoints
    #     every_n_epochs=5,  # Save every 5 epochs
    #     save_weights_only=True,
    # )
    
    # Create metrics callback
    metrics_callback = MetricsCallback(save_dir=metrics_dir)
    final_model_callback = SaveFinalModelCallback(save_dir=checkpoint_dir)
    perturbation_callback = WeightPerturbationCallback(num_magnitudes=num_magnitudes, magnitude_type=magnitude_type, save_dir=f"{metrics_dir}/perturbation_analysis")
    return [metrics_callback, final_model_callback]

