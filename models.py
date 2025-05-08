import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from residuals import WeightPerturbationCallback

class MetricsCallback(Callback):
    """Callback to track and save metrics at the end of each epoch."""
    def __init__(self, save_dir='metrics'):
        super().__init__()
        self.save_dir = save_dir
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        
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
        # Save all metrics to files
        with open(os.path.join(self.save_dir, 'train_losses.txt'), 'w') as f:
            f.write('\n'.join([str(x) for x in self.train_losses]))
        
        with open(os.path.join(self.save_dir, 'train_accs.txt'), 'w') as f:
            f.write('\n'.join([str(x) for x in self.train_accs]))
        
        with open(os.path.join(self.save_dir, 'test_losses.txt'), 'w') as f:
            f.write('\n'.join([str(x) for x in self.test_losses]))
        
        with open(os.path.join(self.save_dir, 'test_accs.txt'), 'w') as f:
            f.write('\n'.join([str(x) for x in self.test_accs]))


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
        weight_decay=1e-5
    ):
        
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

def get_callbacks(checkpoint_dir='checkpoints', metrics_dir='metrics'):
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
    perturbation_callback = WeightPerturbationCallback(save_dir=f"{metrics_dir}/perturbation_analysis")
    return [metrics_callback, perturbation_callback, final_model_callback]

