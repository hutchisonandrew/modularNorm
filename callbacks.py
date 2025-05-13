import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import os
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

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
        print("\nEntering SaveFinalModelCallback.on_train_end...") # Debug print
        # Save the final model when training is complete
        final_model_path = os.path.join(self.save_dir, 'final_model.pt')
        print(f"Attempting to save final model to: {final_model_path}") # Debug print
        try:
            torch.save(pl_module.state_dict(), final_model_path)
            print(f"Final model successfully saved to {final_model_path}") # Debug print
        except Exception as e:
            print(f"Error saving final model: {e}") # Debug print for errors

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
    return [metrics_callback, final_model_callback]