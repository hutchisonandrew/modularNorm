import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from models import MLP

def frob_norm_residual(w, delta_w):
    """Calculate Frobenius norm of residual between w and w+delta_w"""
    return torch.norm(delta_w, p='fro')


def compute_loss_with_perturbed_weights(model, dataloader, layer_name, delta_w, device='cpu'):
    """
    Compute loss with perturbed weights for a specific layer.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for dataset
        layer_name: Name of the layer to perturb (e.g., 'layers.0' for first layer in MLP)
        delta_w: Weight perturbation tensor (same shape as layer weights)
        device: Device to run computation on
        
    Returns:
        tuple: (original_loss, perturbed_loss)
    """
    model.eval()
    model = model.to(device)
    
    # Find the target layer
    layer = None
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            layer = module
            break
    
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Ensure delta_w has same shape as layer weights
    if delta_w.shape != layer.weight.shape:
        raise ValueError(f"delta_w shape {delta_w.shape} does not match layer weight shape {layer.weight.shape}")
    
    # Compute original loss
    original_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            original_loss += loss.item() * x.size(0)
            count += x.size(0)
    
    original_loss /= count
    
    # Store original weights
    original_weights = layer.weight.data.clone()
    
    # Apply perturbation
    layer.weight.data = original_weights + delta_w.to(device)
    
    # Compute perturbed loss
    perturbed_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            perturbed_loss += loss.item() * x.size(0)
            count += x.size(0)
    
    perturbed_loss /= count
    
    # Restore original weights
    layer.weight.data = original_weights
    
    return original_loss, perturbed_loss


class WeightPerturbationCallback(Callback):
    """
    Callback to analyze the effect of weight perturbations on loss at the end of each epoch.
    """
    def __init__(self, layer_names, delta_magnitudes=[0.01, 0.1, 1.0], save_dir='perturbation_analysis'):
        """
        Args:
            layer_names: List of layer names to analyze (e.g., ['layers.0', 'layers.2'])
            delta_magnitudes: Magnitudes of perturbations to apply
            save_dir: Directory to save results
        """
        super().__init__()
        self.layer_names = layer_names
        self.delta_magnitudes = delta_magnitudes
        self.save_dir = save_dir
        self.results = {}
        
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Analyze weight perturbations at the end of each epoch"""
        epoch = trainer.current_epoch
        device = pl_module.device
        
        # Dictionary to store results for this epoch
        epoch_results = {'train': {}, 'test': {}}
        
        # Analyze each layer
        for layer_name in self.layer_names:
            # Find the target layer
            layer = None
            for name, module in pl_module.named_modules():
                if name == layer_name and hasattr(module, 'weight'):
                    layer = module
                    break
            
            if layer is None:
                print(f"Warning: Layer {layer_name} not found in model")
                continue
            
            # For each perturbation magnitude
            for magnitude in self.delta_magnitudes:
                # Generate random perturbation with specified magnitude
                delta_w = torch.randn_like(layer.weight.data)
                # Normalize and scale to desired magnitude
                delta_w = delta_w * magnitude / torch.norm(delta_w, p='fro')
                
                # Compute loss on train set
                train_original, train_perturbed = compute_loss_with_perturbed_weights(
                    pl_module, trainer.train_dataloader, layer_name, delta_w, device
                )
                
                # Compute loss on test set
                test_original, test_perturbed = compute_loss_with_perturbed_weights(
                    pl_module, trainer.test_dataloaders[0], layer_name, delta_w, device
                )
                
                # Store results
                if layer_name not in epoch_results['train']:
                    epoch_results['train'][layer_name] = []
                    epoch_results['test'][layer_name] = []
                
                epoch_results['train'][layer_name].append({
                    'magnitude': magnitude,
                    'original_loss': train_original,
                    'perturbed_loss': train_perturbed,
                    'difference': train_perturbed - train_original
                })
                
                epoch_results['test'][layer_name].append({
                    'magnitude': magnitude,
                    'original_loss': test_original,
                    'perturbed_loss': test_perturbed,
                    'difference': test_perturbed - test_original
                })
        
        # Store results for this epoch
        self.results[epoch] = epoch_results
        
        # Save results
        import json
        with open(f"{self.save_dir}/perturbation_analysis_epoch_{epoch}.json", 'w') as f:
            json.dump(epoch_results, f, indent=2)
        
        print(f"Completed weight perturbation analysis for epoch {epoch}")


if __name__ == "__main__":
    # Load model
    model = MLP.load_from_checkpoint('runs/my_experiment/model-epoch=00.ckpt')
    # Load data
    train_dataloader = model.train_dataloader()
    test_dataloader = model.test_dataloader()
