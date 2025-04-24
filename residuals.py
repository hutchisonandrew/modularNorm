import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from models import MLP
from CIFAR100_dataset import CIFAR10DataModule, CIFAR100DataModule


def compute_residuals(model, dataloader, layer_name, delta_w, device='cpu', max_batches=None):
    """
    Compute loss with perturbed weights for a specific layer and compute gradients.
    Returns per-batch results rather than aggregated values.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for dataset
        layer_name: Name of the layer to perturb (e.g., 'layers.0' for first layer in MLP)
        delta_w: Weight perturbation tensor (same shape as layer weights)
        device: Device to run computation on
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        tuple: (original_losses, original_grads, perturbed_losses)
            - original_losses: List of loss values for each batch with original weights
            - original_grads: List of gradients for each batch with original weights
            - perturbed_losses: List of loss values for each batch with perturbed weights
    """
    model.eval()  # Set model to evaluation mode but keep gradients enabled
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
    
    # Store original weights
    original_weights = layer.weight.data.clone()
    # Lists to store per-batch results
    original_losses = []
    first_order_terms = []
    perturbed_losses = []
    
    residuals = []
    # Process batches
    batch_count = 0
    
    # First pass: compute original losses and gradients
    for batch in dataloader:
        # Check batch limit if specified
        if max_batches is not None and batch_count >= max_batches:
            break
            
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        outputs = model(x)
        loss = nn.functional.cross_entropy(outputs, y)
        
        # Store original loss
        original_losses.append(loss.item())
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Store gradient for this layer
        gradient = layer.weight.grad.clone()
        first_order_term = torch.sum(gradient * delta_w)  
        first_order_terms.append(first_order_term.item())
        
        # Increment batch counter
        batch_count += 1
    
    # Restore model to clean state
    model.zero_grad()
    
    # Apply perturbation
    layer.weight.data = original_weights + delta_w.to(device)
    
    # Second pass: compute perturbed losses
    batch_count = 0
    with torch.no_grad():  # No need for gradients in perturbed case
        for batch in dataloader:
            # Check batch limit if specified
            if max_batches is not None and batch_count >= max_batches:
                break
                
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            
            # Store perturbed loss
            perturbed_losses.append(loss.item())
            
            # Increment batch counter
            batch_count += 1
    
    # Restore original weights
    layer.weight.data = original_weights
    
    return original_losses, first_order_terms, perturbed_losses


def residual_equation(original_losses, first_order_terms, perturbed_losses, norm):
    residuals = []
    for i in range(len(original_losses)):
        lhs = perturbed_losses[i]
        rhs = original_losses[i] + first_order_terms[i] + norm
        residuals.append(lhs - rhs)
        
    return residuals
    
class WeightPerturbationCallback(Callback):
    """
    Callback to analyze the effect of weight perturbations on loss at the end of each epoch.
    """
    def __init__(self, layer_names, delta_magnitudes=[0.01, 0.1, 1.0], save_dir='perturbation_analysis', max_batches=10):
        """
        Args:
            layer_names: List of layer names to analyze (e.g., ['layers.0', 'layers.2'])
            delta_magnitudes: Magnitudes of perturbations to apply
            save_dir: Directory to save results
            max_batches: Maximum number of batches to analyze (for efficiency)
        """
        super().__init__()
        self.layer_names = layer_names
        self.delta_magnitudes = delta_magnitudes
        self.save_dir = save_dir
        self.max_batches = max_batches
        self.results = {}
        
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Analyze weight perturbations at the end of each epoch"""
        epoch = trainer.current_epoch
        device = pl_module.device
        
        # Dictionary to store results for this epoch
        epoch_results = {}
        
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
                
                # Compute batch-wise losses and gradients on train set
                train_orig_losses, train_grads, train_pert_losses = compute_loss_with_perturbed_weights(
                    pl_module, trainer.train_dataloader, layer_name, delta_w, device,
                    max_batches=self.max_batches
                )
                
                # Store results
                if layer_name not in epoch_results:
                    epoch_results[layer_name] = []
                
                # Simply store the original and perturbed losses
                result = {
                    'magnitude': magnitude,
                    'original_losses': train_orig_losses,
                    'original_grads': train_grads,
                    'perturbed_losses': train_pert_losses
                }
                
                epoch_results[layer_name].append(result)
        
        # Store results for this epoch
        self.results[epoch] = epoch_results
        
        print(f"Completed weight perturbation analysis for epoch {epoch}")


if __name__ == "__main__":
    # Load model
    model_path = 'runs/my_experiment/logs/version_0/checkpoints/epoch=0-step=782.ckpt'
    model = MLP.load_from_checkpoint(model_path)
    
    # Get model parameters to determine dataset
    num_classes = model.layers[-1].out_features  # Last layer output size indicates dataset
    
    # Create appropriate data module
    batch_size = 512
    num_workers = 4
    if num_classes == 100:
        data_module = CIFAR100DataModule(num_workers=num_workers, batch_size=batch_size)
    else:
        data_module = CIFAR10DataModule(num_workers=num_workers, batch_size=batch_size)
    
    # Setup data module
    data_module.prepare_data()
    data_module.setup(stage=None)
    
    # Get data loaders
    train_dataloader = data_module.train_dataloader()
    
    # Print model layers
    print("Layers with weights:")
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(f"{name}: {type(module).__name__}, Shape: {module.weight.shape}")
    
    # Example of computing per-batch losses and gradients
    layer_name = 'layers.6'  # Example - output layer
    delta_w = torch.randn_like(model.layers[6].weight.data)
    original_norm = torch.norm(delta_w, p='fro')
    print(f"Original perturbation norm: {original_norm:.4f}")
    
    # Scale perturbation to desired magnitude
    perturbation_magnitude = 200
    delta_w = delta_w * perturbation_magnitude / original_norm
    print(f"Scaled perturbation norm: {torch.norm(delta_w, p='fro'):.4f}")
    
    # Limit to first few batches for demonstration
    max_batches = 3
    
    # Compute batch-wise losses, gradients, and perturbed losses
    orig_losses, grads, pert_losses = compute_loss_with_perturbed_weights(
        model, train_dataloader, layer_name, delta_w, max_batches=max_batches
    )
    
    # Print results for each batch
    print(f"\nLayer: {layer_name}")
    print(f"Perturbation magnitude: {perturbation_magnitude}")
    
    for i, (orig_loss, grad, pert_loss) in enumerate(zip(orig_losses, grads, pert_losses)):
        print(f"\nBatch {i+1}:")
        print(f"  Original loss: {orig_loss:.4f}")
        print(f"  Perturbed loss: {pert_loss:.4f}")
        print(f"  Difference: {pert_loss - orig_loss:.4f}")
        print(f"  Gradient norm: {torch.norm(grad, p='fro'):.4f}")
