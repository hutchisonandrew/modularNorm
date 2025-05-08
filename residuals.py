import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from CIFAR100_dataset import CIFAR10DataModule, CIFAR100DataModule
import os
import json
import numpy as np


def compute_losses(model, dataloader, layer_name, delta_w, device='cpu', max_batches=None):
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
    
    # Create iterators
    data_iter = iter(dataloader)

    # First pass with original weights
    for _ in range(max_batches):
        try:
            batch = next(data_iter)
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
        except StopIteration:
            break
    
    # Restore model to clean state
    model.zero_grad()
    
    # Apply perturbation
    layer.weight.data = original_weights + delta_w.to(device)
    
    # Reset iterator for second pass
    data_iter = iter(dataloader)

    # Second pass with perturbed weights
    for _ in range(max_batches):
        try:
            batch = next(data_iter)
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            
            # Store perturbed loss
            perturbed_losses.append(loss.item())
            
            # Increment batch counter
            batch_count += 1
        except StopIteration:
            break
    
    # Restore original weights
    layer.weight.data = original_weights
    
    return original_losses, first_order_terms, perturbed_losses


def residual_equation(original_losses, first_order_terms, perturbed_losses, norm):
    residuals = []
    for i in range(len(original_losses)):
        lhs = perturbed_losses[i]
        rhs = original_losses[i] + first_order_terms[i] + norm
        residuals.append(rhs - lhs)
        
    return residuals
    
class WeightPerturbationCallback(Callback):
    """
    Callback to analyze the effect of weight perturbations on loss at the end of each epoch.
    """
    def __init__(self, delta_magnitudes=[1.0], save_dir='perturbation_analysis', max_batches=2):
        """
        Args:
            layer_names: List of layer names to analyze (e.g., ['layers.0', 'layers.2'])
            delta_magnitudes: Magnitudes of perturbations to apply
            save_dir: Directory to save results
            max_batches: Maximum number of batches to analyze (for efficiency)
        """
        super().__init__()
        self.delta_magnitudes = delta_magnitudes
        self.save_dir = save_dir
        self.max_batches = max_batches
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Analyze weight perturbations at the end of each epoch"""
        epoch = trainer.current_epoch
        device = pl_module.device
        
        # Dictionary to store results for this epoch
        epoch_results = {}
        
        # Analyze each layer
        
            
        for layer_name, layer in pl_module.named_modules():
            if not hasattr(layer, 'weight'):
                continue
            
            for i in range(10):
                print(f"Analyzing layer {layer_name} and at {i} iteration")
                delta_w = torch.randn_like(layer.weight.data)
                # For each perturbation magnitude
                for magnitude in self.delta_magnitudes:
                    # Generate random perturbation with specified magnitude
                # Normalize and scale to desired magnitude
                    delta_w = delta_w * magnitude 
                    delta_w_frobenius = torch.norm(delta_w, p='fro')
                    # Using torch.linalg.norm with ord=2 for spectral norm
                    delta_w_spectral = torch.linalg.norm(delta_w, ord=2)
                    
                    print(f"Delta w frobenius: {delta_w_frobenius}, Delta w spectral: {delta_w_spectral}")
                    # Compute batch-wise losses and gradients on train set
                    train_orig_losses, train_grads, train_pert_losses = compute_losses(
                        pl_module, trainer.train_dataloader, layer_name, delta_w, device,
                        max_batches=self.max_batches
                    )
                    

                    batch_residuals_frobenius = residual_equation(train_orig_losses, train_grads, train_pert_losses, delta_w_frobenius)
                    batch_residuals_spectral = residual_equation(train_orig_losses, train_grads, train_pert_losses, delta_w_spectral)
                    # Store results
                    if layer_name not in epoch_results:
                        epoch_results[layer_name] = []
                    
                    
                    # Simply store the original and perturbed losses
                    result = {
                        'batch_residuals_frobenius': batch_residuals_frobenius,
                        'batch_residuals_spectral': batch_residuals_spectral
                    }
                    print(result)
                    
                    epoch_results[layer_name].append(result)
            break
        # Save the results for this epoch
        self._save_epoch_results(epoch, epoch_results)
        
        print(f"Completed weight perturbation analysis for epoch {epoch}")
    
    def _save_epoch_results(self, epoch, epoch_results):
        """Save the results for a specific epoch to disk"""
        # Create a serializable version of the results
        serializable_results = self._make_serializable(epoch_results)
        
        # Save to a JSON file
        filename = os.path.join(self.save_dir, f"perturbation_results_epoch_{epoch}.json")
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    
    def _make_serializable(self, results):
        """Convert any non-serializable objects to serializable ones"""
        if isinstance(results, torch.Tensor):
            return results.detach().cpu().numpy().tolist()
        elif isinstance(results, np.ndarray):
            return results.tolist()
        elif isinstance(results, dict):
            return {k: self._make_serializable(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._make_serializable(item) for item in results]
        else:
            return results

# if __name__ == "__main__":
#     # Load model
#     model_path = 'runs/my_experiment/logs/version_0/checkpoints/epoch=0-step=782.ckpt'
#     model = MLP.load_from_checkpoint(model_path)
    
#     # Get model parameters to determine dataset
#     num_classes = model.layers[-1].out_features  # Last layer output size indicates dataset
    
#     # Create appropriate data module
#     batch_size = 512
#     num_workers = 4
#     if num_classes == 100:
#         data_module = CIFAR100DataModule(num_workers=num_workers, batch_size=batch_size)
#     else:
#         data_module = CIFAR10DataModule(num_workers=num_workers, batch_size=batch_size)
    
#     # Setup data module
#     data_module.prepare_data()
#     data_module.setup(stage=None)
    
#     # Get data loaders
#     train_dataloader = data_module.train_dataloader()
    
#     # Print model layers
#     print("Layers with weights:")
#     for name, module in model.named_modules():
#         if hasattr(module, 'weight'):
#             print(f"{name}: {type(module).__name__}, Shape: {module.weight.shape}")
    
#     # Example of computing per-batch losses and gradients
#     layer_name = 'layers.6'  # Example - output layer
#     delta_w = torch.randn_like(model.layers[6].weight.data)
#     original_norm = torch.norm(delta_w, p='fro')
#     print(f"Original perturbation norm: {original_norm:.4f}")
    
#     # Scale perturbation to desired magnitude
#     perturbation_magnitude = 200
#     delta_w = delta_w * perturbation_magnitude / original_norm
#     print(f"Scaled perturbation norm: {torch.norm(delta_w, p='fro'):.4f}")
    
#     # Limit to first few batches for demonstration
#     max_batches = 3
    
#     # Compute batch-wise losses, gradients, and perturbed losses
#     orig_losses, first_order_terms, pert_losses = compute_losses(
#         model, train_dataloader, layer_name, delta_w, max_batches=max_batches
#     )
    
#     # Print results for each batch
#     print(f"\nLayer: {layer_name}")
#     print(f"Perturbation magnitude: {perturbation_magnitude}")
    
#     for i, (orig_loss, grad, pert_loss) in enumerate(zip(orig_losses, grads, pert_losses)):
#         print(f"\nBatch {i+1}:")
#         print(f"  Original loss: {orig_loss:.4f}")
#         print(f"  Perturbed loss: {pert_loss:.4f}")
#         print(f"  Difference: {pert_loss - orig_loss:.4f}")
#         print(f"  Gradient norm: {torch.norm(grad, p='fro'):.4f}")
