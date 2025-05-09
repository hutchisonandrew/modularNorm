import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from CIFAR100_dataset import CIFAR10DataModule, CIFAR100DataModule
import os
import json
import numpy as np
import torch.nn.functional as F
import csv
import tqdm

def compute_updated_loss(model, batch, delta_w1, delta_w2, device='cuda'):
    model.eval()
    model = model.to(device)
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    model.zero_grad()
    
    layer1_module, layer2_module = None, None
    original_w1, original_w2 = None, None

    # Find the first two layers with weights
    found_layers_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if found_layers_count == 0:
                layer1_module = module
                original_w1 = layer1_module.weight.data.clone()
                found_layers_count += 1
            elif found_layers_count == 1:
                layer2_module = module
                original_w2 = layer2_module.weight.data.clone()
                found_layers_count += 1
                break # Found both, no need to search further
    
    # Apply perturbation to the first layer
    if layer1_module is not None:
        layer1_module.weight.data = original_w1 + delta_w1
    else:
        # This case should ideally not be hit if delta_w1 is always meant for an existing layer
        raise ValueError("delta_w1 was provided, but no first layer with weights was found.")

    # Apply perturbation to the second layer if delta_w2 is provided
    if delta_w2 is not None:
        if layer2_module is not None:
            layer2_module.weight.data = original_w2 + delta_w2
        else:
            # This case should ideally not be hit if delta_w2 is always meant for an existing second layer
            raise ValueError("delta_w2 was provided, but no second layer with weights was found.")
    
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)

    # Revert model weights
    if layer1_module is not None and original_w1 is not None:
        layer1_module.weight.data = original_w1
    
    if layer2_module is not None and original_w2 is not None: # original_w2 implies delta_w2 was processed
        layer2_module.weight.data = original_w2
    
    return loss.item()

def compute_gradient_magnitudes(gradients, magnitude_type):
    if magnitude_type == "frobenius":
        layer0_magnitude = torch.norm(gradients['layer_0'], p='fro')
        layer1_magnitude = torch.norm(gradients['layer_1'], p='fro')
    elif magnitude_type == "spectral":
        layer0_magnitude = torch.linalg.norm(gradients['layer_0'], ord=2)
        layer1_magnitude = torch.linalg.norm(gradients['layer_1'], ord=2)
    return layer0_magnitude, layer1_magnitude

def compute_loss_and_all_layer_gradients(model, batch, device='cuda'):
    model.eval()  # Keep model in eval mode if not training, but ensure grads are enabled for params
    model = model.to(device)
    x, y = batch
    x, y = x.to(device), y.to(device)

    model.zero_grad()  

    outputs = model(x)
    loss = F.cross_entropy(outputs, y)

    loss.backward() 

    current_layer = 0
    layer_gradients = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
            layer_gradients[f'layer_{current_layer}'] = module.weight.grad.clone()
            current_layer += 1
    if len(layer_gradients)!= 2:
        raise ValueError("Expected 2 layers with gradients, but got {}".format(len(layer_gradients)))
    return loss.item(), layer_gradients

def compute_modular_residual(original_loss, perturbed_loss, gradients, delta_w1, delta_w2, delta_w1_norm, delta_w2_norm):
    
    layer0_grad = gradients['layer_0']
    layer1_grad = gradients['layer_1']
    
    # These are GPU tensors/scalars
    first_order_term_gpu = torch.sum(layer0_grad * delta_w1) + torch.sum(layer1_grad * delta_w2)
    # delta_w1_norm and delta_w2_norm are GPU scalars (0-dim tensors)
    modular_norm_gpu = torch.maximum(delta_w1_norm, delta_w2_norm)
    
    # original_loss and perturbed_loss are CPU floats.
    # Convert GPU scalars to CPU floats for arithmetic.
    rhs = original_loss + first_order_term_gpu.item() + modular_norm_gpu.item()
    
    return rhs - perturbed_loss

def sample_delta_w(layer_shape, magnitude, device, magnitude_type):
    delta_w = torch.randn(layer_shape, device=device)
    delta_w_frobenius = torch.norm(delta_w, p='fro')
    delta_w_spectral = torch.linalg.norm(delta_w, ord=2)
    
    if magnitude_type == "frobenius":
        delta_w = delta_w * magnitude / delta_w_frobenius
        delta_w_spectral =  delta_w_spectral * magnitude / delta_w_frobenius
        delta_w_frobenius = magnitude
    elif magnitude_type == "spectral":
        delta_w = delta_w * magnitude / delta_w_spectral
        delta_w_frobenius = delta_w_frobenius * magnitude / delta_w_spectral
        delta_w_spectral = magnitude
    return delta_w, delta_w_frobenius, delta_w_spectral
  
class WeightPerturbationCallback(Callback):
    """
    Callback to analyze the effect of weight perturbations on loss at the end of each epoch.
    """
    def __init__(self, num_magnitudes=10, magnitude_type="frobenius", save_dir='perturbation_analysis'):
        """
        Args:
            layer_names: List of layer names to analyze (e.g., ['layers.0', 'layers.2'])
            save_dir: Directory to save results

        """
        super().__init__()
        
        self.magnitude_type = magnitude_type
        self.save_dir = save_dir
        self.number_of_magnitudes = num_magnitudes
        print(self.magnitude_type)
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Analyze weight perturbations at the end of each epoch and save to CSV."""
        
        epoch = trainer.current_epoch
        
        if epoch % 5 != 0:
            return
        
        device = pl_module.device
        print(device)
        # Dictionary to store results for this epoch
        
        
       
        
        #compute gradients and their magnitudes
        random_batch = next(iter(trainer.train_dataloader))
        original_loss, layer_gradients = compute_loss_and_all_layer_gradients(pl_module, random_batch, device)
        layer0_shape = layer_gradients['layer_0'].shape
        layer1_shape = layer_gradients['layer_1'].shape
        gradient_mag1, gradient_mag2 = compute_gradient_magnitudes(layer_gradients, self.magnitude_type)
        
        
        #Construct perturbation magnitudes
        firs_step_size = 0.1 *gradient_mag1 / float(self.number_of_magnitudes)
        first_gradient_magnitudes = [firs_step_size * i for i in range(1, self.number_of_magnitudes + 1)]
        
        second_step_size = 0.1 * gradient_mag2 / float(self.number_of_magnitudes)
        second_gradient_magnitudes = [second_step_size * i for i in range(1, self.number_of_magnitudes + 1)]
        
    
            
        for i in tqdm.tqdm(range(60)):
                # For each perturbation magnitude
            for magnitude_iter in range(self.number_of_magnitudes):
                    # Generate random perturbation with specified magnitude
                    delta_w1, delta_w1_frobenius, delta_w1_spectral = sample_delta_w(layer0_shape, first_gradient_magnitudes[magnitude_iter], device, self.magnitude_type)
                    delta_w2, delta_w2_frobenius, delta_w2_spectral = sample_delta_w(layer1_shape, second_gradient_magnitudes[magnitude_iter], device, self.magnitude_type)
                    # Compute batch-wise losses and gradients on train set
                    updated_loss = compute_updated_loss(pl_module, random_batch, delta_w1, delta_w2, device)
                
                    
                    

                    batch_residual_frobenius = compute_modular_residual(original_loss, updated_loss, layer_gradients, delta_w1, delta_w2, delta_w1_frobenius, delta_w2_frobenius)
                    batch_residual_spectral = compute_modular_residual(original_loss, updated_loss, layer_gradients, delta_w1, delta_w2, delta_w1_spectral, delta_w2_spectral)
                    # Store results
                    
                    data_row = [
                        epoch,
                        batch_residual_frobenius,
                        batch_residual_spectral,
                        delta_w1_spectral.item(),
                        delta_w1_frobenius.item(),
                        delta_w2_spectral.item(),
                        delta_w2_frobenius.item(),
                        magnitude_iter
                    ]
    
                    
                    
                    
                    
           
        # Save the results for this epoch
                    self._save_epoch_results(data_row)
        
        print(f"Completed weight perturbation analysis for epoch {epoch}")
    
    def _save_epoch_results(self, data_row):
        """Save a single data row to the CSV file."""
        if not data_row: # data_row is a single list representing one row
            return

        filepath = os.path.join(self.save_dir, "perturbation_analysis.csv")
        
        # Check if file exists to determine if header is needed
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header only if file is new
            if not file_exists:
                header = [
                    'epoch',
                    'frobenius_residual',
                    'spectral_residual',
                    'deltaw1_spectral_magnitude',
                    'deltaw1_frobenius_magnitude',
                    'deltaw2_spectral_magnitude',
                    'deltaw2_frobenius_magnitude',
                    'magnitude_iteration' # Added new column
                ]
                writer.writerow(header)
            
            writer.writerow(data_row) # Use writerow for a single row

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
