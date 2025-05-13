import sys
import os

# Get the absolute path of the directory containing compute_kernels.py (analysis/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of current_script_dir (project root)
project_root = os.path.dirname(current_script_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models import MLP # Assuming models.py is in the same directory or accessible via PYTHONPATH
import argparse
from analysis.load_model import load_mlp_from_experiment # Import the new loader

def get_post_activations(experiment_dir: str, data_root='./workspace/datasets/cifar10'):

    """
    Computes the output of the second-to-last Linear layer 
    (e.g., for an MLP like L3(A2(L2(A1(L1(x))))), this would be L2(A1(L1(x))))
    for a given model checkpoint on the CIFAR10 test set.

    Args:
        experiment_dir (str): Path to the experiment directory which contains
                              config.yaml and checkpoints/final_model.pt.
        data_root (str): Root directory for CIFAR10 dataset.

    Returns:
        torch.Tensor: A tensor containing the activations for all test samples.
                      Shape: (num_test_samples, activation_dim)
    """
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the model using the new loader function
    try:
        model, _ = load_mlp_from_experiment(experiment_dir, device=device.type) # Pass device.type (e.g., "cuda" or "cpu")
        # The load_mlp_from_experiment function already calls model.eval() and model.to(device)
    except Exception as e:
        print(f"Error loading model from experiment directory {experiment_dir}: {e}")
        return None
    
    # 3. CIFAR10 Test dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) 
    ])

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=transform)
    # DO NOT SHUFFLE THE TEST SET to ensure consistent order
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, # Batch size can be adjusted
                                             shuffle=False, num_workers=2)

    # 4. Register a hook to get the output of the second-to-last Linear layer
    activations = []
    def hook_fn(module, input_activations, output_activations):
        # We want the output of the hooked layer.
        activations.append(output_activations.cpu().detach())

    # Identify the layer to hook: the second-to-last nn.Linear layer in model.layers
    target_hook_layer = None
    if isinstance(model, MLP) and hasattr(model, 'layers') and isinstance(model.layers, torch.nn.Sequential):
        linear_layers_in_sequence = []
        for layer_module in model.layers.children(): # Corrected variable name from 'layer' to 'layer_module' to avoid conflict
            if isinstance(layer_module, torch.nn.Linear):
                linear_layers_in_sequence.append(layer_module)
        
        if len(linear_layers_in_sequence) >= 2:
            target_hook_layer = linear_layers_in_sequence[-2] # The second-to-last Linear layer
        else:
            raise ValueError(f"MLP model's 'layers' attribute does not contain at least two nn.Linear modules. Found {len(linear_layers_in_sequence)}.")
    else:
        # This was the user's existing error for non-MLP or incorrectly structured models.
        raise ValueError("Model is not an MLP or does not have a Sequential 'layers' attribute suitable for this operation.")

    if target_hook_layer is None:
        # This case should ideally be covered by the error checks above, but as a safeguard:
        print("Failed to identify the target layer (second-to-last linear layer) to attach hook.")
        return None

    print(f"Attaching hook to capture output of layer: {target_hook_layer} (the second-to-last Linear layer)")
    handle = target_hook_layer.register_forward_hook(hook_fn)

    # 5. Iterate through the test set and collect activations
    all_activations_list = []
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data
            inputs = inputs.to(device)
            _ = model(inputs) # Forward pass. Activations are captured by the hook.
            # 'activations' list is populated by the hook for each forward pass (batch)
            # We need to extend all_activations_list with the items from 'activations'
            # and then clear 'activations' for the next batch.
            for act_batch in activations:
                all_activations_list.append(act_batch)
            activations.clear() # Clear for the next batch

    # Remove the hook
    handle.remove()

    # 6. Concatenate activations
    if not all_activations_list:
        print("No activations were collected. Check model structure or hook registration.")
        return None
        
    try:
        final_activations = torch.cat(all_activations_list, dim=0)
        print(f"Collected activations of shape: {final_activations.shape}")
        return final_activations
    except Exception as e:
        print(f"Error concatenating activations: {e}")
        print(f"Number of activation batches: {len(all_activations_list)}")
        if all_activations_list:
            print(f"Shape of first activation batch: {all_activations_list[0].shape}")
        return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute pre-activations from a model checkpoint.")
    parser.add_argument(
        "--experiment_dir", 
        type=str, 
        default="./runs/MyExperiment",  # Example default, change as needed
        help="Path to the experiment directory containing the 'checkpoints/final_model.pt' file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./workspace/datasets/cifar10',
        help="Root directory for the CIFAR10 dataset."
    )
    args = parser.parse_args()

    print(f"Attempting to compute activations using experiment directory: {args.experiment_dir}")

    # Construct checkpoint path for information, though the loader handles it internally.
    # checkpoint_path_info = os.path.join(args.experiment_dir, 'checkpoints', 'final_model.pt')
    data_directory = args.data_dir

    # print(f"Expecting checkpoint at: {checkpoint_path_info}") # Informational only
    print(f"Using data directory: {data_directory}")

    activations_tensor = get_post_activations(
        experiment_dir=args.experiment_dir,
        data_root=data_directory
    )

    if activations_tensor is not None:
        print(f"Successfully computed activations. Shape: {activations_tensor.shape}")
        # Save the tensor
        analysis_dir = os.path.join(args.experiment_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        output_save_path = os.path.join(analysis_dir, 'output_second_last_linear_cifar10_test.pt')
        torch.save(activations_tensor, output_save_path)
        print(f"Activations saved to {output_save_path}")
    else:
        print(f"Failed to compute activations. Check checkpoint path and model compatibility.")
