import os
import yaml
import torch
import sys
import argparse

# Get the absolute path of the directory containing this file (analysis/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_script_dir)
# Add the project root to sys.path to allow importing 'models'
if project_root not in sys.path:
    sys.path.append(project_root)
    
from models import MLP # Assuming models.py is in the project root

def load_mlp_from_experiment(experiment_dir: str, checkpoint_filename: str = "final_model.pt", device: str = "cpu"):
    """
    Loads an MLP model from a specified experiment directory.

    This function reads the 'config.yaml' saved in the experiment directory
    to determine the model architecture and hyperparameters, instantiates the
    MLP model, and then loads the weights from the specified checkpoint file.

    Args:
        experiment_dir (str): The path to the experiment directory.
        checkpoint_filename (str, optional): The name of the checkpoint file.
            Defaults to "final_model.pt".
        device (str, optional): The device to load the model onto ('cpu' or 'cuda').
            Defaults to "cpu".

    Returns:
        torch.nn.Module: The loaded MLP model.
        dict: The configuration dictionary loaded from config.yaml.

    Raises:
        FileNotFoundError: If config.yaml or the checkpoint file is not found.
        KeyError: If essential keys are missing from the config.yaml.
    """
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model architecture and relevant hyperparameters
    # Based on how trainer.py saves the config:
    try:
        model_arch = config['model_architecture']
        input_dim = model_arch['input_dim']
        hidden_dims = model_arch['hidden_dims']
        num_classes = model_arch['num_classes']
        
        # Optimizer might be in experiment_hparas or directly in hparas in older configs
        optimizer = "AdamW" # Default if not found
        if 'experiment_hparas' in config and 'optimizer' in config['experiment_hparas']:
            optimizer = config['experiment_hparas']['optimizer']
        elif 'hparas' in config and 'optimizer' in config['hparas']: # Fallback for older config structures
             optimizer = config['hparas']['optimizer']


        # Learning rate and weight decay are part of MLP's __init__ but might not be strictly
        # necessary for just loading the model for inference if not re-training.
        # We will use defaults or extract if available.
        hparas = config.get('hparas', {})
        learning_rate = hparas.get('lr', 1e-3) # Default from MLP
        weight_decay = hparas.get('weight_decay', 1e-5) # Default from MLP

    except KeyError as e:
        raise KeyError(f"Missing essential key in config.yaml: {e}. Ensure the config contains model_architecture and optimizer details.")

    # Instantiate the model
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=learning_rate, # Though not used for inference, good to match
        weight_decay=weight_decay,   # Though not used for inference, good to match
        optimizer=optimizer
    )

    # Load the checkpoint
    checkpoint_path = os.path.join(experiment_dir, "checkpoints", checkpoint_filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        state_dict_checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if 'state_dict' in state_dict_checkpoint: # Common for PyTorch Lightning
            state_dict = state_dict_checkpoint['state_dict']
        else: # Raw PyTorch state_dict
            state_dict = state_dict_checkpoint
        
        model.load_state_dict(state_dict)
        print(f"Model loaded successfully from {checkpoint_path} using config from {config_path}")
    
    except Exception as e:
        raise RuntimeError(f"Error loading state_dict from {checkpoint_path}: {e}")

    model.to(torch.device(device))
    model.eval() # Set to evaluation mode

    return model, config

if __name__ == '__main__':
    # Example usage:
    # Replace with an actual experiment directory you have
    # example_experiment_dir = "runs/mlp_5_256_muon"
    # example_experiment_dir = "runs/mlp_2_512_adamw" # if you have such an experiment
    
    parser = argparse.ArgumentParser(description="Load an MLP model from an experiment directory.")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing config.yaml and checkpoints/final_model.pt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model onto (e.g., 'cpu', 'cuda'). Defaults to 'cpu'."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)
        
    print(f"Attempting to load model from experiment: {args.experiment_dir}")
    try:
        loaded_model, loaded_config = load_mlp_from_experiment(args.experiment_dir, device=args.device)
        print(f"Successfully loaded model: {type(loaded_model)}")
        print(f"Model is on device: {next(loaded_model.parameters()).device}")
        
        # You can now use loaded_model for inference, analysis, etc.
        # For example, print model structure:
        # print("\nModel Structure:")
        # print(loaded_model)
        
        # print("\nLoaded Configuration:")
        # print(yaml.dump(loaded_config, indent=2))

    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        sys.exit(1)
