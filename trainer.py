import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models import MLP, get_callbacks
from CIFAR100_dataset import CIFAR10DataModule, CIFAR100DataModule


def main(args):
    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    num_layers = config.get('num_layers', 2)
    width = config.get('width', 512)
    dataset = config.get('dataset', 'cifar10') 
    hparas = config.get('hparas', {})
    
    # Extract hyperparameters with defaults
    epochs = hparas.get('epochs', 100)
    lr = hparas.get('lr', 0.001)
    weight_decay = hparas.get('weight_decay', 0.0001)
    batch_size = hparas.get('batch_size', 128)
    
    experiment_hparas = config.get('experiment_hparas', {})
    if len(experiment_hparas) == 0:
        raise ValueError("Experiment hyperparameters are not provided in the config file")
    num_magnitudes = experiment_hparas.get('num_magnitudes', 10)
    magnitude_type = experiment_hparas.get('magnitude_type', 'frobenius')
    
    # Create output directory based on command line argument
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define checkpoint and metrics directories
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    
    # Set up model
    # Convert num_layers and width to MLP architecture
    # For num_layers=5 and width=2000, we create [2000, 2000, 2000, 2000] hidden dims 
    hidden_dims = [width] * (num_layers - 1)
    
    # Determine input and output dimensions based on dataset
    if dataset.lower() == 'cifar100':
        input_dim = 3072  # 32x32x3
        num_classes = 100
    else:  # Default to CIFAR10
        input_dim = 3072  # 32x32x3
        num_classes = 10
    
    # Create model
    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=lr,
        weight_decay=weight_decay
    )
    
    # Set up data module
    if dataset.lower() == 'cifar100':
        data_module = CIFAR100DataModule(
            num_workers=args.num_workers,
            batch_size=batch_size
        )
    else:
        data_module = CIFAR10DataModule(
            num_workers=args.num_workers, 
            batch_size=batch_size
        )
    
    # Set up callbacks
    callbacks = get_callbacks(checkpoint_dir=checkpoint_dir, metrics_dir=metrics_dir, num_magnitudes=num_magnitudes, magnitude_type=magnitude_type)
    
    # Set up logger
    # logger = TensorBoardLogger(save_dir=output_dir, name="logs") # Disabled
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=False,  # Disable default logger
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Save model configuration
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as file:
        full_config = {
            'num_layers': num_layers,
            'width': width,
            'hparas': {
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'dataset': dataset
            },
            'model_architecture': {
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'num_classes': num_classes
            },
            'experiment_hparas': {
                'num_magnitudes': num_magnitudes,
                'magnitude_type': magnitude_type
            }
        }
        yaml.dump(full_config, file)
    
    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP with PyTorch Lightning")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model, checkpoints, and metrics')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args) 