import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models import MLP
from ViT import LitViT
from callbacks import get_callbacks
from CIFAR100_dataset import CIFAR10DataModule, CIFAR100DataModule


def main(args):
    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    if "model_type" not in config:
        model_type = "MLP"
        num_layers = config['num_layers']
        width = config['width']
        dataset = config['dataset']
        hparas = config['hparas']
        
        # Extract hyperparameters with defaults
        epochs = hparas['epochs']
        lr = hparas['lr']
        weight_decay = hparas['weight_decay']
        batch_size = hparas['batch_size']
        momentum = hparas['momentum']
        
        experiment_hparas = config['experiment_hparas']
        if len(experiment_hparas) == 0:
            raise ValueError("Experiment hyperparameters are not provided in the config file")
        num_magnitudes = experiment_hparas['num_magnitudes']
        magnitude_type = experiment_hparas['magnitude_type']
        optimizer = experiment_hparas['optimizer']
    else:
        model_type = config['model_type']
        dataset = config['dataset']
        hparas = config['hparas']
        
        # Extract hyperparameters with defaults
        epochs = hparas['epochs']
        lr = hparas['lr']
        weight_decay = hparas['weight_decay']
        batch_size = hparas['batch_size']
        momentum = hparas['momentum']
        
        experiment_hparas = config['experiment_hparas']
        num_magnitudes = experiment_hparas['num_magnitudes']
        magnitude_type = experiment_hparas['magnitude_type']
        optimizer = experiment_hparas['optimizer']
        
        # Extract ViT specific parameters if model_type is 'vit'
        if model_type.lower() == 'vit':
            img_size = config['img_size']
            patch_size = config['patch_size']
            in_chans = config['in_chans']
            num_classes = config['num_classes']
            embed_dim = config['embed_dim']
            depth = config['depth']
            num_heads = config['num_heads']
            mlp_ratio = config['mlp_ratio']
            dropout = config['dropout']
    
    # Create output directory based on command line argument
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Define checkpoint and metrics directories
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    metrics_dir = os.path.join(output_dir, 'metrics')
    
    # Set up model
    # Convert num_layers and width to MLP architecture
    # For num_layers=5 and width=2000, we create [2000, 2000, 2000, 2000] hidden dims 
    if model_type.lower() == 'mlp':
        hidden_dims = [width] * (num_layers - 1)
        
        # Determine input and output dimensions based on dataset
        if dataset.lower() == 'cifar100':
            input_dim = 3072  # 32x32x3
            num_classes = 100
        else:  # Default to CIFAR10
            input_dim = 3072  # 32x32x3
            num_classes = 10
            
        # Create MLP model
        model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer=optimizer,
        )
    elif model_type.lower() == 'vit':
        model = LitViT(
            learning_rate=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer=optimizer,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
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
        enable_checkpointing=True, # Disable default checkpointing
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
        if model_type.lower() == 'mlp':
            full_config = {
                'model_type': 'mlp',
                'num_layers': num_layers,
                'width': width,
                'dataset': dataset,
                'hparas': {
                    'epochs': epochs,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'dataset': dataset,
                    'momentum': momentum
                },
                'model_architecture': {
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims,
                    'num_classes': num_classes
                },
                'experiment_hparas': {
                    'num_magnitudes': num_magnitudes,
                    'magnitude_type': magnitude_type,
                    'optimizer': optimizer,
                    'momentum': momentum
                }
            }
        elif model_type.lower() == 'vit':
            full_config = {
                'model_type': 'vit',
                'dataset': dataset,
                'img_size': img_size,
                'patch_size': patch_size,
                'in_chans': in_chans,
                'num_classes': num_classes,
                'embed_dim': embed_dim,
                'depth': depth,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'dropout': dropout,
                'hparas': {
                    'epochs': epochs,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'momentum': momentum
                },
                'experiment_hparas': {
                    'num_magnitudes': num_magnitudes,
                    'magnitude_type': magnitude_type,
                    'optimizer': optimizer
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