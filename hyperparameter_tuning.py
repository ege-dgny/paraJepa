"""
Hyperparameter Tuning Script for ParaJEPA using Weights & Biases

This script supports both:
1. Single runs with W&B logging
2. W&B Sweeps for automated hyperparameter search

Usage:
    # Single run with W&B logging
    python hyperparameter_tuning.py
    
    # Initialize a sweep (run once)
    wandb sweep sweep_config.yaml
    
    # Run agent (run on each machine/GPU)
    wandb agent <sweep_id>
"""

import wandb
import torch
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import random
import numpy as np
import os

from dataload import WikiAutoAssetDataset
from para_jepa_train import ParaJEPA, train_para_jepa
from config import Config


def get_device():
    """Get the available device (CUDA, MPS, or CPU)"""
    if torch.backends.cuda.is_built():
        print("Using CUDA")
        device = torch.device('cuda')
    elif torch.backends.mps.is_built():
        print("Using MPS")
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def set_seed(seed=11):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def config_to_dict(config):
    """Convert Config object to dictionary for W&B"""
    return {
        'model_name': config.model_name,
        'hidden_dim': config.hidden_dim,
        'ema_decay': config.ema_decay,
        'pred_depth': config.pred_depth,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'epochs': config.epochs,
        'max_length': config.max_length,
        'seed': config.seed,
    }


def train(config=None, use_wandb=True):
    """
    Main training function that can be used for both single runs and sweeps
    
    Args:
        config: Config object or dict. If None, uses default Config()
        use_wandb: Whether to log to W&B
    """
    # Initialize W&B
    if use_wandb:
        if wandb.run is None:
            # Single run mode
            wandb.init(
                project="para-jepa",
                name="single-run",
                config=config_to_dict(config) if config else None
            )
        else:
            # Sweep mode - config comes from wandb.config
            # Convert wandb.config to Config object
            sweep_config = wandb.config
            config = Config()
            config.model_name = sweep_config.get('model_name', config.model_name)
            config.hidden_dim = sweep_config.get('hidden_dim', config.hidden_dim)
            config.ema_decay = sweep_config.get('ema_decay', config.ema_decay)
            config.pred_depth = sweep_config.get('pred_depth', config.pred_depth)
            config.batch_size = sweep_config.get('batch_size', config.batch_size)
            config.learning_rate = sweep_config.get('learning_rate', config.learning_rate)
            config.weight_decay = sweep_config.get('weight_decay', config.weight_decay)
            config.epochs = sweep_config.get('epochs', config.epochs)
            config.max_length = sweep_config.get('max_length', config.max_length)
            config.seed = sweep_config.get('seed', config.seed)
    
    # Use default config if not provided
    if config is None:
        config = Config()
    
    # Extract hyperparameters
    model_name = config.model_name
    hidden_dim = config.hidden_dim
    ema_decay = config.ema_decay
    pred_depth = config.pred_depth
    batch_size = config.batch_size
    lr = config.learning_rate
    weight_decay = config.weight_decay
    epochs = config.epochs
    max_length = config.max_length
    seed = config.seed
    
    # Set seed for reproducibility
    set_seed(seed)
    device = get_device()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load WikiAuto/ASSET text simplification dataset (same as main.py)
    # Use explicit splits provided by the dataset.
    print("Loading WikiAutoAssetDataset for Simplification Task (hyperparameter tuning)...")
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=max_length)
        valid_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
        try:
            test_set = WikiAutoAssetDataset(tokenizer, split='test_asset', max_length=max_length)
        except Exception:
            print("Warning: 'test_asset' split not found, using 'validation' for testing in hyperparameter tuning.")
            test_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
    except Exception as e:
        print(f"Failed to load WikiAutoAssetDataset for hyperparameter tuning: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    # Initialize model
    model = ParaJEPA(
        model_name=model_name,
        hidden_dim=hidden_dim,
        ema_decay=ema_decay,
        pred_depth=pred_depth
    ).to(device)
    
    # Count parameters for logging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if use_wandb:
        wandb.config.update({
            'total_params': total_params,
            'trainable_params': trainable_params,
        })
        wandb.watch(model, log='all', log_freq=100)
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train model
    metrics = train_para_jepa(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        use_wandb=use_wandb
    )
    
    # Log final metrics to W&B
    if use_wandb:
        wandb.log({
            'best_val_loss': metrics['best_val_loss'],
            'final_train_loss': metrics['final_train_loss'],
            'final_val_loss': metrics['final_val_loss'],
            'final_val_cosine_sim': metrics['final_val_cosine_sim'],
        })
        
        # Mark best run (validation metrics are what we optimize)
        wandb.run.summary['best_val_loss'] = metrics['best_val_loss']
        wandb.run.summary['final_val_cosine_sim'] = metrics['final_val_cosine_sim']
    
    return metrics


def main():
    """Main entry point for hyperparameter tuning"""
    # Check if running as part of a sweep
    if wandb.run is None:
        # Single run mode
        print("Running in single-run mode with W&B logging")
        config = Config()
        train(config=config, use_wandb=True)
    else:
        # Sweep mode
        print("Running in sweep mode")
        train(use_wandb=True)


if __name__ == "__main__":
    main()