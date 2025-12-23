"""
Example script to run evaluation after training.

Usage:
    python run_evaluation.py
"""

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

from dataload import WikiAutoAssetDataset
from para_jepa_train import ParaJEPA
from config import Config
from evaluation_utils import run_full_evaluation, Evaluator
from main import get_device, set_seed


def main():
    """Run evaluation on trained model"""
    config = Config()
    device = get_device()
    seed = set_seed(config.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load WikiAuto/ASSET text simplification dataset (same family as training)
    # We mirror the splits used in main.py: train / validation / test_asset (with fallback).
    print("Loading WikiAutoAssetDataset for Simplification Task (evaluation)...")
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=config.max_length)
        valid_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=config.max_length)
        try:
            test_set = WikiAutoAssetDataset(tokenizer, split='test_asset', max_length=config.max_length)
        except Exception:
            print("Warning: 'test_asset' split not found, using 'validation' for testing in evaluation.")
            test_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=config.max_length)
    except Exception as e:
        print(f"Failed to load WikiAutoAssetDataset for evaluation: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=False,  # Don't shuffle for evaluation
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=config.batch_size, 
        shuffle=False,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    # Load model
    model = ParaJEPA(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        ema_decay=config.ema_decay,
        pred_depth=config.pred_depth
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load("para_jepa_best_model.pt", map_location=device))
        print("Loaded trained model weights from para_jepa_best_model.pt")
    except FileNotFoundError:
        print("Error: para_jepa_best_model.pt not found. Please train the model first.")
        return
    
    # Run full evaluation
    results = run_full_evaluation(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        max_length=config.max_length
    )
    
    print("\nEvaluation complete! Check the generated plots:")
    print("  - tsne_content.png: Should show clustering by content")
    print("  - tsne_style.png: Should show mixing (no clustering) by style")


if __name__ == "__main__":
    main()

