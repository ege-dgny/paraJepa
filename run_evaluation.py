"""
Example script to run evaluation after training.

Usage:
    python run_evaluation.py
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

from dataload import ParaphraseDataset
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
    
    # Load and split dataset (same as training)
    dataset = load_dataset('humarin/chatgpt-paraphrases', split='train')
    train_test = dataset.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)
    
    # Create datasets
    train_set = ParaphraseDataset(tokenizer, max_length=config.max_length, data=train_test['train'])
    valid_set = ParaphraseDataset(tokenizer, max_length=config.max_length, data=test_valid['train'])
    test_set = ParaphraseDataset(tokenizer, max_length=config.max_length, data=test_valid['test'])
    
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
    
    # Get datasets for label extraction (optional but recommended)
    train_dataset = train_test['train']
    test_dataset = test_valid['test']
    
    # Run full evaluation
    results = run_full_evaluation(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        device=device,
        max_length=config.max_length
    )
    
    print("\nEvaluation complete! Check the generated plots:")
    print("  - tsne_content.png: Should show clustering by content")
    print("  - tsne_style.png: Should show mixing (no clustering) by style")


if __name__ == "__main__":
    main()

