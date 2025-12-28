"""
Conditional JEPA Training Script

This script trains ParaJEPAConditional, which explicitly feeds the target length
to the predictor. This "explains away" the length information, forcing the
context encoder to focus on semantics instead of acting as a length-identity-mapper.

Usage:
    python main_conditional.py
"""

from dataload import WikiAutoAssetDataset
from para_jepa_conditional import ParaJEPAConditional
from para_jepa_train import train_para_jepa
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import shutil
from config import Config

def get_device():
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    return device

def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def main(config=None):
    if config is None:
        config = Config()
    
    model_name = config.model_name
    hidden_dim = config.hidden_dim
    ema_decay = config.ema_decay
    pred_depth = config.pred_depth
    batch_size = config.batch_size
    lr = config.learning_rate
    weight_decay = config.weight_decay
    epochs = config.epochs
    max_length = config.max_length
    num_workers = config.num_workers
    # Use a moderate bottleneck (128) to test if conditioning solves the identity mapping
    # without needing extreme compression (8-dim)
    pred_hidden_dim = 128 
    seed = set_seed(config.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("="*60)
    print("CONDITIONAL JEPA TRAINING")
    print("="*60)
    print("Strategy: Feed Target Length to Predictor to 'Explain Away' Style")
    print(f"Bottleneck Dim: {pred_hidden_dim}")
    print("Loading WikiAutoAssetDataset...")
    
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=max_length)
        valid_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
        try:
            test_set = WikiAutoAssetDataset(tokenizer, split='test_asset', max_length=max_length)
        except:
            print("Warning: 'test_asset' split not found, using 'validation' for testing.")
            test_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    # Initialize ParaJEPAConditional
    model = ParaJEPAConditional(
        model_name=model_name, 
        hidden_dim=hidden_dim, 
        ema_decay=ema_decay, 
        pred_depth=pred_depth, 
        pred_hidden_dim=pred_hidden_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train model
    # We can reuse the standard train loop because the forward signature matches
    train_para_jepa(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        optimizer=optimizer, 
        device=device, 
        epochs=epochs
    )
    
    # Rename checkpoint
    original_checkpoint = "para_jepa_best_model.pt"
    conditional_checkpoint = "para_jepa_conditional_best_model.pt"
    
    if os.path.exists(original_checkpoint):
        shutil.copy(original_checkpoint, conditional_checkpoint)
        print(f"\n✅ Checkpoint saved as: {conditional_checkpoint}")
    else:
        print(f"\n⚠️ Warning: Checkpoint {original_checkpoint} not found after training.")
    
    # Run test evaluation manually since we need to load the specific checkpoint class
    print("\n" + "="*60)
    print("RUNNING TEST EVALUATION")
    print("="*60)
    try:
        model.load_state_dict(torch.load(conditional_checkpoint, map_location=device))
        print(f"✅ Loaded checkpoint: {conditional_checkpoint}")
    except FileNotFoundError:
        print(f"❌ Could not load {conditional_checkpoint}")
        return
    
    from para_jepa_train import evaluation
    test_loss, test_cosine_sim = evaluation(model, test_loader, device, desc="Test")
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Cosine Similarity: {test_cosine_sim:.4f}")

if __name__ == '__main__':
    main()

