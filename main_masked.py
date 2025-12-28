from src.dataload import WikiAutoAssetMaskedDataset
from src.para_jepa_train import ParaJEPA, train_para_jepa, test_run
from src.JEPA_Models import JEPAEncoder, JEPAPredictor
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import shutil
from config import Config

def get_device():
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        print("Device: CUDA")
        device = torch.device('cuda')
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        print("Device: MPS")
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Device: CPU")
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
    pred_hidden_dim = config.pred_hidden_dim
    mask_prob = config.mask_prob
    seed = set_seed(config.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("-" * 30)
    print("Masked JEPA Training")
    print("-" * 30)
    print(f"Masking Probability: {mask_prob}")
    print("Loading WikiAutoAssetMaskedDataset for Simplification Task...")
    
    try:
        train_set = WikiAutoAssetMaskedDataset(
            tokenizer, 
            split='train', 
            max_length=max_length,
            mask_prob=mask_prob
        )
        valid_set = WikiAutoAssetMaskedDataset(
            tokenizer, 
            split='validation', 
            max_length=max_length,
            mask_prob=mask_prob
        )
        try:
            test_set = WikiAutoAssetMaskedDataset(
                tokenizer, 
                split='test_asset', 
                max_length=max_length,
                mask_prob=mask_prob
            )
        except:
            print("Warning: 'test_asset' split not found, using 'validation' for testing.")
            test_set = WikiAutoAssetMaskedDataset(
                tokenizer, 
                split='validation', 
                max_length=max_length,
                mask_prob=mask_prob
            )
    except Exception as e:
        print(f"Failed to load WikiAutoAssetMaskedDataset: {e}")
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

    model = ParaJEPA(
        model_name=model_name, 
        hidden_dim=hidden_dim, 
        ema_decay=ema_decay, 
        pred_depth=pred_depth, 
        pred_hidden_dim=pred_hidden_dim
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_para_jepa(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        optimizer=optimizer, 
        device=device, 
        epochs=epochs
    )
    
    original_checkpoint = "para_jepa_best_model.pt"
    masked_checkpoint = "para_jepa_masked_best_model.pt"
    
    if os.path.exists(original_checkpoint):
        shutil.copy(original_checkpoint, masked_checkpoint)
        print(f"[INFO] Checkpoint saved as: {masked_checkpoint}")
        print(f"[INFO] Original checkpoint preserved as: {original_checkpoint}")
    else:
        print(f"[WARN] Checkpoint {original_checkpoint} not found after training.")
    
    print("-" * 30)
    print("Running Test Evaluation")
    print("-" * 30)
    try:
        model.load_state_dict(torch.load(masked_checkpoint, map_location=device))
        print(f"[INFO] Loaded masked checkpoint: {masked_checkpoint}")
    except FileNotFoundError:
        print(f"[ERROR] Could not load {masked_checkpoint}")
        return
    
    from src.para_jepa_train import evaluation
    test_loss, test_cosine_sim = evaluation(model, test_loader, device, desc="Test")
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Cosine Similarity: {test_cosine_sim:.4f}")

if __name__ == '__main__':
    main()
