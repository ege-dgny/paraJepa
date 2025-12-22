from dataload import ParaphraseDataset, WikiAutoAssetDataset
from para_jepa_train import ParaJEPA, train_para_jepa, test_run
from JEPA_Models import JEPAEncoder, JEPAPredictor
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
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

def main(config = None):
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
    seed = set_seed(config.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use WikiAutoAssetDataset (Simplification Task) instead of ParaphraseDataset
    print("Loading WikiAutoAssetDataset for Simplification Task...")
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=max_length)
        valid_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
        # Try 'test_asset' for testing, fallback to validation if not found
        try:
            test_set = WikiAutoAssetDataset(tokenizer, split='test_asset', max_length=max_length)
        except:
            print("Warning: 'test_asset' split not found, using 'validation' for testing.")
            test_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=max_length)
    except Exception as e:
        print(f"Failed to load WikiAutoAssetDataset: {e}")
        return

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    # Initialize ParaJEPA with bottleneck (default pred_hidden_dim=128 in class init)
    model = ParaJEPA(model_name=model_name, hidden_dim=hidden_dim, ema_decay=ema_decay, pred_depth=pred_depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_para_jepa(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        optimizer=optimizer, 
        device=device, 
        epochs=epochs
    )
    test_run(model=model, test_loader=test_loader, device=device)

if __name__ == '__main__':
    main()
