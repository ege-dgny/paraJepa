import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np

from src.dataload import WikiAutoAssetDataset
from src.para_jepa_train import ParaJEPA
from config import Config
from src.evaluation_utils import run_full_evaluation, Evaluator
from main import get_device, set_seed


def main():
    config = Config()
    device = get_device()
    seed = set_seed(config.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
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
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=False,  
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=config.batch_size, 
        shuffle=False,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    model = ParaJEPA(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        ema_decay=config.ema_decay,
        pred_depth=config.pred_depth
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("para_jepa_best_model.pt", map_location=device))
        print("Loaded trained model weights from para_jepa_best_model.pt")
    except FileNotFoundError:
        print("Error: para_jepa_best_model.pt not found. Please train the model first.")
        return
    
    results = run_full_evaluation(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        max_length=config.max_length
    )
    
    print("-" * 30)
    print("Evaluation Complete")
    print("-" * 30)
    print("Generated plots:")
    print("  - tsne_content.png")
    print("  - tsne_style.png")


if __name__ == "__main__":
    main()
