import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from src.dataload import WikiAutoAssetDataset
from src.para_jepa_train import ParaJEPA
from config import Config
from src.evaluation_utils import Evaluator
from main import get_device, set_seed

def train_short(model, train_loader, optimizer, device, max_steps=500):
    model.train()
    total_loss = 0
    steps = 0
    
    progress_bar = tqdm(total=max_steps, desc="Training (Short)", leave=False)
    
    while steps < max_steps:
        for batch in train_loader:
            if steps >= max_steps:
                break
                
            style_input = {
                'input_ids': batch['style_input_ids'].to(device),
                'attention_mask': batch['style_attention_mask'].to(device),
            }
            content_input = {
                'input_ids': batch['content_input_ids'].to(device),
                'attention_mask': batch['content_attention_mask'].to(device),
            }

            loss, _, _ = model(style_input, content_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_ema()

            total_loss += loss.item()
            steps += 1
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item()})
            
    return total_loss / steps

def run_ablation():
    os.makedirs("reports", exist_ok=True)
    
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    print("-" * 30)
    print("Starting Ablation Study: Bottleneck Size")
    print("-" * 30)
    
    print("Loading Data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=config.max_length)
    except Exception as e:
        print(f"Error loading train set: {e}")
        return

    try:
        val_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=config.max_length)
    except Exception as e:
        print(f"Error loading val set: {e}")
        return
    
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    n_val = min(len(val_set), 500)
    val_subset_indices = np.random.choice(len(val_set), n_val, replace=False)
    val_subset = Subset(val_set, val_subset_indices)
    
    bottleneck_dims = [128, 384, 768]
    results = {
        'dim': [],
        'style_acc': [],
        'content_acc': []
    }
    
    evaluator = Evaluator(device=device)
    
    for dim in bottleneck_dims:
        print(f"\nTesting Bottleneck Dim: {dim}")
        
        model = ParaJEPA(
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            pred_hidden_dim=dim, 
            pred_depth=config.pred_depth
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        print(f"Training for 500 steps...")
        train_short(model, train_loader, optimizer, device, max_steps=500)
        
        print("Evaluating Probes...")
        embeddings, style_labels, content_labels = evaluator.get_embeddings_with_labels(
            model, val_subset, tokenizer, max_length=config.max_length, batch_size=config.batch_size
        )
        
        n = len(embeddings)
        split = int(n * 0.8)
        
        style_acc = evaluator.linear_probe(
            embeddings[:split], style_labels[:split],
            embeddings[split:], style_labels[split:],
            task_name=f"Style Probe (Dim={dim})"
        )
        
        content_acc = evaluator.linear_probe(
            embeddings[:split], content_labels[:split],
            embeddings[split:], content_labels[split:],
            task_name=f"Content Probe (Dim={dim})"
        )
        
        results['dim'].append(dim)
        results['style_acc'].append(style_acc)
        results['content_acc'].append(content_acc)
        
        del model
        torch.cuda.empty_cache()

    print("\nGenerating Ablation Plot...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['dim'], results['style_acc'], 'o-', linewidth=2, label='Style Accuracy (Lower is better)')
    
    plt.plot(results['dim'], results['content_acc'], 's--', linewidth=2, label='Content Accuracy (Higher is better)')
    
    plt.title("Ablation Study: Predictor Bottleneck Size vs. Disentanglement")
    plt.xlabel("Predictor Hidden Dimension")
    plt.ylabel("Probe Accuracy")
    plt.xticks(bottleneck_dims)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    for i, dim in enumerate(results['dim']):
        plt.annotate(f"{results['style_acc'][i]:.2f}", (dim, results['style_acc'][i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.tight_layout()
    plt.savefig("reports/ablation_plot.png")
    print("Plot saved to reports/ablation_plot.png")
    
    df = pd.DataFrame(results)
    df.to_csv("reports/ablation_results.csv", index=False)
    print("Results saved to reports/ablation_results.csv")
    
    print("-" * 30)
    print("Ablation Study Complete")
    print("-" * 30)

if __name__ == "__main__":
    run_ablation()
