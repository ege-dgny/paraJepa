import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from dataload import WikiAutoAssetDataset
from para_jepa_train import ParaJEPA
from config import Config
from evaluation_utils import Evaluator
from main import get_device, set_seed

def train_short(model, train_loader, optimizer, device, max_steps=500):
    """Train the model for a limited number of steps."""
    model.train()
    total_loss = 0
    steps = 0
    
    progress_bar = tqdm(total=max_steps, desc="Training (Short)", leave=False)
    
    # Infinite loop over loader until max_steps
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
    # Create reports dir if not exists
    os.makedirs("reports", exist_ok=True)
    
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    print("=" * 60)
    print("STARTING ABLATION STUDY: Bottleneck Size")
    print("=" * 60)
    
    # Setup Data
    print("Loading Data...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Use train set for training
    try:
        train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=config.max_length)
    except Exception as e:
        print(f"Error loading train set: {e}")
        return

    # Use validation set for evaluation
    try:
        val_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=config.max_length)
    except Exception as e:
        print(f"Error loading val set: {e}")
        return
    
    # Train Loader (Full set, but we stop early)
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    # Validation Subset for fast evaluation (e.g., 500 samples)
    # Ensure we don't pick more than available
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
        print(f"\n\n>>> Testing Bottleneck Dim: {dim}")
        
        # Init Model with specific bottleneck dim
        model = ParaJEPA(
            model_name=config.model_name,
            hidden_dim=config.hidden_dim,
            pred_hidden_dim=dim, # <--- The Independent Variable
            pred_depth=config.pred_depth
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Train Short
        print(f"Training for 500 steps...")
        train_short(model, train_loader, optimizer, device, max_steps=500)
        
        # Evaluate
        print("Evaluating Probes...")
        # Get embeddings with labels from the validation subset
        embeddings, style_labels, content_labels = evaluator.get_embeddings_with_labels(
            model, val_subset, tokenizer, max_length=config.max_length, batch_size=config.batch_size
        )
        
        # Split into train/test for the probe (80/20 split)
        n = len(embeddings)
        split = int(n * 0.8)
        
        # Style Probe
        style_acc = evaluator.linear_probe(
            embeddings[:split], style_labels[:split],
            embeddings[split:], style_labels[split:],
            task_name=f"Style Probe (Dim={dim})"
        )
        
        # Content Probe
        content_acc = evaluator.linear_probe(
            embeddings[:split], content_labels[:split],
            embeddings[split:], content_labels[split:],
            task_name=f"Content Probe (Dim={dim})"
        )
        
        results['dim'].append(dim)
        results['style_acc'].append(style_acc)
        results['content_acc'].append(content_acc)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Plotting
    print("\nGenerating Ablation Plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot Style Accuracy
    plt.plot(results['dim'], results['style_acc'], 'o-', linewidth=2, label='Style Accuracy (Lower is better)')
    
    # Plot Content Accuracy
    plt.plot(results['dim'], results['content_acc'], 's--', linewidth=2, label='Content Accuracy (Higher is better)')
    
    plt.title("Ablation Study: Predictor Bottleneck Size vs. Disentanglement")
    plt.xlabel("Predictor Hidden Dimension")
    plt.ylabel("Probe Accuracy")
    plt.xticks(bottleneck_dims)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations
    for i, dim in enumerate(results['dim']):
        plt.annotate(f"{results['style_acc'][i]:.2f}", (dim, results['style_acc'][i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.tight_layout()
    plt.savefig("reports/ablation_plot.png")
    print("Plot saved to reports/ablation_plot.png")
    
    # Save raw data
    df = pd.DataFrame(results)
    df.to_csv("reports/ablation_results.csv", index=False)
    print("Results saved to reports/ablation_results.csv")
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_ablation()

