import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tabulate import tabulate

from dataload import WikiAutoAssetDataset
from para_jepa_train import ParaJEPA
from config import Config
from evaluation_utils import run_full_evaluation
from main import get_device, set_seed

class RobertaBaseline(nn.Module):
    """
    Baseline 1: Frozen RoBERTa (CLS token)
    Represents raw pre-training capabilities.
    """
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token
            return outputs.last_hidden_state[:, 0, :]

class ContrastiveBaseline(nn.Module):
    """
    Baseline 2: Contrastive Model (Sentence-BERT style)
    Uses a pre-trained sentence transformer with Mean Pooling.
    """
    def __init__(self, model_name='sentence-transformers/all-distilroberta-v1'):
        super().__init__()
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except OSError:
            print(f"Could not load {model_name}, falling back to roberta-base")
            self.model = AutoModel.from_pretrained('roberta-base')
            
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            
            # Mean Pooling - Take attention mask into account for correct averaging
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            return sum_embeddings / sum_mask

def run_evaluation_for_model(model_name_display, model, train_loader, test_loader, device, config, train_dataset=None, test_dataset=None):
    print(f"\n\n{'#'*20} Evaluating {model_name_display} {'#'*20}")
    model = model.to(device)
    model.eval()
    
    # Run evaluation
    # Pass datasets explicitly to use label hashing
    results = run_full_evaluation(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=AutoTokenizer.from_pretrained(config.model_name),
        device=device,
        max_length=config.max_length
    )
    
    # Extract key metrics
    metrics = {
        "Model": model_name_display,
        "Content Acc (↑)": f"{results['content_accuracy']*100:.1f}%",
        "Style Acc (↓)": f"{results['style_accuracy']*100:.1f}%",
        "Disentanglement (H-Mean)": "N/A", # Calculate later
        "Recall@1": f"{results['retrieval']['R@1']:.4f}"
    }
    
    # Calculate Disentanglement Score (Harmonic Mean of Content Acc and (1-Style Acc))
    # Note: 1-Style Acc because lower style acc is better
    content_score = results['content_accuracy']
    style_inv_score = 1.0 - results['style_accuracy']
    
    if content_score > 0 and style_inv_score > 0:
        h_mean = 2 * (content_score * style_inv_score) / (content_score + style_inv_score)
        metrics["Disentanglement (H-Mean)"] = f"{h_mean:.3f}"
    else:
        metrics["Disentanglement (H-Mean)"] = "0.000"
        
    return metrics

def main():
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    print("Preparing Evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 1. Load Data
    print("Loading Datasets...")
    
    # Test Set
    try:
        test_set = WikiAutoAssetDataset(tokenizer, split='test_asset', max_length=config.max_length)
    except Exception:
        print("Using validation set as proxy for test set")
        test_set = WikiAutoAssetDataset(tokenizer, split='validation', max_length=config.max_length)
        
    test_loader = DataLoader(
        test_set, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )

    # Train Set (Subset for Probe Training)
    # We use a subset of 10,000 samples to make probe training faster
    try:
        full_train_set = WikiAutoAssetDataset(tokenizer, split='train', max_length=config.max_length)
        indices = np.random.choice(len(full_train_set), size=min(10000, len(full_train_set)), replace=False)
        train_subset = Subset(full_train_set, indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
    except Exception as e:
        print(f"Error loading train set: {e}")
        return
    
    all_metrics = []
    
    # 2. Evaluate Baseline 1: RoBERTa Frozen
    roberta = RobertaBaseline(model_name='roberta-base')
    metrics_roberta = run_evaluation_for_model("RoBERTa (Frozen)", roberta, train_loader, test_loader, device, config, train_dataset=train_subset, test_dataset=test_set)
    all_metrics.append(metrics_roberta)
    del roberta
    torch.cuda.empty_cache()
    
    # 3. Evaluate Baseline 2: Contrastive
    # Using all-distilroberta-v1 as a strong sentence embedding baseline
    contrastive = ContrastiveBaseline(model_name='sentence-transformers/all-distilroberta-v1')
    metrics_contrastive = run_evaluation_for_model("SimCLR (Contrastive)", contrastive, train_loader, test_loader, device, config, train_dataset=train_subset, test_dataset=test_set)
    all_metrics.append(metrics_contrastive)
    del contrastive
    torch.cuda.empty_cache()
    
    # 4. Evaluate ParaJEPA (Your Model)
    print("\nLoading ParaJEPA...")
    parajepa = ParaJEPA(
        model_name=config.model_name,
        hidden_dim=config.hidden_dim,
        pred_depth=config.pred_depth
    )
    try:
        parajepa.load_state_dict(torch.load("para_jepa_best_model.pt", map_location=device))
        print("Loaded ParaJEPA weights.")
        metrics_jepa = run_evaluation_for_model("ParaJEPA (Yours)", parajepa, train_loader, test_loader, device, config, train_dataset=train_subset, test_dataset=test_set)
        all_metrics.append(metrics_jepa)
    except FileNotFoundError:
        print("ParaJEPA weights not found! Skipping ParaJEPA evaluation.")
    
    del parajepa
    torch.cuda.empty_cache()
    
    # 5. Print "The Money Table"
    print("\n" + "="*80)
    print("THE MONEY TABLE (Final Results)")
    print("="*80)
    
    df = pd.DataFrame(all_metrics)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Save to CSV
    df.to_csv("reports/money_table_results.csv", index=False)
    print("\nResults saved to reports/money_table_results.csv")

if __name__ == "__main__":
    main()