import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.dataload import WikiAutoAssetDataset
from src.para_jepa_train import ParaJEPA, train_para_jepa, test_run
from src.JEPA_Models import JEPAEncoder, JEPAPredictor
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import numpy as np
from config import Config

print("Setting up Diagnostic Environment...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

model = ParaJEPA(
    model_name='roberta-base',
    hidden_dim=768,
    pred_depth=Config.pred_depth,
    pred_hidden_dim=Config.pred_hidden_dim
).to(device)

drive_path = "/home/omnigibson/Desktop/cv/nlp/paraJepa/para_jepa_best_model_12_24_5.pt"

try:
    print(f"Loading checkpoint from: {drive_path}")
    state_dict = torch.load(drive_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Checkpoint not found at {drive_path}")
    print("Using untrained model (Expect random results).")
except Exception as e:
    print(f"Error loading model: {e}")

model.eval()

print("Loading validation sample...")
val_dataset = WikiAutoAssetDataset(tokenizer, split='validation', max_samples=100)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def diagnose_collapse(model, dataloader, device):
    print("-" * 30)
    print("Running Hybrid Forensics")
    print("-" * 30)

    batch = next(iter(dataloader))

    style_inputs = {
        'input_ids': batch['style_input_ids'].to(device),
        'attention_mask': batch['style_attention_mask'].to(device),
    }
    content_inputs = {
        'input_ids': batch['content_input_ids'].to(device),
        'attention_mask': batch['content_attention_mask'].to(device),
    }

    with torch.no_grad():
        input_emb = model.context_encoder(style_inputs['input_ids'], style_inputs['attention_mask'])
        target_emb = model.target_encoder(content_inputs['input_ids'], content_inputs['attention_mask'])

        raw_pred = model.predictor(input_emb)

        pred_norm = F.normalize(raw_pred, p=2, dim=1)
        input_norm = F.normalize(input_emb, p=2, dim=1)
        target_norm = F.normalize(target_emb, p=2, dim=1)

        baseline_sim = F.cosine_similarity(input_norm, target_norm, dim=-1).mean().item()
        pred_target_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1).mean().item()
        pred_input_sim = F.cosine_similarity(pred_norm, input_norm, dim=-1).mean().item()

        context_std = torch.sqrt(input_emb.var(dim=0) + 1e-4).mean().item()
        pred_std = torch.sqrt(raw_pred.var(dim=0) + 1e-4).mean().item()

        print(f"Batch Statistics:")
        print(f"  --- Angles (Cosine) ---")
        print(f"  1. Natural Similarity : {baseline_sim:.5f}")
        print(f"  2. Model Performance  : {pred_target_sim:.5f} (Higher is better)")
        print(f"  3. Identity Check     : {pred_input_sim:.5f}")

        print(f"\n  --- Expansion Check (Target: ~1.0) ---")
        print(f"  4. Context Std Dev    : {context_std:.5f}")
        print(f"  5. Predictor Std Dev  : {pred_std:.5f}")

        print("\nDiagnosis:")

        if context_std < 0.1:
            print("Expansion Failed: Context Variance is still tiny. Weights didn't update?")
        elif context_std > 0.5:
            print("Expansion Success: Context Encoder is increasing variance.")

        if pred_target_sim > baseline_sim:
            print("Learning Confirmed: The model beat the natural baseline!")
            gain = pred_target_sim - baseline_sim
            print(f"   Gain: +{gain:.5f}")
        else:
             print("Struggling: Model has expanded, but hasn't beaten the baseline yet.")

    return input_emb, target_emb, raw_pred

input_e, target_e, pred_e = diagnose_collapse(model, val_loader, device)
