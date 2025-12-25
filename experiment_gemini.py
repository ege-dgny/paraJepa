import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataload import WikiAutoAssetDataset
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

# --- 1. SETUP & RELOAD ---
print("🔄 Setting up Diagnostic Environment...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# The ParaJEPA class is already defined in memory from your training cell
model = ParaJEPA(
    model_name='roberta-base',
    hidden_dim=768,
    pred_depth=Config.pred_depth,
    pred_hidden_dim=Config.pred_hidden_dim
).to(device)

# --- FIX: Load the 'Hybrid' Checkpoint ---
drive_path = "/home/omnigibson/Desktop/cv/nlp/paraJepa/para_jepa_best_model_12_24_5.pt"

try:
    print(f"📂 Loading checkpoint from: {drive_path}")
    state_dict = torch.load(drive_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Checkpoint not found at {drive_path}")
    print("⚠️ Using untrained model (Expect random results).")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

model.eval()

# Load a tiny validation batch
print("📚 Loading validation sample...")
val_dataset = WikiAutoAssetDataset(tokenizer, split='validation', max_samples=100)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 2. THE DIAGNOSTIC FUNCTION ---
def diagnose_collapse(model, dataloader, device):
    print("\n" + "="*40)
    print("🔎 RUNNING HYBRID FORENSICS")
    print("="*40)

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
        # A. Encode
        # Note: These are raw, unbounded embeddings now
        input_emb = model.context_encoder(style_inputs['input_ids'], style_inputs['attention_mask'])
        target_emb = model.target_encoder(content_inputs['input_ids'], content_inputs['attention_mask'])

        # B. Predict
        raw_pred = model.predictor(input_emb)

        # C. Normalize for Angle Checks
        pred_norm = F.normalize(raw_pred, p=2, dim=1)
        input_norm = F.normalize(input_emb, p=2, dim=1)
        target_norm = F.normalize(target_emb, p=2, dim=1)

        # --- METRICS ---

        # 1. Angles (Cosine)
        baseline_sim = F.cosine_similarity(input_norm, target_norm, dim=-1).mean().item()
        pred_target_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1).mean().item()
        pred_input_sim = F.cosine_similarity(pred_norm, input_norm, dim=-1).mean().item()

        # 2. Magnitudes (VICReg Success Check)
        # We check Context specifically because that's where we applied the fix!
        context_std = torch.sqrt(input_emb.var(dim=0) + 1e-4).mean().item()
        pred_std = torch.sqrt(raw_pred.var(dim=0) + 1e-4).mean().item()

        # --- REPORTING ---
        print(f"📊 BATCH STATISTICS:")
        print(f"  --- Angles (Cosine) ---")
        print(f"  1. Natural Similarity : {baseline_sim:.5f}")
        print(f"  2. Model Performance  : {pred_target_sim:.5f} (Higher is better)")
        print(f"  3. Identity Check     : {pred_input_sim:.5f}")

        print(f"\n  --- Expansion Check (Target: ~1.0) ---")
        print(f"  4. Context Std Dev    : {context_std:.5f}  <-- DID THE DEAD GRADIENT FIX WORK?")
        print(f"  5. Predictor Std Dev  : {pred_std:.5f}")

        print("\n🧐 DIAGNOSIS:")

        # Check Expansion
        if context_std < 0.1:
            print("🚨 EXPANSION FAILED: Context Variance is still tiny. Weights didn't update?")
        elif context_std > 0.5:
            print("✅ EXPANSION SUCCESS: Context Encoder is exploding the variance as requested!")

        # Check Learning
        if pred_target_sim > baseline_sim:
            print("🚀 LEARNING CONFIRMED: The model beat the natural baseline!")
            gain = pred_target_sim - baseline_sim
            print(f"   Gain: +{gain:.5f}")
        else:
             print("⚠️ STRUGGLING: Model has expanded, but hasn't beaten the baseline yet.")

    return input_emb, target_emb, raw_pred

# --- 3. EXECUTE ---
input_e, target_e, pred_e = diagnose_collapse(model, val_loader, device)