import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from src.JEPA_Models import JEPAEncoder, JEPAPredictor
import copy
from tqdm import tqdm
import json
import wandb

class ParaJEPA(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196, funnel=False):
        super().__init__()
        self.ema_decay = ema_decay

        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(
            input_dim=hidden_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=hidden_dim,
            depth=pred_depth,
            funnel=funnel,
        )
    
    def forward(self, style_input, content_input):
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        prediction = self.predictor(context_embeddings)

        loss_pred = F.mse_loss(prediction, target_embeddings)

        std_pred = torch.sqrt(prediction.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target_embeddings.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        pred_norm = prediction - prediction.mean(dim=0)
        cov_pred = (pred_norm.T @ pred_norm) / (prediction.size(0) - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / prediction.size(1)

        loss = loss_pred + (25.0 * std_loss) + (1.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight


def evaluation(model, dataloader, device, desc='Validation'):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            style_inputs = {
                'input_ids': batch['style_input_ids'].to(device),
                'attention_mask': batch['style_attention_mask'].to(device),
            }

            content_inputs = {
                'input_ids': batch['content_input_ids'].to(device),
                'attention_mask': batch['content_attention_mask'].to(device),
            }

            loss, prediction, target_embeddings = model(style_inputs, content_inputs)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    cosine_sim = F.cosine_similarity(prediction, target_embeddings, dim=-1).mean().item()
    return avg_loss, cosine_sim

def train_para_jepa(model, train_loader, valid_loader, optimizer, device, epochs=10, use_wandb=False):
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_cosine_sim': [],
        'epochs': [],
    }
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            style_input = {
                'input_ids': batch['style_input_ids'].to(device),
                'attention_mask': batch['style_attention_mask'].to(device),
            }

            content_input = {
                'input_ids': batch['content_input_ids'].to(device),
                'attention_mask': batch['content_attention_mask'].to(device),
            }

            loss, prediction, target_embeddings = model(style_input, content_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target_ema()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"epoch {epoch+1}/{epochs} completed, training loss: {avg_loss:.4f}")

        val_loss, val_cosine_sim = evaluation(model, valid_loader, device, desc="Validation")
        print(f"epoch {epoch+1}/{epochs} completed, validation loss: {val_loss:.4f}, validation cosine similarity: {val_cosine_sim:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"para_jepa_best_model.pt")

        training_history['train_loss'].append(avg_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_cosine_sim'].append(val_cosine_sim)
        training_history['epochs'].append(epoch + 1)
        
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'val_cosine_sim': val_cosine_sim,
                'best_val_loss': best_val_loss
            })
        
        with open('training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)

    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': avg_loss,
        'final_val_loss': val_loss,
        'final_val_cosine_sim': val_cosine_sim
    }
            
def test_run(model, test_loader, device):
    print('Running Final Test')

    try:
        model.load_state_dict(torch.load(f"para_jepa_best_model.pt", map_location=device))
    except FileNotFoundError:
        print('No Checkpoint Found')
        return None
    
    test_loss, test_cosine_sim = evaluation(model, test_loader, device, desc="Test")
    print(f"Test loss: {test_loss:.4f}, Test cosine similarity: {test_cosine_sim:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_cosine_sim': test_cosine_sim
    }
