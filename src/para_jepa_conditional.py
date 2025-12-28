import torch
import torch.nn as nn
import torch.nn.functional as F
from src.JEPA_Models import JEPAEncoder, JEPAPredictor
import copy

class ParaJEPAConditional(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196):
        super().__init__()
        self.ema_decay = ema_decay

        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = JEPAPredictor(
            input_dim=hidden_dim + 1,
            hidden_dim=pred_hidden_dim,
            output_dim=hidden_dim,
            depth=pred_depth,
        )
    
    def forward(self, style_input, content_input):
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        target_len = content_input['attention_mask'].sum(dim=1, keepdim=True).float()
        target_len_norm = target_len / 128.0 

        predictor_input = torch.cat([context_embeddings, target_len_norm], dim=1)

        prediction = self.predictor(predictor_input)

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

        loss = loss_pred + (5.0 * std_loss) + (5.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight
