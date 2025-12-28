import torch
import torch.nn as nn
import torch.nn.functional as F
from JEPA_Models import JEPAEncoder, JEPAPredictor
import copy

class ParaJEPAConditional(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196):
        super().__init__()
        self.ema_decay = ema_decay

        # Initialize context encoder
        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        # Initialize target encoder (no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False # Disable gradient update for target encoder

        # Predictor takes context embedding + 1 scalar (length)
        self.predictor = JEPAPredictor(
            input_dim=hidden_dim + 1,  # +1 for length feature
            hidden_dim=pred_hidden_dim, # Bottleneck
            output_dim=hidden_dim,
            depth=pred_depth,
        )
    
    def forward(self, style_input, content_input):
        # 1. Encode Context (Complex/Style Input)
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        # 2. Encode Target (Simple/Content Input)
        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        # 3. Calculate Normalized Length of Target
        # Attention mask sum gives the number of real tokens. We normalize by max_length (e.g. 128)
        # to keep the value in a reasonable range (0.0 - 1.0).
        # We use the target's length because that's what we want the predictor to output (conceptually),
        # but since we are giving it as input, we are "explaining away" the length information.
        target_len = content_input['attention_mask'].sum(dim=1, keepdim=True).float()
        # Normalize by an arbitrary factor (e.g., 128 or just let it be raw count if simple)
        # Using 128.0 ensures it's roughly 0-1, which matches neural network activations better.
        target_len_norm = target_len / 128.0 

        # 4. Concatenate Length to Context Embedding
        # context_embeddings: [B, 768]
        # target_len_norm: [B, 1]
        # predictor_input: [B, 769]
        predictor_input = torch.cat([context_embeddings, target_len_norm], dim=1)

        # 5. Predict Target Embedding
        prediction = self.predictor(predictor_input)

        loss_pred = F.mse_loss(prediction, target_embeddings)

        # --- ANTI-COLLAPSE REGULARIZATION (VICReg-style) ---
        # 1. Variance Loss
        std_pred = torch.sqrt(prediction.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target_embeddings.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # 2. Covariance Loss
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        pred_norm = prediction - prediction.mean(dim=0)
        cov_pred = (pred_norm.T @ pred_norm) / (prediction.size(0) - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / prediction.size(1)

        # Total Loss
        # Using the weights from your "disentanglement" experiment (5.0, 5.0)
        loss = loss_pred + (5.0 * std_loss) + (5.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight



import torch.nn as nn
import torch.nn.functional as F
from JEPA_Models import JEPAEncoder, JEPAPredictor
import copy

class ParaJEPAConditional(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196):
        super().__init__()
        self.ema_decay = ema_decay

        # Initialize context encoder
        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        # Initialize target encoder (no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False # Disable gradient update for target encoder

        # Predictor takes context embedding + 1 scalar (length)
        self.predictor = JEPAPredictor(
            input_dim=hidden_dim + 1,  # +1 for length feature
            hidden_dim=pred_hidden_dim, # Bottleneck
            output_dim=hidden_dim,
            depth=pred_depth,
        )
    
    def forward(self, style_input, content_input):
        # 1. Encode Context (Complex/Style Input)
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        # 2. Encode Target (Simple/Content Input)
        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        # 3. Calculate Normalized Length of Target
        # Attention mask sum gives the number of real tokens. We normalize by max_length (e.g. 128)
        # to keep the value in a reasonable range (0.0 - 1.0).
        # We use the target's length because that's what we want the predictor to output (conceptually),
        # but since we are giving it as input, we are "explaining away" the length information.
        target_len = content_input['attention_mask'].sum(dim=1, keepdim=True).float()
        # Normalize by an arbitrary factor (e.g., 128 or just let it be raw count if simple)
        # Using 128.0 ensures it's roughly 0-1, which matches neural network activations better.
        target_len_norm = target_len / 128.0 

        # 4. Concatenate Length to Context Embedding
        # context_embeddings: [B, 768]
        # target_len_norm: [B, 1]
        # predictor_input: [B, 769]
        predictor_input = torch.cat([context_embeddings, target_len_norm], dim=1)

        # 5. Predict Target Embedding
        prediction = self.predictor(predictor_input)

        loss_pred = F.mse_loss(prediction, target_embeddings)

        # --- ANTI-COLLAPSE REGULARIZATION (VICReg-style) ---
        # 1. Variance Loss
        std_pred = torch.sqrt(prediction.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target_embeddings.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # 2. Covariance Loss
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        pred_norm = prediction - prediction.mean(dim=0)
        cov_pred = (pred_norm.T @ pred_norm) / (prediction.size(0) - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / prediction.size(1)

        # Total Loss
        # Using the weights from your "disentanglement" experiment (5.0, 5.0)
        loss = loss_pred + (5.0 * std_loss) + (5.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight



import torch.nn as nn
import torch.nn.functional as F
from JEPA_Models import JEPAEncoder, JEPAPredictor
import copy

class ParaJEPAConditional(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196):
        super().__init__()
        self.ema_decay = ema_decay

        # Initialize context encoder
        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        # Initialize target encoder (no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False # Disable gradient update for target encoder

        # Predictor takes context embedding + 1 scalar (length)
        self.predictor = JEPAPredictor(
            input_dim=hidden_dim + 1,  # +1 for length feature
            hidden_dim=pred_hidden_dim, # Bottleneck
            output_dim=hidden_dim,
            depth=pred_depth,
        )
    
    def forward(self, style_input, content_input):
        # 1. Encode Context (Complex/Style Input)
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        # 2. Encode Target (Simple/Content Input)
        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        # 3. Calculate Normalized Length of Target
        # Attention mask sum gives the number of real tokens. We normalize by max_length (e.g. 128)
        # to keep the value in a reasonable range (0.0 - 1.0).
        # We use the target's length because that's what we want the predictor to output (conceptually),
        # but since we are giving it as input, we are "explaining away" the length information.
        target_len = content_input['attention_mask'].sum(dim=1, keepdim=True).float()
        # Normalize by an arbitrary factor (e.g., 128 or just let it be raw count if simple)
        # Using 128.0 ensures it's roughly 0-1, which matches neural network activations better.
        target_len_norm = target_len / 128.0 

        # 4. Concatenate Length to Context Embedding
        # context_embeddings: [B, 768]
        # target_len_norm: [B, 1]
        # predictor_input: [B, 769]
        predictor_input = torch.cat([context_embeddings, target_len_norm], dim=1)

        # 5. Predict Target Embedding
        prediction = self.predictor(predictor_input)

        loss_pred = F.mse_loss(prediction, target_embeddings)

        # --- ANTI-COLLAPSE REGULARIZATION (VICReg-style) ---
        # 1. Variance Loss
        std_pred = torch.sqrt(prediction.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target_embeddings.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # 2. Covariance Loss
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        pred_norm = prediction - prediction.mean(dim=0)
        cov_pred = (pred_norm.T @ pred_norm) / (prediction.size(0) - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / prediction.size(1)

        # Total Loss
        # Using the weights from your "disentanglement" experiment (5.0, 5.0)
        loss = loss_pred + (5.0 * std_loss) + (5.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight



import torch.nn as nn
import torch.nn.functional as F
from JEPA_Models import JEPAEncoder, JEPAPredictor
import copy

class ParaJEPAConditional(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768, ema_decay=0.996, pred_depth=3, pred_hidden_dim=196):
        super().__init__()
        self.ema_decay = ema_decay

        # Initialize context encoder
        self.context_encoder = JEPAEncoder(model_name, hidden_dim)
        # Initialize target encoder (no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False # Disable gradient update for target encoder

        # Predictor takes context embedding + 1 scalar (length)
        self.predictor = JEPAPredictor(
            input_dim=hidden_dim + 1,  # +1 for length feature
            hidden_dim=pred_hidden_dim, # Bottleneck
            output_dim=hidden_dim,
            depth=pred_depth,
        )
    
    def forward(self, style_input, content_input):
        # 1. Encode Context (Complex/Style Input)
        context_embeddings = self.context_encoder(style_input['input_ids'], style_input['attention_mask'])

        # 2. Encode Target (Simple/Content Input)
        with torch.no_grad():
            target_embeddings = self.target_encoder(content_input['input_ids'], content_input['attention_mask'])

        # 3. Calculate Normalized Length of Target
        # Attention mask sum gives the number of real tokens. We normalize by max_length (e.g. 128)
        # to keep the value in a reasonable range (0.0 - 1.0).
        # We use the target's length because that's what we want the predictor to output (conceptually),
        # but since we are giving it as input, we are "explaining away" the length information.
        target_len = content_input['attention_mask'].sum(dim=1, keepdim=True).float()
        # Normalize by an arbitrary factor (e.g., 128 or just let it be raw count if simple)
        # Using 128.0 ensures it's roughly 0-1, which matches neural network activations better.
        target_len_norm = target_len / 128.0 

        # 4. Concatenate Length to Context Embedding
        # context_embeddings: [B, 768]
        # target_len_norm: [B, 1]
        # predictor_input: [B, 769]
        predictor_input = torch.cat([context_embeddings, target_len_norm], dim=1)

        # 5. Predict Target Embedding
        prediction = self.predictor(predictor_input)

        loss_pred = F.mse_loss(prediction, target_embeddings)

        # --- ANTI-COLLAPSE REGULARIZATION (VICReg-style) ---
        # 1. Variance Loss
        std_pred = torch.sqrt(prediction.var(dim=0) + 0.0001)
        std_target = torch.sqrt(target_embeddings.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_pred)) + torch.mean(F.relu(1 - std_target))

        # 2. Covariance Loss
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        pred_norm = prediction - prediction.mean(dim=0)
        cov_pred = (pred_norm.T @ pred_norm) / (prediction.size(0) - 1)
        cov_loss = off_diagonal(cov_pred).pow_(2).sum() / prediction.size(1)

        # Total Loss
        # Using the weights from your "disentanglement" experiment (5.0, 5.0)
        loss = loss_pred + (5.0 * std_loss) + (5.0 * cov_loss)

        return loss, prediction, target_embeddings
    
    @torch.no_grad()
    def update_target_ema(self):
        for param, ema_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            prev_weight, next_weight = ema_param.data, param.data
            ema_param.data = self.ema_decay * prev_weight + (1 - self.ema_decay) * next_weight







