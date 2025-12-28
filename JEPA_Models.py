import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer

class JEPAEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=768):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.projection(cls_token)

        return cls_token

class JEPAPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=16, output_dim=768, depth=3, funnel = False):
        super().__init__()

        # hidden_dim defaults to 128 to act as a bottleneck if not specified otherwise
        if funnel:
            layers = []
            layers.append(nn.Linear(input_dim, 192))
            layers.append(nn.LayerNorm(192))
            layers.append(nn.GELU())

            layers.append(nn.Linear(192, 48))
            layers.append(nn.LayerNorm(48))
            layers.append(nn.GELU())

            layers.append(nn.Linear(48,12))
            layers.append(nn.LayerNorm(12))
            layers.append(nn.GELU())

            layers.append(nn.Linear(12, output_dim))
            self.layers = nn.Sequential(*layers)


        else:
            layers = []
            
            # First layer: input_dim -> hidden_dim (Compression)
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            
            # Middle layers
            for _ in range(depth-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())

            # Last layer: hidden_dim -> output_dim (Expansion)
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)