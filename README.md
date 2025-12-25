
---
tags:
- pytorch
- style-transfer
- representation-learning
- jepa
---

# ParaJEPA: Joint-Embedding Predictive Architecture for Text Style Transfer

This model uses a JEPA architecture to disentangle **Content** and **Style** representations from text.

## Usage

To use this model, you need to download the code files along with the weights.

import torch
from para_jepa_train import ParaJEPA
from config import Config

# 1. Initialize Model Architecture
config = Config()
model = ParaJEPA(
    model_name=config.model_name,
    hidden_dim=config.hidden_dim,
    pred_depth=config.pred_depth
)

# 2. Load Weights
# (Download para_jepa_best_model.pt manually or via hf_hub_download)
model.load_state_dict(torch.load("para_jepa_best_model.pt"))
model.eval()## Architecture
- **Backbone**: RoBERTa-base
- **Method**: Predictive Bottleneck (JEPA)
- **Objective**: Disentangle Style (Paraphrase) from Content (Meaning)
