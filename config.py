class Config:
    # Model hyperparameters
    model_name = 'roberta-base'
    hidden_dim = 768
    ema_decay = 0.99
    pred_depth = 3
    pred_hidden_dim = 128
    pred_dim = 16
    
    # Training hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 5
    
    # Data hyperparameters
    max_length = 128
    num_workers = 12
    
    # Masking hyperparameters (for masked JEPA training)
    mask_prob = 0.4  # Probability of masking a token (0.0 to 1.0)
    use_masking = False  # Flag to enable/disable masking in existing pipeline
    
    # System
    seed = 11