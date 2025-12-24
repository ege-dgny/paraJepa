class Config:
    # Model hyperparameters
    model_name = 'roberta-base'
    hidden_dim = 768
    ema_decay = 0.99
    pred_depth = 3
    
    # Training hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 5
    
    # Data hyperparameters
    max_length = 128
    num_workers = 12
    
    # System
    seed = 11