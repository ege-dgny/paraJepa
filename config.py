class Config:
    model_name = 'roberta-base'
    hidden_dim = 768
    ema_decay = 0.99
    pred_depth = 3
    pred_hidden_dim = 128
    pred_dim = 16
    
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 5
    
    max_length = 128
    num_workers = 12
    
    mask_prob = 0.4
    use_masking = False
    
    seed = 11
