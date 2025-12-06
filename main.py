from dataload import ParaphraseDataset, test_run
from para_jepa_train import ParaJEPA, train_para_jepa
from JEPA_Models import JEPAEncoder, JEPAPredictor
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
import random
import numpy as np
from config import Config
import os

def setup_ddp():
    """Initialize DDP if multiple GPUs are available, otherwise return None"""
    if not torch.cuda.is_available():
        return None, None, None
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return None, None, None
    
    # Check if DDP is already initialized (from torchrun)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    else:
        # Initialize manually for local multi-GPU
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        rank = 0
        world_size = num_gpus
        local_rank = 0
    
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, device

def cleanup_ddp():
    """Clean up DDP process group"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def get_device(use_ddp=False, local_rank=None):
    """Get device, handling both single GPU and DDP cases"""
    if use_ddp and local_rank is not None:
        return torch.device(f'cuda:{local_rank}')
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using CUDA with {num_gpus} GPU(s)")
        device = torch.device('cuda')
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU available)")
    return device

def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config = None):
    if config is None:
        config = Config()
    
    # Setup DDP if multiple GPUs available
    rank, world_size, ddp_device = setup_ddp()
    use_ddp = (rank is not None and world_size is not None)
    
    if use_ddp:
        device = ddp_device
        is_main_process = (rank == 0)
        if is_main_process:
            print(f"Using DDP with {world_size} GPUs")
    else:
        device = get_device()
        is_main_process = True
    
    model_name = config.model_name
    hidden_dim = config.hidden_dim
    ema_decay = config.ema_decay
    pred_depth = config.pred_depth
    batch_size = config.batch_size
    lr = config.learning_rate
    weight_decay = config.weight_decay
    epochs = config.epochs
    max_length = config.max_length
    seed = set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset('humarin/chatgpt-paraphrases', split='train')
    train_test = dataset.train_test_split(test_size=0.2, seed=seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=seed)

    train_set = ParaphraseDataset(tokenizer, max_length=max_length, data=train_test['train'])
    valid_set = ParaphraseDataset(tokenizer, max_length=max_length, data=test_valid['train'])
    test_set = ParaphraseDataset(tokenizer, max_length=max_length, data=test_valid['test'])

    # Create data loaders with DistributedSampler if using DDP
    if use_ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            sampler=train_sampler,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id + rank * 1000)
        )
        valid_loader = DataLoader(
            valid_set, 
            batch_size=batch_size, 
            sampler=valid_sampler,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id + rank * 1000)
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            sampler=test_sampler,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id + rank * 1000)
        )
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    model = ParaJEPA(model_name=model_name, hidden_dim=hidden_dim, ema_decay=ema_decay, pred_depth=pred_depth).to(device)
    
    # Wrap with DDP if using multi-GPU
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        train_para_jepa(
            model=model, 
            train_loader=train_loader, 
            valid_loader=valid_loader, 
            optimizer=optimizer, 
            device=device, 
            epochs=epochs,
            use_ddp=use_ddp,
            is_main_process=is_main_process
        )
        if is_main_process:
            test_run(model=model, test_loader=test_loader, device=device)
    finally:
        if use_ddp:
            cleanup_ddp()

if __name__ == '__main__':
    main()
