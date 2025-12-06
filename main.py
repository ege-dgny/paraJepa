from dataload import ParaphraseDataset, test_run
from para_jepa_train import ParaJEPA, train_para_jepa
from JEPA_Models import JEPAEncoder, JEPAPredictor
from transformers import AutoTokenizer
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_device():
    if torch.backends.cuda.is_built():
        print("Using CUDA")
        device =torch.device('cuda')
    elif torch.backends.mps.is_built():
        print("Using MPS")
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print("Using CPU")
        raise Exception("No GPU found")
    return device

def main():

    device = get_device()
    batch_size = 16
    lr = 2e-5
    epochs = 10
    model_name = 'roberta-base'
    hidden_dim = 768
    ema_decay = 0.996
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset('humarin/chatgpt-paraphrases', split='train')
    train_test = dataset.train_test_split(test_size=0.2, seed=23) # 80% for train
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=23) # 10% for vali, 10% for test

    train_set = ParaphraseDataset(tokenizer, data=train_test['train'])
    valid_set = ParaphraseDataset(tokenizer, data=test_valid['train'])
    test_set = ParaphraseDataset(tokenizer, data=test_valid['test'])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = ParaJEPA(model_name=model_name, hidden_dim=hidden_dim, ema_decay=ema_decay).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_para_jepa(model=model, train_loader=train_loader, validation_loader=valid_loader, optimizer=optimizer, device=device, epochs=epochs)
    test_run(model=model, test_loader=test_loader, device=device)
