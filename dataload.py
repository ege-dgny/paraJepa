from torch.utils.data import Dataset
from datasets import load_dataset
import random

class ParaphraseDataset(Dataset):

    def __init__(self, tokenizer, max_length=128, data=None):
        self.dataset = data
        if self.dataset is None:
            raise ValueError("Dataset is required")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        content_text = item['text']
        paraphrases = item['paraphrases']

        if len(paraphrases) > 0:
            style_text = random.choice(paraphrases)
        else:
            style_text = content_text

        style_enc = self.tokenizer(
            style_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        content_enc = self.tokenizer(
            content_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'style_input_ids': style_enc['input_ids'].squeeze(0),
            'style_attention_mask': style_enc['attention_mask'].squeeze(0),
            'content_input_ids': content_enc['input_ids'].squeeze(0),
            'content_attention_mask': content_enc['attention_mask'].squeeze(0),
        }