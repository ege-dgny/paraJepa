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

class WikiAutoAssetDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=128, max_samples=None, data=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # Load dataset if not provided
        if data is None:
            try:
                # We load the dataset here if not provided, but main.py typically handles splitting.
                # However, WikiAutoAsset has explicit splits.
                self.dataset = load_dataset("GEM/wiki_auto_asset_turk", split=split)
            except Exception as e:
                print(f"Error loading dataset: {e}")
                raise e
        else:
            self.dataset = data

        if max_samples is not None:
            print(f"SUBSET MODE: Limiting to {max_samples} samples.")
            limit = min(len(self.dataset), max_samples)
            self.dataset = self.dataset.select(range(limit))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        complex_text = item['source']

        # Logic for simple text based on split (from jepa_2.ipynb)
        if self.split == 'train':
            # Training set usually has 'target'
            if 'target' in item:
                simple_text = item['target']
            elif 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            else:
                simple_text = complex_text # Fallback
        else:
            # Validation/Test often have 'references' (list of valid simplifications)
            if 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            elif 'target' in item:
                simple_text = item['target']
            else:
                simple_text = complex_text # Fallback
        
        # Tokenize
        # Map: Complex -> Style (Input), Simple -> Content (Target)
        
        style_enc = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        content_enc = self.tokenizer(
            simple_text,
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
