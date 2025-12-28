from torch.utils.data import Dataset
from datasets import load_dataset
import random
import torch

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


class WikiAutoAssetMaskedDataset(WikiAutoAssetDataset):
    """
    Masked version of WikiAutoAssetDataset that applies random token masking
    to the complex text (style_input) before encoding.
    
    This prevents the model from learning identity shortcuts and should enable
    better disentanglement even with larger bottleneck dimensions.
    """
    def __init__(self, tokenizer, split="train", max_length=128, max_samples=None, 
                 data=None, mask_prob=0.4):
        """
        Args:
            tokenizer: Tokenizer instance
            split: Dataset split ('train', 'validation', 'test_asset')
            max_length: Maximum sequence length
            max_samples: Optional limit on number of samples
            data: Optional pre-loaded dataset
            mask_prob: Probability of masking each token (default: 0.4)
        """
        # Initialize parent class
        super().__init__(tokenizer, split, max_length, max_samples, data)
        self.mask_prob = mask_prob
        
        # Get special token IDs for exclusion from masking
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id for masked training")
    
    def __getitem__(self, idx):
        """
        Returns a data sample with masked complex text (style_input).
        The simple text (content_input) remains unmasked.
        """
        # Get base item from parent class
        item = self.dataset[idx]
        complex_text = item['source']
        
        # Logic for simple text based on split
        if self.split == 'train':
            if 'target' in item:
                simple_text = item['target']
            elif 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            else:
                simple_text = complex_text
        else:
            if 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            elif 'target' in item:
                simple_text = item['target']
            else:
                simple_text = complex_text
        
        # Tokenize complex text (will be masked)
        style_enc = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize simple text (unmasked)
        content_enc = self.tokenizer(
            simple_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Apply masking to style_input_ids (complex text)
        style_input_ids = style_enc['input_ids'].squeeze(0).clone()
        
        # Generate random mask: each token has mask_prob chance of being masked
        random_mask = torch.rand(style_input_ids.shape) < self.mask_prob
        
        # Exclude special tokens from masking
        special_tokens_mask = (
            (style_input_ids == self.cls_token_id) |
            (style_input_ids == self.sep_token_id) |
            (style_input_ids == self.pad_token_id)
        )
        
        # Only mask non-special tokens
        mask = random_mask & ~special_tokens_mask
        
        # Apply mask: replace masked tokens with mask_token_id
        style_input_ids[mask] = self.mask_token_id
        
        return {
            'style_input_ids': style_input_ids,
            'style_attention_mask': style_enc['attention_mask'].squeeze(0),
            'content_input_ids': content_enc['input_ids'].squeeze(0),
            'content_attention_mask': content_enc['attention_mask'].squeeze(0),
        }



class WikiAutoAssetMaskedDataset(WikiAutoAssetDataset):
    """
    Masked version of WikiAutoAssetDataset that applies random token masking
    to the complex text (style_input) before encoding.
    
    This prevents the model from learning identity shortcuts and should enable
    better disentanglement even with larger bottleneck dimensions.
    """
    def __init__(self, tokenizer, split="train", max_length=128, max_samples=None, 
                 data=None, mask_prob=0.4):
        """
        Args:
            tokenizer: Tokenizer instance
            split: Dataset split ('train', 'validation', 'test_asset')
            max_length: Maximum sequence length
            max_samples: Optional limit on number of samples
            data: Optional pre-loaded dataset
            mask_prob: Probability of masking each token (default: 0.4)
        """
        # Initialize parent class
        super().__init__(tokenizer, split, max_length, max_samples, data)
        self.mask_prob = mask_prob
        
        # Get special token IDs for exclusion from masking
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id for masked training")
    
    def __getitem__(self, idx):
        """
        Returns a data sample with masked complex text (style_input).
        The simple text (content_input) remains unmasked.
        """
        # Get base item from parent class
        item = self.dataset[idx]
        complex_text = item['source']
        
        # Logic for simple text based on split
        if self.split == 'train':
            if 'target' in item:
                simple_text = item['target']
            elif 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            else:
                simple_text = complex_text
        else:
            if 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            elif 'target' in item:
                simple_text = item['target']
            else:
                simple_text = complex_text
        
        # Tokenize complex text (will be masked)
        style_enc = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize simple text (unmasked)
        content_enc = self.tokenizer(
            simple_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Apply masking to style_input_ids (complex text)
        style_input_ids = style_enc['input_ids'].squeeze(0).clone()
        
        # Generate random mask: each token has mask_prob chance of being masked
        random_mask = torch.rand(style_input_ids.shape) < self.mask_prob
        
        # Exclude special tokens from masking
        special_tokens_mask = (
            (style_input_ids == self.cls_token_id) |
            (style_input_ids == self.sep_token_id) |
            (style_input_ids == self.pad_token_id)
        )
        
        # Only mask non-special tokens
        mask = random_mask & ~special_tokens_mask
        
        # Apply mask: replace masked tokens with mask_token_id
        style_input_ids[mask] = self.mask_token_id
        
        return {
            'style_input_ids': style_input_ids,
            'style_attention_mask': style_enc['attention_mask'].squeeze(0),
            'content_input_ids': content_enc['input_ids'].squeeze(0),
            'content_attention_mask': content_enc['attention_mask'].squeeze(0),
        }



class WikiAutoAssetMaskedDataset(WikiAutoAssetDataset):
    """
    Masked version of WikiAutoAssetDataset that applies random token masking
    to the complex text (style_input) before encoding.
    
    This prevents the model from learning identity shortcuts and should enable
    better disentanglement even with larger bottleneck dimensions.
    """
    def __init__(self, tokenizer, split="train", max_length=128, max_samples=None, 
                 data=None, mask_prob=0.4):
        """
        Args:
            tokenizer: Tokenizer instance
            split: Dataset split ('train', 'validation', 'test_asset')
            max_length: Maximum sequence length
            max_samples: Optional limit on number of samples
            data: Optional pre-loaded dataset
            mask_prob: Probability of masking each token (default: 0.4)
        """
        # Initialize parent class
        super().__init__(tokenizer, split, max_length, max_samples, data)
        self.mask_prob = mask_prob
        
        # Get special token IDs for exclusion from masking
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id for masked training")
    
    def __getitem__(self, idx):
        """
        Returns a data sample with masked complex text (style_input).
        The simple text (content_input) remains unmasked.
        """
        # Get base item from parent class
        item = self.dataset[idx]
        complex_text = item['source']
        
        # Logic for simple text based on split
        if self.split == 'train':
            if 'target' in item:
                simple_text = item['target']
            elif 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            else:
                simple_text = complex_text
        else:
            if 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            elif 'target' in item:
                simple_text = item['target']
            else:
                simple_text = complex_text
        
        # Tokenize complex text (will be masked)
        style_enc = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize simple text (unmasked)
        content_enc = self.tokenizer(
            simple_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Apply masking to style_input_ids (complex text)
        style_input_ids = style_enc['input_ids'].squeeze(0).clone()
        
        # Generate random mask: each token has mask_prob chance of being masked
        random_mask = torch.rand(style_input_ids.shape) < self.mask_prob
        
        # Exclude special tokens from masking
        special_tokens_mask = (
            (style_input_ids == self.cls_token_id) |
            (style_input_ids == self.sep_token_id) |
            (style_input_ids == self.pad_token_id)
        )
        
        # Only mask non-special tokens
        mask = random_mask & ~special_tokens_mask
        
        # Apply mask: replace masked tokens with mask_token_id
        style_input_ids[mask] = self.mask_token_id
        
        return {
            'style_input_ids': style_input_ids,
            'style_attention_mask': style_enc['attention_mask'].squeeze(0),
            'content_input_ids': content_enc['input_ids'].squeeze(0),
            'content_attention_mask': content_enc['attention_mask'].squeeze(0),
        }



class WikiAutoAssetMaskedDataset(WikiAutoAssetDataset):
    """
    Masked version of WikiAutoAssetDataset that applies random token masking
    to the complex text (style_input) before encoding.
    
    This prevents the model from learning identity shortcuts and should enable
    better disentanglement even with larger bottleneck dimensions.
    """
    def __init__(self, tokenizer, split="train", max_length=128, max_samples=None, 
                 data=None, mask_prob=0.4):
        """
        Args:
            tokenizer: Tokenizer instance
            split: Dataset split ('train', 'validation', 'test_asset')
            max_length: Maximum sequence length
            max_samples: Optional limit on number of samples
            data: Optional pre-loaded dataset
            mask_prob: Probability of masking each token (default: 0.4)
        """
        # Initialize parent class
        super().__init__(tokenizer, split, max_length, max_samples, data)
        self.mask_prob = mask_prob
        
        # Get special token IDs for exclusion from masking
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id for masked training")
    
    def __getitem__(self, idx):
        """
        Returns a data sample with masked complex text (style_input).
        The simple text (content_input) remains unmasked.
        """
        # Get base item from parent class
        item = self.dataset[idx]
        complex_text = item['source']
        
        # Logic for simple text based on split
        if self.split == 'train':
            if 'target' in item:
                simple_text = item['target']
            elif 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            else:
                simple_text = complex_text
        else:
            if 'references' in item and len(item['references']) > 0:
                simple_text = item['references'][0]
            elif 'target' in item:
                simple_text = item['target']
            else:
                simple_text = complex_text
        
        # Tokenize complex text (will be masked)
        style_enc = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Tokenize simple text (unmasked)
        content_enc = self.tokenizer(
            simple_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Apply masking to style_input_ids (complex text)
        style_input_ids = style_enc['input_ids'].squeeze(0).clone()
        
        # Generate random mask: each token has mask_prob chance of being masked
        random_mask = torch.rand(style_input_ids.shape) < self.mask_prob
        
        # Exclude special tokens from masking
        special_tokens_mask = (
            (style_input_ids == self.cls_token_id) |
            (style_input_ids == self.sep_token_id) |
            (style_input_ids == self.pad_token_id)
        )
        
        # Only mask non-special tokens
        mask = random_mask & ~special_tokens_mask
        
        # Apply mask: replace masked tokens with mask_token_id
        style_input_ids[mask] = self.mask_token_id
        
        return {
            'style_input_ids': style_input_ids,
            'style_attention_mask': style_enc['attention_mask'].squeeze(0),
            'content_input_ids': content_enc['input_ids'].squeeze(0),
            'content_attention_mask': content_enc['attention_mask'].squeeze(0),
        }
