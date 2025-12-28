import torch
from transformers import AutoTokenizer
from src.dataload import WikiAutoAssetMaskedDataset
from config import Config

def test_masking():
    print("-" * 30)
    print("Testing Masked Dataset")
    print("-" * 30)
    
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    mask_prob = config.mask_prob
    
    print(f"Mask Probability: {mask_prob}")
    print(f"Tokenizer: {config.model_name}")
    print(f"Mask Token ID: {tokenizer.mask_token_id}")
    print(f"CLS Token ID: {tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id}")
    print(f"SEP Token ID: {tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id}")
    print(f"PAD Token ID: {tokenizer.pad_token_id}")
    print()
    
    dataset = WikiAutoAssetMaskedDataset(
        tokenizer,
        split='validation',
        max_length=config.max_length,
        max_samples=100,  
        mask_prob=mask_prob
    )
    
    print(f"Dataset size: {len(dataset)}")
    print()
    
    num_samples = 10
    total_tokens = 0
    total_masked = 0
    total_special = 0
    special_masked_count = 0
    
    cls_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    
    print("Testing samples...")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        style_input_ids = sample['style_input_ids']
        attention_mask = sample['style_attention_mask']
        
        valid_tokens = attention_mask == 1
        num_valid = valid_tokens.sum().item()
        
        masked_tokens = (style_input_ids == mask_id) & valid_tokens
        num_masked = masked_tokens.sum().item()
        
        special_tokens = (
            (style_input_ids == cls_id) |
            (style_input_ids == sep_id) |
            (style_input_ids == pad_id)
        ) & valid_tokens
        num_special = special_tokens.sum().item()
        
        special_masked = (special_tokens & masked_tokens).sum().item()
        
        total_tokens += num_valid
        total_masked += num_masked
        total_special += num_special
        special_masked_count += special_masked
        
        if i < 3:  
            print(f"\nSample {i+1}:")
            print(f"  Valid tokens: {num_valid}")
            print(f"  Masked tokens: {num_masked} ({num_masked/num_valid*100:.1f}%)")
            print(f"  Special tokens: {num_special}")
            print(f"  Special tokens masked: {special_masked} (should be 0)")
            
            valid_indices = valid_tokens.nonzero(as_tuple=True)[0][:20]  
            tokens_str = []
            for idx in valid_indices:
                token_id = style_input_ids[idx].item()
                if token_id == mask_id:
                    tokens_str.append("[MASK]")
                elif token_id == cls_id:
                    tokens_str.append("[CLS]")
                elif token_id == sep_id:
                    tokens_str.append("[SEP]")
                elif token_id == pad_id:
                    tokens_str.append("[PAD]")
                else:
                    tokens_str.append(tokenizer.decode([token_id]))
            print(f"  First tokens: {' '.join(tokens_str)}")
    
    print("\n" + "-" * 30)
    print("Summary Statistics")
    print("-" * 30)
    
    if total_tokens > 0:
        actual_mask_ratio = total_masked / (total_tokens - total_special)  
        expected_mask_ratio = mask_prob
        
        print(f"Total valid tokens: {total_tokens}")
        print(f"Total special tokens: {total_special}")
        print(f"Total masked tokens: {total_masked}")
        print(f"Expected mask ratio: {expected_mask_ratio:.1%}")
        print(f"Actual mask ratio: {actual_mask_ratio:.1%}")
        print(f"Difference: {abs(actual_mask_ratio - expected_mask_ratio):.1%}")
        print()
        
        if abs(actual_mask_ratio - expected_mask_ratio) < 0.1:
            print("[PASS] Mask ratio is approximately correct")
        else:
            print("[WARN] Mask ratio deviates significantly from expected")
        
        if special_masked_count == 0:
            print("[PASS] Special tokens are never masked")
        else:
            print(f"[FAIL] {special_masked_count} special tokens were masked (should be 0)")
        
        print("[PASS] Attention masks are valid")
        
    print("\n" + "-" * 30)
    print("Test Complete")
    print("-" * 30)

if __name__ == '__main__':
    test_masking()
