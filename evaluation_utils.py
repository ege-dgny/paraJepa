"""
Evaluation utilities for ParaJEPA style-content disentanglement evaluation.

This module implements the evaluation strategy from the research proposal:
- Style Probe (should have LOW accuracy)
- Content Probe (should have HIGH accuracy)
- Retrieval Metrics (Recall@K)
- t-SNE Visualizations
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import hashlib


class Evaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def _unwrap_model(self, model):
        """Unwrap DDP model if needed"""
        if isinstance(model, nn.parallel.DistributedDataParallel):
            return model.module
        return model

    def _get_batch_embeddings(self, model, input_ids, attention_mask):
        """Helper to get embeddings from different model types"""
        # 1. Check for 'encode' method (Wrapper / Custom Interface)
        if hasattr(model, 'encode'):
            return model.encode(input_ids, attention_mask)
        # 2. Check for 'context_encoder' (ParaJEPA)
        elif hasattr(model, 'context_encoder'):
            return model.context_encoder(input_ids, attention_mask)
        # 3. Fallback: Assume model(input_ids, attention_mask) returns embeddings
        # This handles simple cases, but users should prefer wrappers
        else:
            return model(input_ids, attention_mask)

    @torch.no_grad()
    def get_embeddings(self, model, dataloader, extract_from='content', return_labels=False):
        """
        Extracts embeddings from the model for the entire dataloader.
        
        Args:
            model: ParaJEPA model (can be DDP-wrapped)
            dataloader: DataLoader with batches
            extract_from: 'content' (original text) or 'style' (paraphrase)
            return_labels: If True, also extract/generate labels
        
        Returns:
            embeddings: numpy array of shape (N, hidden_dim)
            labels_style: numpy array of style labels (if return_labels=True)
            labels_content: numpy array of content labels (if return_labels=True)
        """
        model = self._unwrap_model(model)
        model.eval()
        embeddings = []
        labels_style = []
        labels_content = []

        print(f"Extracting embeddings from {extract_from} inputs...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            # Extract from content inputs (original text) or style inputs (paraphrase)
            if extract_from == 'content':
                input_ids = batch['content_input_ids'].to(self.device)
                attention_mask = batch['content_attention_mask'].to(self.device)
            else:  # style
                input_ids = batch['style_input_ids'].to(self.device)
                attention_mask = batch['style_attention_mask'].to(self.device)
            
            # Use context encoder to get embeddings
            emb = self._get_batch_embeddings(model, input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
            
            if return_labels:
                # Generate labels from batch using text hashing if possible, or fallback to index
                # Note: Index-based labels are unstable if shuffling is on!
                # Ideally, dataloader should return labels or text.
                # For now, we warn the user if this is used without a dataset.
                batch_size = input_ids.shape[0]
                
                # Check if batch has 'text' or equivalent for hashing
                if 'content_text' in batch: # Assuming we might add this to collate_fn later
                     content_texts = batch['content_text']
                     content_labels = [int(hashlib.md5(t.encode()).hexdigest()[:8], 16) for t in content_texts]
                else:
                    # FALLBACK: Use simple index, but this is BROKEN for shuffling
                    # We will try to rely on get_embeddings_with_labels for probes instead
                    content_labels = [batch_idx * batch_size + i for i in range(batch_size)]
                
                style_labels = [batch_idx * batch_size + i for i in range(batch_size)]
                
                labels_style.extend(style_labels)
                labels_content.extend(content_labels)

        embeddings = np.vstack(embeddings)
        labels_style = np.array(labels_style) if return_labels else np.array([])
        labels_content = np.array(labels_content) if return_labels else np.array([])
        
        return embeddings, labels_style, labels_content

    @torch.no_grad()
    def get_embeddings_with_labels(self, model, dataset, tokenizer, max_length=128, batch_size=32, extract_from='content'):
        """
        Extract embeddings with proper labels by processing the dataset directly.
        This method generates labels based on content text hashing.
        
        Args:
            model: ParaJEPA model
            dataset: HuggingFace dataset (not DataLoader)
            tokenizer: Tokenizer
            max_length: Max sequence length
            batch_size: Batch size for processing
            extract_from: 'content' or 'style'
        
        Returns:
            embeddings, style_labels, content_labels
        """
        model = self._unwrap_model(model)
        model.eval()
        
        embeddings = []
        style_labels = []
        content_labels = []
        
        print(f"Extracting embeddings from {extract_from} with labels...")
        
        # Process dataset to get labels
        content_id_map = {}  # Map content text to unique ID
        style_id_map = {}    # Map paraphrase index to style ID
        
        # Helper function to extract raw HF dataset item
        def get_raw_item(dataset, idx):
            """
            Traverse through Subset/Wrapper layers to get the raw HuggingFace dataset item.
            Handles: Subset(WikiAutoAssetDataset(HF)) -> HF
            """
            current = dataset
            current_idx = idx
            
            # Step 1: Handle Subset - map index and get underlying dataset
            if hasattr(current, 'indices') and hasattr(current, 'dataset'):
                # This is a Subset from torch.utils.data
                current_idx = current.indices[current_idx]
                current = current.dataset
            
            # Step 2: Handle WikiAutoAssetDataset wrapper - get underlying HF dataset
            if hasattr(current, 'dataset') and hasattr(current, '__getitem__'):
                # Test if this is a wrapper by checking what __getitem__ returns
                try:
                    test_item = current[0] if len(current) > 0 else None
                    if test_item is not None and isinstance(test_item, dict):
                        if 'content_input_ids' in test_item or 'style_input_ids' in test_item:
                            # This is the wrapper (returns tensors), go deeper
                            if hasattr(current, 'dataset'):
                                current = current.dataset
                        elif 'source' in test_item or 'text' in test_item:
                            # Already raw HF dataset, use it directly
                            return current[current_idx]
                except:
                    # If test fails, try to access dataset attribute directly
                    if hasattr(current, 'dataset'):
                        current = current.dataset
            
            # Step 3: Try to access the final dataset
            try:
                item = current[current_idx]
                if isinstance(item, dict):
                    if 'source' in item or 'text' in item:
                        # Success! We have the raw HF item
                        return item
            except:
                pass
            
            # Fallback: return None (will use tensor decoding)
            return None
        
        for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
            # Try to get raw HF dataset item first
            item = get_raw_item(dataset, idx)
            
            # If we couldn't get raw item, fall back to accessing dataset[idx] directly
            if item is None:
                item = dataset[idx]
            
            # Now process the item
            if 'source' in item:
                # This is WikiAutoAsset
                content_text = item['source'] # Complex text
                # Find simplifications
                if 'references' in item and len(item['references']) > 0:
                    paraphrases = item['references']
                elif 'target' in item:
                    paraphrases = [item['target']]
                else:
                    paraphrases = []
            elif 'text' in item:
                # This is the ChatGPT Paraphrase Dataset (Original)
                content_text = item['text']
                paraphrases = item.get('paraphrases', [])
            else:
                # Unknown format, try to guess or fail gracefully
                # If we are here, it means we got the TENSOR dict from dataload.py
                # We can't hash tensors to get content labels easily/consistently across epochs if they are transformed
                # BUT, since we are stuck, let's try to decode the input_ids back to text?
                # That requires tokenizer.
                
                if 'content_input_ids' in item:
                    content_text = tokenizer.decode(item['content_input_ids'], skip_special_tokens=True)
                    # For style, decode style_input_ids
                    style_text = tokenizer.decode(item['style_input_ids'], skip_special_tokens=True)
                    paraphrases = [style_text]
                else:
                    print(f"Warning: Unknown dataset format at index {idx}. Keys: {item.keys()}")
                    continue
            # -------------------------------------------
            
            # Generate content label (hash of original text)
            content_hash = int(hashlib.md5(content_text.encode()).hexdigest()[:8], 16)
            if content_hash not in content_id_map:
                content_id_map[content_hash] = len(content_id_map)
            content_label = content_id_map[content_hash]
            
            # Generate style label (paraphrase index)
            if len(paraphrases) > 0:
                # Use first paraphrase as style (or random for variety)
                style_text = paraphrases[0]
                style_hash = int(hashlib.md5(style_text.encode()).hexdigest()[:8], 16)
                if style_hash not in style_id_map:
                    style_id_map[style_hash] = len(style_id_map)
                style_label = style_id_map[style_hash]
            else:
                style_label = 0
            
            # Tokenize and get embedding
            if extract_from == 'content':
                text = content_text
            else:
                text = paraphrases[0] if len(paraphrases) > 0 else content_text
            
            enc = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
            input_ids = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)
            
            emb = self._get_batch_embeddings(model, input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
            style_labels.append(style_label)
            content_labels.append(content_label)
        
        return np.vstack(embeddings), np.array(style_labels), np.array(content_labels)

    def linear_probe(self, train_embeddings, train_labels, test_embeddings, test_labels, task_name="Task", max_samples=10000):
        """
        Trains a simple logistic regression probe.
        
        Args:
            train_embeddings: Training embeddings (N, D)
            train_labels: Training labels (N,)
            test_embeddings: Test embeddings (M, D)
            test_labels: Test labels (M,)
            task_name: Name of the task for logging
            max_samples: Maximum number of samples to use (to avoid memory issues)
        """
        print(f"--- Running Probe: {task_name} ---")
        
        # Subsample if too many samples
        if len(train_embeddings) > max_samples:
            print(f"Subsampling from {len(train_embeddings)} to {max_samples} samples...")
            indices = np.random.choice(len(train_embeddings), max_samples, replace=False)
            train_embeddings = train_embeddings[indices]
            train_labels = train_labels[indices]
        
        if len(test_embeddings) > max_samples:
            indices = np.random.choice(len(test_embeddings), max_samples, replace=False)
            test_embeddings = test_embeddings[indices]
            test_labels = test_labels[indices]
        
        # Check unique labels
        unique_labels = len(np.unique(train_labels))
        if unique_labels < 2:
            print(f"Warning: Only {unique_labels} unique labels found. Skipping probe.")
            return 0.0
        
        # If too many classes, use a different approach
        if unique_labels > 1000:
            print(f"Warning: {unique_labels} unique labels found. Using multiclass with limited classes...")
            # Keep only most common classes
            from collections import Counter
            label_counts = Counter(train_labels)
            top_classes = [label for label, _ in label_counts.most_common(100)]
            mask = np.isin(train_labels, top_classes)
            train_embeddings = train_embeddings[mask]
            train_labels = train_labels[mask]
            mask = np.isin(test_labels, top_classes)
            test_embeddings = test_embeddings[mask]
            test_labels = test_labels[mask]
        
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs', n_jobs=1)
        clf.fit(train_embeddings, train_labels)
        
        preds = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, preds)
        
        print(f"{task_name} Accuracy: {acc:.4f}")
        if "Style" in task_name:
            print(f"  → Lower is better (target: ~random chance)")
        elif "Content" in task_name:
            print(f"  → Higher is better (target: >0.7)")
        
        return acc

    def compute_retrieval_metrics(self, query_embeddings, key_embeddings, k_values=[1, 5, 10]):
        """
        Computes Recall@K.
        
        Args:
            query_embeddings: (N, D) - Embeddings of the 'Style' views (paraphrases)
            key_embeddings: (N, D) - Embeddings of the 'Content' views (original texts)
            k_values: List of k values for Recall@K
        
        Returns:
            Dictionary with Recall@K values
        """
        print("--- Computing Retrieval Metrics ---")
        
        # Normalize for Cosine Similarity
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        k_norm = key_embeddings / (np.linalg.norm(key_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Similarity Matrix (N x N)
        # sim[i, j] = similarity between query i and key j
        sim_matrix = np.matmul(q_norm, k_norm.T)
        
        num_samples = sim_matrix.shape[0]
        results = {}
        
        for k in k_values:
            correct = 0
            for i in range(num_samples):
                # Get indices of top k matches for query i
                # argsort is ascending, so take last k and reverse
                top_k_indices = np.argsort(sim_matrix[i])[-k:][::-1]
                
                if i in top_k_indices:
                    correct += 1
            
            recall = correct / num_samples
            results[f"R@{k}"] = recall
            print(f"Recall@{k}: {recall:.4f}")
        
        return results

    def plot_tsne(self, embeddings, labels, title="t-SNE Plot", save_path="tsne.png", max_points=2000):
        """
        Generates a t-SNE plot colored by labels.
        
        Args:
            embeddings: Embeddings array (N, D)
            labels: Labels array (N,)
            title: Plot title
            save_path: Path to save the plot
            max_points: Maximum number of points to plot (for speed)
        """
        print(f"Generating t-SNE plot: {title}...")
        
        # Limit points for speed
        n_points = min(len(embeddings), max_points)
        indices = np.random.choice(len(embeddings), n_points, replace=False)
        emb_subset = embeddings[indices]
        labels_subset = labels[indices]
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_points - 1))
        emb_2d = tsne.fit_transform(emb_subset)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels_subset, palette="tab10", legend="full", alpha=0.6)
        plt.title(title)
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")


# Example usage function
def run_full_evaluation(model, train_loader, test_loader, train_dataset=None, test_dataset=None, 
                       tokenizer=None, device='cuda', max_length=128):
    """
    Run the complete evaluation suite as described in the research proposal.
    
    Args:
        model: Trained ParaJEPA model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        train_dataset: Optional - HuggingFace dataset for label extraction
        test_dataset: Optional - HuggingFace dataset for label extraction
        tokenizer: Optional - Tokenizer (needed if using datasets)
        device: Device to run on
        max_length: Max sequence length
    """
    evaluator = Evaluator(device=device)
    
    print("=" * 60)
    print("Running Full Evaluation Suite")
    print("=" * 60)
    
    # Extract embeddings
    if train_dataset is not None and test_dataset is not None and tokenizer is not None:
        # Use dataset-based extraction for proper labels
        print("\n1. Extracting embeddings with labels from datasets...")
        train_emb, train_style, train_content = evaluator.get_embeddings_with_labels(
            model, train_dataset, tokenizer, max_length=max_length, extract_from='content'
        )
        test_emb, test_style, test_content = evaluator.get_embeddings_with_labels(
            model, test_dataset, tokenizer, max_length=max_length, extract_from='content'
        )
    else:
        # Fallback to DataLoader, BUT we strongly warn this might be broken for probes
        print("\nWARNING: Using DataLoader extraction without explicit text access.")
        print("   If train_loader has shuffle=True, probe accuracy will be garbage.")
        print("\n1. Extracting embeddings from data loaders...")
        train_emb, train_style, train_content = evaluator.get_embeddings(
            model, train_loader, extract_from='content', return_labels=True
        )
        test_emb, test_style, test_content = evaluator.get_embeddings(
            model, test_loader, extract_from='content', return_labels=True
        )
    
    # Extract style embeddings for retrieval
    print("\n2. Extracting style embeddings for retrieval...")
    train_style_emb, _, _ = evaluator.get_embeddings(
        model, train_loader, extract_from='style'
    )
    test_style_emb, _, _ = evaluator.get_embeddings(
        model, test_loader, extract_from='style'
    )
    
    # Run probes
    print("\n3. Running Linear Probes...")
    style_acc = evaluator.linear_probe(
        train_emb, train_style, test_emb, test_style, 
        task_name="Style Probe (Should be LOW)"
    )
    content_acc = evaluator.linear_probe(
        train_emb, train_content, test_emb, test_content,
        task_name="Content Probe (Should be HIGH)"
    )
    
    # Compute retrieval metrics
    print("\n4. Computing Retrieval Metrics...")
    retrieval_results = evaluator.compute_retrieval_metrics(
        test_style_emb, test_emb, k_values=[1, 5, 10]
    )
    
    # Generate visualizations
    print("\n5. Generating t-SNE Visualizations...")
    evaluator.plot_tsne(
        test_emb, test_content, 
        title="Embeddings by Content (Should Cluster)", 
        save_path="tsne_content.png"
    )
    evaluator.plot_tsne(
        test_emb, test_style,
        title="Embeddings by Style (Should Mix)", 
        save_path="tsne_style.png"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Style Probe Accuracy: {style_acc:.4f} (Lower is better)")
    print(f"Content Probe Accuracy: {content_acc:.4f} (Higher is better)")
    print(f"Retrieval Recall@1: {retrieval_results['R@1']:.4f}")
    print(f"Retrieval Recall@5: {retrieval_results['R@5']:.4f}")
    print(f"Retrieval Recall@10: {retrieval_results['R@10']:.4f}")
    print("=" * 60)
    
    return {
        'style_accuracy': style_acc,
        'content_accuracy': content_acc,
        'retrieval': retrieval_results
    }

