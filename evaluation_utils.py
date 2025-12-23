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
            emb = model.context_encoder(input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
            
            if return_labels:
                # Generate labels from batch indices and paraphrase info
                # Style label: use batch index as proxy (each sample has different paraphrase)
                batch_size = input_ids.shape[0]
                style_labels = [batch_idx * batch_size + i for i in range(batch_size)]
                labels_style.extend(style_labels)
                
                # Content label: use a hash of content text to group same content
                # Since we don't have access to original text here, use batch index
                # In practice, you'd want to modify dataloader to return content IDs
                content_labels = [batch_idx * batch_size + i for i in range(batch_size)]
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
        
        for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
            item = dataset[idx]
            content_text = item['text']
            paraphrases = item['paraphrases']
            
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
            
            emb = model.context_encoder(input_ids, attention_mask)
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
        # Use DataLoader-based extraction (labels will be approximate)
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

