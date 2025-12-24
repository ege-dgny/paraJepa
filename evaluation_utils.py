"""
Evaluation utilities for ParaJEPA style-content disentanglement evaluation.

Refactored to ensure rigorous evaluation:
- Style Probe: Binary classification (Original vs Paraphrase). Target: 50% acc (random) for invariance.
- Content Metric: Retrieval Recall@K (Content Preservation).
- t-SNE: Visualizes overlap between Original and Paraphrase distributions.
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
        else:
            return model(input_ids, attention_mask)

    @torch.no_grad()
    def get_embeddings(self, model, dataloader, extract_from='content', return_labels=False):
        """
        Extracts embeddings.
        If return_labels=True, assigns:
            Label 0: Content/Original
            Label 1: Style/Paraphrase
        """
        model = self._unwrap_model(model)
        model.eval()
        embeddings = []
        labels = []

        print(f"Extracting embeddings from {extract_from} inputs...")
        
        for batch in tqdm(dataloader, desc=f"Extracting {extract_from}"):
            if extract_from == 'content':
                input_ids = batch['content_input_ids'].to(self.device)
                attention_mask = batch['content_attention_mask'].to(self.device)
                current_labels = np.zeros(input_ids.shape[0]) # Label 0 for Original
            else:  # style
                input_ids = batch['style_input_ids'].to(self.device)
                attention_mask = batch['style_attention_mask'].to(self.device)
                current_labels = np.ones(input_ids.shape[0]) # Label 1 for Paraphrase
            
            emb = self._get_batch_embeddings(model, input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())
            
            if return_labels:
                labels.extend(current_labels)

        embeddings = np.vstack(embeddings)
        labels = np.array(labels) if return_labels else np.array([])
        
        return embeddings, labels

    def linear_probe(self, train_embeddings, train_labels, test_embeddings, test_labels, task_name="Task"):
        """
        Trains a logistic regression probe.
        """
        print(f"--- Running Probe: {task_name} ---")
        
        # Binary classification (0 vs 1)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
        clf.fit(train_embeddings, train_labels)
        
        preds = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, preds)
        
        print(f"{task_name} Accuracy: {acc:.4f}")
        return acc

    def compute_retrieval_metrics(self, query_embeddings, key_embeddings, k_values=[1, 5, 10]):
        """
        Computes Recall@K between Style embeddings (Query) and Content embeddings (Key).
        Matches i-th query to i-th key.
        """
        print("--- Computing Retrieval Metrics ---")
        
        # Normalize for Cosine Similarity
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        k_norm = key_embeddings / (np.linalg.norm(key_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Similarity Matrix (N x N)
        sim_matrix = np.matmul(q_norm, k_norm.T)
        
        num_samples = sim_matrix.shape[0]
        results = {}
        
        for k in k_values:
            correct = 0
            for i in range(num_samples):
                # Get indices of top k matches for query i
                top_k_indices = np.argsort(sim_matrix[i])[-k:][::-1]
                
                if i in top_k_indices:
                    correct += 1
            
            recall = correct / num_samples
            results[f"R@{k}"] = recall
            print(f"Recall@{k}: {recall:.4f}")
        
        return results

    def plot_tsne(self, content_embeddings, style_embeddings, title="t-SNE Plot", save_path="tsne.png", max_points=1000):
        """
        Generates a t-SNE plot showing Original (Content) vs Paraphrase (Style) distributions.
        Ideally, they should overlap perfectly (invariance).
        """
        print(f"Generating t-SNE plot: {title}...")
        
        # Subsample equal amounts
        n_points = min(len(content_embeddings), max_points)
        indices = np.random.choice(len(content_embeddings), n_points, replace=False)
        
        cont_sub = content_embeddings[indices]
        style_sub = style_embeddings[indices]
        
        # Combine
        combined_emb = np.vstack([cont_sub, style_sub])
        combined_labels = np.array([0] * n_points + [1] * n_points)
        label_names = np.array(['Original'] * n_points + ['Paraphrase'] * n_points)
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(combined_emb)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=emb_2d[:, 0], y=emb_2d[:, 1], 
            hue=label_names, 
            style=label_names,
            palette={'Original': 'blue', 'Paraphrase': 'orange'}, 
            alpha=0.6
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved to {save_path}")


def run_full_evaluation(model, train_loader, test_loader, device='cuda', max_length=128):
    """
    Run the complete evaluation suite.
    """
    evaluator = Evaluator(device=device)
    
    print("=" * 60)
    print("Running Full Evaluation Suite (Rigorous Mode)")
    print("=" * 60)
    
    # 1. Extract Embeddings for Probe Training (From Train Loader)
    print("\n1. Preparing Training Data for Probes...")
    train_cont_emb, train_labels_0 = evaluator.get_embeddings(model, train_loader, 'content', True)
    train_style_emb, train_labels_1 = evaluator.get_embeddings(model, train_loader, 'style', True)
    
    # Stack for Binary Classification (0 vs 1)
    X_train = np.vstack([train_cont_emb, train_style_emb])
    y_train = np.concatenate([train_labels_0, train_labels_1])
    
    # 2. Extract Embeddings for Testing (From Test Loader)
    print("\n2. Preparing Test Data for Probes & Retrieval...")
    test_cont_emb, test_labels_0 = evaluator.get_embeddings(model, test_loader, 'content', True)
    test_style_emb, test_labels_1 = evaluator.get_embeddings(model, test_loader, 'style', True)
    
    X_test = np.vstack([test_cont_emb, test_style_emb])
    y_test = np.concatenate([test_labels_0, test_labels_1])
    
    # 3. Run Style Discrimination Probe
    # Task: distinguish Original from Paraphrase
    print("\n3. Running Style Discrimination Probe...")
    # Ideally: ~0.50 (Random) -> Model is perfectly invariant
    # Reality: High accuracy means model retains style/length cues
    style_acc = evaluator.linear_probe(
        X_train, y_train, X_test, y_test, 
        task_name="Style Discrimination (Original vs Paraphrase)"
    )
    
    # 4. Compute Retrieval Metrics (Content Preservation)
    print("\n4. Computing Retrieval Metrics...")
    retrieval_results = evaluator.compute_retrieval_metrics(
        test_style_emb, test_cont_emb, k_values=[1, 5, 10]
    )
    
    # 5. Generate Visualizations
    print("\n5. Generating t-SNE Visualizations...")
    evaluator.plot_tsne(
        test_cont_emb, test_style_emb, 
        title="Distribution Overlap (Original vs Paraphrase)", 
        save_path="tsne_overlap.png"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Style Discrim Accuracy: {style_acc:.4f} (Lower is better for invariance)")
    print(f"Retrieval Recall@1:     {retrieval_results['R@1']:.4f} (Content Preservation)")
    print("=" * 60)
    
    return {
        'style_accuracy': style_acc,
        'retrieval': retrieval_results
    }