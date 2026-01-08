"""
Script 6b: Compare PCA vs Autoencoder Alignment
================================================

This script aligns text embeddings with BOTH PCA and Autoencoder embeddings,
then compares which method works better.

This addresses the question: "Why only PCA, not Autoencoder?"

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Import classes and functions from script 06
# We'll define them here to avoid import issues
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """CLIP-style contrastive loss."""
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)


class CLIPLoss(nn.Module):
    """CLIP Loss as implemented in CellWhisperer."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_embeds, transcriptome_embeds):
        text_embeds = F.normalize(text_embeds, dim=-1)
        transcriptome_embeds = F.normalize(transcriptome_embeds, dim=-1)
        logits = torch.matmul(text_embeds, transcriptome_embeds.T) / self.temperature
        text_loss = contrastive_loss(logits)
        transcriptome_loss = contrastive_loss(logits.T)
        return (text_loss + transcriptome_loss) / 2.0


class ProjectionHead(nn.Module):
    """Projects embeddings to shared space."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        
    def forward(self, x):
        return self.projection(x)


class DualEncoderModel(nn.Module):
    """Simplified version of CellWhisperer's TranscriptomeTextDualEncoderModel."""
    def __init__(self, text_dim, transcriptome_dim, projection_dim):
        super().__init__()
        self.text_projection = ProjectionHead(text_dim, projection_dim)
        self.transcriptome_projection = ProjectionHead(transcriptome_dim, projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, text_embeds, transcriptome_embeds):
        text_projected = self.text_projection(text_embeds)
        transcriptome_projected = self.transcriptome_projection(transcriptome_embeds)
        return text_projected, transcriptome_projected

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

PROJECTION_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 200
TEMPERATURE = 0.07

# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def train_alignment(text_emb, expr_emb, metadata, epochs=200):
    """Train the projection layers to align text and expression embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_tensor = torch.FloatTensor(text_emb).to(device)
    expr_tensor = torch.FloatTensor(expr_emb).to(device)
    
    model = DualEncoderModel(
        text_dim=text_emb.shape[1],
        transcriptome_dim=expr_emb.shape[1],
        projection_dim=PROJECTION_DIM
    ).to(device)
    
    criterion = CLIPLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    losses = []
    for epoch in tqdm(range(epochs), desc="Training", leave=False):
        model.train()
        optimizer.zero_grad()
        
        text_proj, expr_proj = model(text_tensor, expr_tensor)
        loss = criterion(text_proj, expr_proj)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return model, losses


def get_aligned_embeddings(model, text_emb, expr_emb):
    """Extract the aligned embeddings from the trained model."""
    device = next(model.parameters()).device
    
    text_tensor = torch.FloatTensor(text_emb).to(device)
    expr_tensor = torch.FloatTensor(expr_emb).to(device)
    
    model.eval()
    with torch.no_grad():
        text_aligned, expr_aligned = model(text_tensor, expr_tensor)
        text_aligned = F.normalize(text_aligned, dim=-1)
        expr_aligned = F.normalize(expr_aligned, dim=-1)
    
    return text_aligned.cpu().numpy(), expr_aligned.cpu().numpy()


def load_all_embeddings():
    """Load text, PCA, and Autoencoder embeddings."""
    print("=" * 60)
    print("Loading All Embeddings")
    print("=" * 60)
    
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    pca_emb = np.load(DATA_DIR / "expression_embeddings_pca.npy")
    autoencoder_emb = np.load(DATA_DIR / "expression_embeddings_autoencoder.npy")
    
    print(f"✓ Text embeddings: {text_emb.shape}")
    print(f"✓ PCA embeddings: {pca_emb.shape}")
    print(f"✓ Autoencoder embeddings: {autoencoder_emb.shape}")
    print(f"✓ Samples: {len(metadata)}")
    
    return metadata, text_emb, pca_emb, autoencoder_emb


def calculate_alignment_metrics(text_aligned, expr_aligned, metadata):
    """Calculate metrics to compare alignment quality."""
    metrics = {}
    
    # 1. Diagonal similarity (should be high)
    similarity_matrix = np.dot(text_aligned, expr_aligned.T)
    diagonal_similarities = np.diag(similarity_matrix)
    metrics['mean_diagonal_similarity'] = np.mean(diagonal_similarities)
    metrics['min_diagonal_similarity'] = np.min(diagonal_similarities)
    metrics['max_diagonal_similarity'] = np.max(diagonal_similarities)
    
    # 2. Off-diagonal similarity (should be low)
    off_diagonal = similarity_matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    metrics['mean_off_diagonal_similarity'] = np.mean(off_diagonal)
    metrics['max_off_diagonal_similarity'] = np.max(off_diagonal)
    
    # 3. Alignment accuracy (exact matches)
    correct = 0
    for i in range(len(metadata)):
        best_match = np.argmax(similarity_matrix[i])
        if i == best_match:
            correct += 1
    metrics['alignment_accuracy'] = correct / len(metadata) * 100
    
    # 4. Silhouette score (clustering quality by disease)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    disease_labels = le.fit_transform(metadata['disease_abbrev'].values)
    
    # Combined embeddings for clustering
    combined = np.vstack([text_aligned, expr_aligned])
    combined_labels = np.hstack([disease_labels, disease_labels])
    
    metrics['silhouette_score'] = silhouette_score(combined, combined_labels)
    
    return metrics


def compare_alignments(metadata, text_emb, pca_emb, autoencoder_emb):
    """Train and compare both alignment methods."""
    print("\n" + "=" * 60)
    print("COMPARING PCA vs AUTOENCODER ALIGNMENT")
    print("=" * 60)
    
    results = {}
    
    # 1. Align with PCA
    print("\n" + "-" * 60)
    print("TRAINING: Text ↔ PCA Alignment")
    print("-" * 60)
    model_pca, losses_pca = train_alignment(text_emb, pca_emb, metadata, epochs=EPOCHS)
    text_pca, expr_pca = get_aligned_embeddings(model_pca, text_emb, pca_emb)
    metrics_pca = calculate_alignment_metrics(text_pca, expr_pca, metadata)
    results['PCA'] = {
        'model': model_pca,
        'losses': losses_pca,
        'text_aligned': text_pca,
        'expr_aligned': expr_pca,
        'metrics': metrics_pca
    }
    
    # 2. Align with Autoencoder
    print("\n" + "-" * 60)
    print("TRAINING: Text ↔ Autoencoder Alignment")
    print("-" * 60)
    model_ae, losses_ae = train_alignment(text_emb, autoencoder_emb, metadata, epochs=EPOCHS)
    text_ae, expr_ae = get_aligned_embeddings(model_ae, text_emb, autoencoder_emb)
    metrics_ae = calculate_alignment_metrics(text_ae, expr_ae, metadata)
    results['Autoencoder'] = {
        'model': model_ae,
        'losses': losses_ae,
        'text_aligned': text_ae,
        'expr_aligned': expr_ae,
        'metrics': metrics_ae
    }
    
    return results


def visualize_comparison(results, metadata):
    """Create comprehensive comparison visualization."""
    print("\n" + "=" * 60)
    print("Creating Comparison Visualization")
    print("=" * 60)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Color map
    colors = {
        'OA': '#1f77b4', 'RA': '#d62728', 'SLE': '#9467bd',
        'MIC': '#2ca02c', 'SA': '#ff7f0e'
    }
    labels = metadata['disease_abbrev'].values
    
    # ========================================================================
    # Plot 1: Training Loss Comparison
    # ========================================================================
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(results['PCA']['losses'], label='PCA', alpha=0.8)
    ax1.plot(results['Autoencoder']['losses'], label='Autoencoder', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CLIP Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Metrics Comparison (Bar Chart)
    # ========================================================================
    ax2 = plt.subplot(3, 3, 2)
    metrics_to_compare = [
        'mean_diagonal_similarity',
        'alignment_accuracy',
        'silhouette_score'
    ]
    pca_values = [results['PCA']['metrics'][m] for m in metrics_to_compare]
    ae_values = [results['Autoencoder']['metrics'][m] for m in metrics_to_compare]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    ax2.bar(x - width/2, pca_values, width, label='PCA', alpha=0.8)
    ax2.bar(x + width/2, ae_values, width, label='Autoencoder', alpha=0.8)
    ax2.set_ylabel('Score')
    ax2.set_title('Alignment Quality Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Diagonal\nSimilarity', 'Accuracy\n(%)', 'Silhouette\nScore'], rotation=0, ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 3: Similarity Matrix - PCA
    # ========================================================================
    ax3 = plt.subplot(3, 3, 3)
    similarity_pca = np.dot(results['PCA']['text_aligned'], results['PCA']['expr_aligned'].T)
    im3 = ax3.imshow(similarity_pca, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax3.set_title('PCA Alignment\nSimilarity Matrix')
    ax3.set_xlabel('Expression samples')
    ax3.set_ylabel('Text samples')
    plt.colorbar(im3, ax=ax3)
    
    # ========================================================================
    # Plot 4: Similarity Matrix - Autoencoder
    # ========================================================================
    ax4 = plt.subplot(3, 3, 4)
    similarity_ae = np.dot(results['Autoencoder']['text_aligned'], results['Autoencoder']['expr_aligned'].T)
    im4 = ax4.imshow(similarity_ae, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax4.set_title('Autoencoder Alignment\nSimilarity Matrix')
    ax4.set_xlabel('Expression samples')
    ax4.set_ylabel('Text samples')
    plt.colorbar(im4, ax=ax4)
    
    # ========================================================================
    # Plot 5: PCA Aligned Embeddings (2D)
    # ========================================================================
    ax5 = plt.subplot(3, 3, 5)
    combined_pca = np.vstack([results['PCA']['text_aligned'], results['PCA']['expr_aligned']])
    tsne_pca = TSNE(n_components=2, perplexity=5, random_state=42)
    combined_2d_pca = tsne_pca.fit_transform(combined_pca)
    n_samples = len(results['PCA']['text_aligned'])
    text_2d_pca = combined_2d_pca[:n_samples]
    expr_2d_pca = combined_2d_pca[n_samples:]
    
    for disease in colors.keys():
        mask = labels == disease
        if mask.any():
            ax5.scatter(text_2d_pca[mask, 0], text_2d_pca[mask, 1],
                       c=colors[disease], marker='o', s=80, alpha=0.7, label=f'{disease} (text)')
            ax5.scatter(expr_2d_pca[mask, 0], expr_2d_pca[mask, 1],
                       c=colors[disease], marker='^', s=80, alpha=0.7, label=f'{disease} (expr)')
    
    for i in range(n_samples):
        ax5.plot([text_2d_pca[i, 0], expr_2d_pca[i, 0]],
                [text_2d_pca[i, 1], expr_2d_pca[i, 1]],
                'k-', alpha=0.1, linewidth=0.5)
    
    ax5.set_title('PCA Alignment\n(2D visualization)')
    ax5.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 6: Autoencoder Aligned Embeddings (2D)
    # ========================================================================
    ax6 = plt.subplot(3, 3, 6)
    combined_ae = np.vstack([results['Autoencoder']['text_aligned'], results['Autoencoder']['expr_aligned']])
    tsne_ae = TSNE(n_components=2, perplexity=5, random_state=42)
    combined_2d_ae = tsne_ae.fit_transform(combined_ae)
    text_2d_ae = combined_2d_ae[:n_samples]
    expr_2d_ae = combined_2d_ae[n_samples:]
    
    for disease in colors.keys():
        mask = labels == disease
        if mask.any():
            ax6.scatter(text_2d_ae[mask, 0], text_2d_ae[mask, 1],
                       c=colors[disease], marker='o', s=80, alpha=0.7)
            ax6.scatter(expr_2d_ae[mask, 0], expr_2d_ae[mask, 1],
                       c=colors[disease], marker='^', s=80, alpha=0.7)
    
    for i in range(n_samples):
        ax6.plot([text_2d_ae[i, 0], expr_2d_ae[i, 0]],
                [expr_2d_ae[i, 1], expr_2d_ae[i, 1]],
                'k-', alpha=0.1, linewidth=0.5)
    
    ax6.set_title('Autoencoder Alignment\n(2D visualization)')
    ax6.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 7: Detailed Metrics Table
    # ========================================================================
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    metrics_table = []
    for metric_name in ['mean_diagonal_similarity', 'min_diagonal_similarity', 
                       'max_diagonal_similarity', 'mean_off_diagonal_similarity',
                       'alignment_accuracy', 'silhouette_score']:
        pca_val = results['PCA']['metrics'][metric_name]
        ae_val = results['Autoencoder']['metrics'][metric_name]
        
        # Format values
        if 'accuracy' in metric_name:
            pca_str = f"{pca_val:.1f}%"
            ae_str = f"{ae_val:.1f}%"
        elif 'similarity' in metric_name or 'silhouette' in metric_name:
            pca_str = f"{pca_val:.4f}"
            ae_str = f"{ae_val:.4f}"
        else:
            pca_str = f"{pca_val:.4f}"
            ae_str = f"{ae_val:.4f}"
        
        # Determine winner
        if 'diagonal' in metric_name or 'accuracy' in metric_name or 'silhouette' in metric_name:
            winner = "PCA" if pca_val > ae_val else "Autoencoder"
        else:  # off_diagonal should be lower
            winner = "PCA" if pca_val < ae_val else "Autoencoder"
        
        metrics_table.append([
            metric_name.replace('_', ' ').title(),
            pca_str,
            ae_str,
            winner
        ])
    
    table = ax7.table(cellText=metrics_table,
                     colLabels=['Metric', 'PCA', 'Autoencoder', 'Winner'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax7.set_title('Detailed Metrics Comparison', pad=20)
    
    # ========================================================================
    # Plot 8: Diagonal Similarity Distribution
    # ========================================================================
    ax8 = plt.subplot(3, 3, 8)
    diag_pca = np.diag(similarity_pca)
    diag_ae = np.diag(similarity_ae)
    ax8.hist(diag_pca, bins=15, alpha=0.6, label='PCA', color='blue')
    ax8.hist(diag_ae, bins=15, alpha=0.6, label='Autoencoder', color='orange')
    ax8.set_xlabel('Diagonal Similarity')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Diagonal Similarity Distribution')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 9: Summary Text
    # ========================================================================
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    pca_metrics = results['PCA']['metrics']
    ae_metrics = results['Autoencoder']['metrics']
    
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*50}
    
    PCA Alignment:
    • Final Loss: {results['PCA']['losses'][-1]:.4f}
    • Accuracy: {pca_metrics['alignment_accuracy']:.1f}%
    • Mean Diagonal Similarity: {pca_metrics['mean_diagonal_similarity']:.4f}
    • Silhouette Score: {pca_metrics['silhouette_score']:.4f}
    
    Autoencoder Alignment:
    • Final Loss: {results['Autoencoder']['losses'][-1]:.4f}
    • Accuracy: {ae_metrics['alignment_accuracy']:.1f}%
    • Mean Diagonal Similarity: {ae_metrics['mean_diagonal_similarity']:.4f}
    • Silhouette Score: {ae_metrics['silhouette_score']:.4f}
    
    RECOMMENDATION:
    {'PCA' if pca_metrics['alignment_accuracy'] > ae_metrics['alignment_accuracy'] else 'Autoencoder'} 
    performs better for this dataset!
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax9.transAxes)
    
    plt.tight_layout()
    
    # Save
    save_path = RESULTS_DIR / "pca_vs_autoencoder_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {save_path}")
    
    return fig


def print_comparison_summary(results):
    """Print detailed comparison summary."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    pca_metrics = results['PCA']['metrics']
    ae_metrics = results['Autoencoder']['metrics']
    
    print("\n📊 PCA Alignment Results:")
    print(f"  • Final Training Loss: {results['PCA']['losses'][-1]:.4f}")
    print(f"  • Alignment Accuracy: {pca_metrics['alignment_accuracy']:.1f}%")
    print(f"  • Mean Diagonal Similarity: {pca_metrics['mean_diagonal_similarity']:.4f}")
    print(f"  • Silhouette Score: {pca_metrics['silhouette_score']:.4f}")
    
    print("\n📊 Autoencoder Alignment Results:")
    print(f"  • Final Training Loss: {results['Autoencoder']['losses'][-1]:.4f}")
    print(f"  • Alignment Accuracy: {ae_metrics['alignment_accuracy']:.1f}%")
    print(f"  • Mean Diagonal Similarity: {ae_metrics['mean_diagonal_similarity']:.4f}")
    print(f"  • Silhouette Score: {ae_metrics['silhouette_score']:.4f}")
    
    print("\n🏆 Winner:")
    if pca_metrics['alignment_accuracy'] > ae_metrics['alignment_accuracy']:
        print("  → PCA performs better for this dataset!")
        print(f"  → Difference: {pca_metrics['alignment_accuracy'] - ae_metrics['alignment_accuracy']:.1f}% higher accuracy")
    elif ae_metrics['alignment_accuracy'] > pca_metrics['alignment_accuracy']:
        print("  → Autoencoder performs better for this dataset!")
        print(f"  → Difference: {ae_metrics['alignment_accuracy'] - pca_metrics['alignment_accuracy']:.1f}% higher accuracy")
    else:
        print("  → Both perform equally well!")
    
    print("\n💡 Interpretation:")
    print("  • Higher accuracy = Better alignment")
    print("  • Higher diagonal similarity = Better matching")
    print("  • Higher silhouette score = Better disease clustering")


def save_aligned_embeddings(results):
    """Save both sets of aligned embeddings."""
    print("\n" + "=" * 60)
    print("Saving Aligned Embeddings")
    print("=" * 60)
    
    # Save PCA aligned
    np.save(DATA_DIR / "text_embeddings_aligned_pca.npy", results['PCA']['text_aligned'])
    np.save(DATA_DIR / "expression_embeddings_aligned_pca.npy", results['PCA']['expr_aligned'])
    print("✓ Saved PCA-aligned embeddings")
    
    # Save Autoencoder aligned
    np.save(DATA_DIR / "text_embeddings_aligned_autoencoder.npy", results['Autoencoder']['text_aligned'])
    np.save(DATA_DIR / "expression_embeddings_aligned_autoencoder.npy", results['Autoencoder']['expr_aligned'])
    print("✓ Saved Autoencoder-aligned embeddings")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("PCA vs AUTOENCODER ALIGNMENT COMPARISON")
    print("=" * 60)
    
    # Load embeddings
    try:
        metadata, text_emb, pca_emb, autoencoder_emb = load_all_embeddings()
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("Please run scripts 01, 02, and 03 first!")
        return
    
    # Compare alignments
    results = compare_alignments(metadata, text_emb, pca_emb, autoencoder_emb)
    
    # Visualize comparison
    visualize_comparison(results, metadata)
    
    # Print summary
    print_comparison_summary(results)
    
    # Save embeddings
    save_aligned_embeddings(results)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nYou now have:")
    print("  1. Comparison visualization (pca_vs_autoencoder_comparison.png)")
    print("  2. Both sets of aligned embeddings saved")
    print("  3. Detailed metrics showing which method works better")
    print("\nThis answers: 'Why only PCA, not Autoencoder?'")


if __name__ == "__main__":
    main()

