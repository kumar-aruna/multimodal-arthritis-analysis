"""
Script 4: Visualize and Compare Embeddings
==========================================

This script visualizes the embeddings created by:
1. BioBERT (text embeddings)
2. PCA (expression embeddings)
3. Autoencoder (expression embeddings)

Visualization methods:
- t-SNE: Non-linear dimensionality reduction
- UMAP: Better at preserving global structure
- PCA: Linear, shows main variance directions

This helps understand:
- How well samples cluster by disease type
- Whether text and expression embeddings capture similar information
- The quality of our embedding approaches

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP (optional)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not installed. Using t-SNE only.")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Color palette for diseases
DISEASE_COLORS = {
    'OA': '#1f77b4',   # Blue - Osteoarthritis
    'RA': '#d62728',   # Red - Rheumatoid Arthritis
    'SLE': '#9467bd',  # Purple - Lupus
    'MIC': '#2ca02c',  # Green - Microcrystalline
    'SA': '#ff7f0e'    # Orange - Seronegative
}

# ============================================================================
# LOAD DATA
# ============================================================================

def load_all_data():
    """
    Load all embeddings and metadata.
    """
    print("=" * 60)
    print("Loading All Data")
    print("=" * 60)
    
    # Load metadata
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    print(f"✓ Loaded metadata: {len(metadata)} samples")
    
    # Load embeddings
    embeddings = {}
    
    # Text embeddings (BioBERT)
    text_path = DATA_DIR / "text_embeddings.npy"
    if text_path.exists():
        embeddings['BioBERT (Text)'] = np.load(text_path)
        print(f"✓ Loaded text embeddings: {embeddings['BioBERT (Text)'].shape}")
    else:
        print("⚠ Text embeddings not found - run script 02 first")
    
    # PCA embeddings
    pca_path = DATA_DIR / "expression_embeddings_pca.npy"
    if pca_path.exists():
        embeddings['PCA (Expression)'] = np.load(pca_path)
        print(f"✓ Loaded PCA embeddings: {embeddings['PCA (Expression)'].shape}")
    else:
        print("⚠ PCA embeddings not found - run script 03 first")
    
    # Autoencoder embeddings
    ae_path = DATA_DIR / "expression_embeddings_autoencoder.npy"
    if ae_path.exists():
        embeddings['Autoencoder (Expression)'] = np.load(ae_path)
        print(f"✓ Loaded autoencoder embeddings: {embeddings['Autoencoder (Expression)'].shape}")
    else:
        print("⚠ Autoencoder embeddings not found - run script 03 first")
    
    return metadata, embeddings


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def reduce_dimensions(embeddings, method='tsne', n_components=2, perplexity=5):
    """
    Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: High-dimensional embeddings
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of output dimensions (2 for plotting)
        perplexity: t-SNE parameter (lower for small datasets)
        
    Returns:
        Reduced embeddings (n_samples × n_components)
    """
    if method == 'tsne':
        # Use low perplexity for small datasets
        perplexity = min(perplexity, len(embeddings) - 1)
        reducer = TSNE(
            n_components=n_components, 
            perplexity=perplexity,
            random_state=42,
            init='pca'
        )
        return reducer.fit_transform(embeddings)
    
    elif method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(5, len(embeddings) - 1),
            random_state=42
        )
        return reducer.fit_transform(embeddings)
    
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_single_embedding(ax, coords, labels, title, colors=DISEASE_COLORS):
    """
    Plot a single embedding visualization.
    """
    for disease in colors.keys():
        mask = labels == disease
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=colors[disease],
                label=disease,
                s=100,
                alpha=0.7,
                edgecolors='white',
                linewidth=1
            )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)


def create_comparison_plot(metadata, embeddings, method='tsne'):
    """
    Create a comparison plot of all embeddings.
    """
    print(f"\n" + "=" * 60)
    print(f"Creating Comparison Plot ({method.upper()})")
    print("=" * 60)
    
    n_embeddings = len(embeddings)
    if n_embeddings == 0:
        print("No embeddings to visualize!")
        return None
    
    # Create figure
    fig, axes = plt.subplots(1, n_embeddings, figsize=(6 * n_embeddings, 5))
    if n_embeddings == 1:
        axes = [axes]
    
    labels = metadata['disease_abbrev'].values
    
    for ax, (name, emb) in zip(axes, embeddings.items()):
        print(f"  Processing {name}...")
        coords = reduce_dimensions(emb, method=method)
        plot_single_embedding(ax, coords, labels, f"{name}\n({method.upper()})")
    
    plt.tight_layout()
    
    # Save figure
    save_path = RESULTS_DIR / f"embedding_comparison_{method}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {save_path}")
    
    return fig


def create_detailed_plot(metadata, embeddings):
    """
    Create a detailed visualization with multiple reduction methods.
    """
    print("\n" + "=" * 60)
    print("Creating Detailed Visualization")
    print("=" * 60)
    
    if not embeddings:
        print("No embeddings to visualize!")
        return None
    
    methods = ['pca', 'tsne']
    if HAS_UMAP:
        methods.append('umap')
    
    n_methods = len(methods)
    n_embeddings = len(embeddings)
    
    fig, axes = plt.subplots(n_methods, n_embeddings, 
                             figsize=(5 * n_embeddings, 5 * n_methods))
    
    if n_embeddings == 1:
        axes = axes.reshape(-1, 1)
    
    labels = metadata['disease_abbrev'].values
    
    for i, method in enumerate(methods):
        print(f"\n  {method.upper()} reduction:")
        for j, (name, emb) in enumerate(embeddings.items()):
            print(f"    - {name}")
            coords = reduce_dimensions(emb, method=method)
            plot_single_embedding(
                axes[i, j], 
                coords, 
                labels, 
                f"{name}\n({method.upper()})"
            )
    
    plt.tight_layout()
    
    # Save figure
    save_path = RESULTS_DIR / "embedding_detailed.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved detailed plot to: {save_path}")
    
    return fig


def plot_similarity_heatmap(metadata, embeddings):
    """
    Create similarity heatmaps for each embedding type.
    """
    print("\n" + "=" * 60)
    print("Creating Similarity Heatmaps")
    print("=" * 60)
    
    from scipy.spatial.distance import pdist, squareform
    
    n_embeddings = len(embeddings)
    if n_embeddings == 0:
        return None
    
    fig, axes = plt.subplots(1, n_embeddings, figsize=(6 * n_embeddings, 5))
    if n_embeddings == 1:
        axes = [axes]
    
    sample_names = metadata['sample_title'].values
    
    for ax, (name, emb) in zip(axes, embeddings.items()):
        # Calculate cosine similarity
        distances = pdist(emb, metric='cosine')
        similarity = 1 - squareform(distances)
        
        # For better visualization, adjust color scale based on actual data
        # Exclude diagonal (self-similarity = 1.0) to set scale for comparisons
        mask_diagonal = np.eye(len(similarity), dtype=bool)
        off_diagonal_values = similarity[~mask_diagonal]
        
        # Use actual min/max of off-diagonal values for better contrast
        # This makes relative differences more visible
        vmin_data = off_diagonal_values.min()
        vmax_data = off_diagonal_values.max()
        
        # For expression embeddings, values might be low, so adjust scale
        if 'Expression' in name:
            # Use symmetric scale around median for better contrast
            median_val = np.median(off_diagonal_values)
            range_val = max(abs(vmax_data - median_val), abs(median_val - vmin_data))
            vmin = median_val - range_val * 1.2
            vmax = median_val + range_val * 1.2
            # But ensure we show the full range
            vmin = min(vmin, vmin_data)
            vmax = max(vmax, vmax_data)
        else:
            # For text embeddings, use standard 0-1 scale
            vmin = 0
            vmax = 1
        
        # Create heatmap with adjusted color scale
        sns.heatmap(
            similarity,
            ax=ax,
            cmap='RdYlBu_r',
            xticklabels=sample_names,
            yticklabels=sample_names,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Similarity'}
        )
        ax.set_title(f"{name}\nSimilarity Matrix", fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    plt.tight_layout()
    
    # Save figure
    save_path = RESULTS_DIR / "similarity_heatmaps.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved heatmaps to: {save_path}")
    
    return fig


def calculate_clustering_metrics(metadata, embeddings):
    """
    Calculate clustering quality metrics.
    
    Silhouette score measures how similar samples are to their own
    cluster compared to other clusters. Range: -1 to 1 (higher is better)
    """
    print("\n" + "=" * 60)
    print("Calculating Clustering Metrics")
    print("=" * 60)
    
    from sklearn.metrics import silhouette_score
    
    labels = metadata['disease_abbrev'].values
    
    results = []
    for name, emb in embeddings.items():
        score = silhouette_score(emb, labels)
        results.append({'Embedding': name, 'Silhouette Score': score})
        print(f"  {name}: {score:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Save results
    save_path = RESULTS_DIR / "clustering_metrics.csv"
    results_df.to_csv(save_path, index=False)
    print(f"\n✓ Saved metrics to: {save_path}")
    
    return results_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to create all visualizations.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING VISUALIZATION PIPELINE")
    print("=" * 60)
    
    # Load all data
    metadata, embeddings = load_all_data()
    
    if not embeddings:
        print("\n⚠ No embeddings found! Please run scripts 02 and 03 first.")
        return
    
    # Create comparison plots
    create_comparison_plot(metadata, embeddings, method='tsne')
    create_comparison_plot(metadata, embeddings, method='pca')
    
    # Create detailed visualization
    create_detailed_plot(metadata, embeddings)
    
    # Create similarity heatmaps
    plot_similarity_heatmap(metadata, embeddings)
    
    # Calculate clustering metrics
    calculate_clustering_metrics(metadata, embeddings)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("\nWhat to look for in the plots:")
    print("  1. Do samples cluster by disease type?")
    print("  2. Are the clusters similar across embedding methods?")
    print("  3. Which method gives the best separation?")
    print("\nNext step:")
    print("  Run 05_similarity_search.py to explore similarity search")


if __name__ == "__main__":
    main()

