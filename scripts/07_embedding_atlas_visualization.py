"""
Script 7: Embedding Atlas Interactive Visualization
===================================================

This script prepares your embeddings for visualization with Apple's
Embedding Atlas tool - an interactive embedding explorer.

GitHub: https://github.com/apple/embedding-atlas

Features:
- Interactive 2D embedding visualization
- Automatic clustering and labeling
- Real-time search and nearest neighbors
- Cross-filtering with metadata

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def prepare_embedding_atlas_data():
    """
    Prepare data in the format Embedding Atlas expects.
    
    Embedding Atlas works best with:
    - A DataFrame with x, y coordinates (2D projection)
    - Metadata columns for filtering/coloring
    - Original high-dimensional embeddings (optional)
    """
    print("=" * 60)
    print("Preparing Data for Embedding Atlas")
    print("=" * 60)
    
    # Load metadata
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    print(f"✓ Loaded metadata: {len(metadata)} samples")
    
    # Load all embeddings
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    expr_pca = np.load(DATA_DIR / "expression_embeddings_pca.npy")
    expr_ae = np.load(DATA_DIR / "expression_embeddings_autoencoder.npy")
    
    # Load aligned embeddings if available
    try:
        text_aligned = np.load(DATA_DIR / "text_embeddings_aligned.npy")
        expr_aligned = np.load(DATA_DIR / "expression_embeddings_aligned.npy")
        has_aligned = True
        print("✓ Loaded aligned embeddings")
    except FileNotFoundError:
        has_aligned = False
        print("⚠ Aligned embeddings not found (run script 06 first)")
    
    # ========================================================================
    # Create DataFrame 1: Text Embeddings
    # ========================================================================
    print("\nCreating text embeddings dataset...")
    
    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    text_2d = tsne.fit_transform(text_emb)
    
    df_text = pd.DataFrame({
        'x': text_2d[:, 0],
        'y': text_2d[:, 1],
        'sample_id': metadata['sample_title'],
        'geo_accession': metadata['geo_accession'],
        'disease': metadata['disease_abbrev'],
        'disease_full': metadata['disease'],
        'age': metadata['age'],
        'sex': metadata['gender'],
        'treatment': metadata['treatment'],
        'text_description': metadata['text_description'],
        'embedding_type': 'BioBERT (Text)'
    })
    
    # Save as parquet (Embedding Atlas preferred format)
    df_text.to_parquet(RESULTS_DIR / "atlas_text_embeddings.parquet")
    print(f"  ✓ Saved: atlas_text_embeddings.parquet")
    
    # ========================================================================
    # Create DataFrame 2: Expression Embeddings (PCA)
    # ========================================================================
    print("\nCreating expression PCA embeddings dataset...")
    
    # Use first 2 PCs directly (already 2D-ish) or apply t-SNE
    if expr_pca.shape[1] >= 2:
        expr_2d = expr_pca[:, :2]  # Use first 2 PCs
    else:
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        expr_2d = tsne.fit_transform(expr_pca)
    
    df_expr = pd.DataFrame({
        'x': expr_2d[:, 0],
        'y': expr_2d[:, 1],
        'sample_id': metadata['sample_title'],
        'geo_accession': metadata['geo_accession'],
        'disease': metadata['disease_abbrev'],
        'disease_full': metadata['disease'],
        'age': metadata['age'],
        'sex': metadata['gender'],
        'treatment': metadata['treatment'],
        'text_description': metadata['text_description'],
        'embedding_type': 'PCA (Expression)'
    })
    
    df_expr.to_parquet(RESULTS_DIR / "atlas_expression_embeddings.parquet")
    print(f"  ✓ Saved: atlas_expression_embeddings.parquet")
    
    # ========================================================================
    # Create DataFrame 3: Aligned Embeddings (COMBINED!)
    # ========================================================================
    if has_aligned:
        print("\nCreating aligned embeddings dataset (MOST IMPORTANT!)...")
        
        # Combine text and expression aligned embeddings
        combined = np.vstack([text_aligned, expr_aligned])
        
        # Reduce combined to 2D
        tsne = TSNE(n_components=2, perplexity=8, random_state=42)
        combined_2d = tsne.fit_transform(combined)
        
        n_samples = len(metadata)
        
        # Create combined dataframe
        df_aligned = pd.DataFrame({
            'x': combined_2d[:, 0],
            'y': combined_2d[:, 1],
            'sample_id': list(metadata['sample_title']) + list(metadata['sample_title']),
            'geo_accession': list(metadata['geo_accession']) + list(metadata['geo_accession']),
            'disease': list(metadata['disease_abbrev']) + list(metadata['disease_abbrev']),
            'disease_full': list(metadata['disease']) + list(metadata['disease']),
            'age': list(metadata['age']) + list(metadata['age']),
            'sex': list(metadata['gender']) + list(metadata['gender']),
            'modality': ['Text'] * n_samples + ['Expression'] * n_samples,
            'text_description': list(metadata['text_description']) + list(metadata['text_description']),
        })
        
        df_aligned.to_parquet(RESULTS_DIR / "atlas_aligned_embeddings.parquet")
        print(f"  ✓ Saved: atlas_aligned_embeddings.parquet")
    
    # ========================================================================
    # Create Combined DataFrame (All Methods)
    # ========================================================================
    print("\nCreating combined comparison dataset...")
    
    # Combine all embeddings for comparison
    all_embeddings = []
    
    # Text embeddings
    tsne_text = TSNE(n_components=2, perplexity=5, random_state=42)
    text_2d = tsne_text.fit_transform(text_emb)
    for i in range(len(metadata)):
        all_embeddings.append({
            'x': text_2d[i, 0],
            'y': text_2d[i, 1],
            'sample_id': metadata.iloc[i]['sample_title'],
            'disease': metadata.iloc[i]['disease_abbrev'],
            'age': metadata.iloc[i]['age'],
            'sex': metadata.iloc[i]['gender'],
            'method': 'BioBERT (Text)',
            'modality': 'Text'
        })
    
    # Expression PCA
    pca_2d = PCA(n_components=2).fit_transform(expr_pca)
    for i in range(len(metadata)):
        all_embeddings.append({
            'x': pca_2d[i, 0],
            'y': pca_2d[i, 1],
            'sample_id': metadata.iloc[i]['sample_title'],
            'disease': metadata.iloc[i]['disease_abbrev'],
            'age': metadata.iloc[i]['age'],
            'sex': metadata.iloc[i]['gender'],
            'method': 'PCA (Expression)',
            'modality': 'Expression'
        })
    
    # Expression Autoencoder
    tsne_ae = TSNE(n_components=2, perplexity=5, random_state=42)
    ae_2d = tsne_ae.fit_transform(expr_ae)
    for i in range(len(metadata)):
        all_embeddings.append({
            'x': ae_2d[i, 0],
            'y': ae_2d[i, 1],
            'sample_id': metadata.iloc[i]['sample_title'],
            'disease': metadata.iloc[i]['disease_abbrev'],
            'age': metadata.iloc[i]['age'],
            'sex': metadata.iloc[i]['gender'],
            'method': 'Autoencoder (Expression)',
            'modality': 'Expression'
        })
    
    df_all = pd.DataFrame(all_embeddings)
    df_all.to_parquet(RESULTS_DIR / "atlas_all_embeddings.parquet")
    print(f"  ✓ Saved: atlas_all_embeddings.parquet")
    
    # Also save as CSV for easy inspection
    df_all.to_csv(RESULTS_DIR / "atlas_all_embeddings.csv", index=False)
    
    return df_aligned if has_aligned else df_all


def print_usage_instructions():
    """Print instructions for using Embedding Atlas."""
    print("\n" + "=" * 60)
    print("HOW TO USE EMBEDDING ATLAS")
    print("=" * 60)
    
    print("""
Option 1: Command Line (Recommended)
─────────────────────────────────────
pip install embedding-atlas
embedding-atlas results/atlas_aligned_embeddings.parquet

Option 2: Jupyter Notebook Widget
─────────────────────────────────
from embedding_atlas.widget import EmbeddingAtlasWidget
import pandas as pd

df = pd.read_parquet('results/atlas_aligned_embeddings.parquet')
EmbeddingAtlasWidget(df)

Option 3: Online Demo
────────────────────
Visit: https://apple.github.io/embedding-atlas
Upload your parquet file there!

Files Created:
─────────────
1. atlas_text_embeddings.parquet      - BioBERT text embeddings
2. atlas_expression_embeddings.parquet - PCA expression embeddings  
3. atlas_aligned_embeddings.parquet   - ALIGNED embeddings (best!)
4. atlas_all_embeddings.parquet       - All methods combined

RECOMMENDED: Start with atlas_aligned_embeddings.parquet
This shows text and expression in the SAME space (CellWhisperer-style!)
""")


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("EMBEDDING ATLAS DATA PREPARATION")
    print("=" * 60)
    
    # Prepare data
    df = prepare_embedding_atlas_data()
    
    # Print instructions
    print_usage_instructions()
    
    print("\n" + "=" * 60)
    print("READY FOR EMBEDDING ATLAS!")
    print("=" * 60)


if __name__ == "__main__":
    main()

