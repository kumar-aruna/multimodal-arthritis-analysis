"""
Script 3: Create Gene Expression Embeddings
============================================

This script creates embeddings from gene expression data.

Important Note about Geneformer:
--------------------------------
Geneformer is designed specifically for SINGLE-CELL RNA-seq data.
Your data (GSE36700) is BULK microarray data, so Geneformer isn't
directly applicable.

Instead, we'll use practical alternatives:
1. PCA (Principal Component Analysis) - Simple, interpretable
2. Autoencoder - Learns compressed representations

Both methods create meaningful embeddings that capture the
biological variation in your gene expression data.

What these methods do:
- PCA: Finds directions of maximum variance (linear transformation)
- Autoencoder: Learns a compressed representation (non-linear)

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Embedding dimensions (to match BioBERT's 768 for comparison)
# Or use smaller dimension for faster computation
EMBEDDING_DIM = 128  # Can be 768 to match BioBERT


# ============================================================================
# STEP 1: LOAD AND PREPROCESS EXPRESSION DATA
# ============================================================================

def load_expression_data():
    """
    Load and preprocess the gene expression matrix.
    """
    print("=" * 60)
    print("STEP 1: Loading Expression Data")
    print("=" * 60)
    
    # Load transposed matrix (samples × genes)
    expression_path = DATA_DIR / "expression_matrix_transposed.csv"
    
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {expression_path}\n"
            "Please run 01_parse_geo_data.py first!"
        )
    
    df = pd.read_csv(expression_path, index_col=0)
    
    print(f"\n✓ Loaded expression matrix:")
    print(f"  - Samples: {df.shape[0]}")
    print(f"  - Genes/Probes: {df.shape[1]}")
    
    return df


def preprocess_expression(expression_df):
    """
    Preprocess expression data for embedding.
    
    Steps:
    1. Log2 transform (if not already done)
    2. Remove low-variance genes
    3. Standardize (zero mean, unit variance)
    
    Returns:
        numpy array: Preprocessed expression matrix
        list: Gene names kept after filtering
    """
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Expression Data")
    print("=" * 60)
    
    X = expression_df.values.copy()
    gene_names = list(expression_df.columns)
    
    # Check if data needs log transformation
    if X.max() > 100:  # Likely not log-transformed
        print("\nApplying log2 transformation...")
        X = np.log2(X + 1)
    
    print(f"\nBefore filtering: {X.shape[1]} genes")
    
    # Remove low-variance genes (keep top 5000 most variable)
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-5000:]
    X = X[:, top_indices]
    gene_names = [gene_names[i] for i in top_indices]
    
    print(f"After filtering: {X.shape[1]} genes (top variable)")
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print("\n✓ Preprocessing complete")
    print(f"  - Final shape: {X.shape}")
    print(f"  - Mean: {X.mean():.6f} (should be ~0)")
    print(f"  - Std: {X.std():.6f} (should be ~1)")
    
    return X, gene_names


# ============================================================================
# METHOD 1: PCA EMBEDDINGS
# ============================================================================

def create_pca_embeddings(X, n_components=128):
    """
    Create embeddings using Principal Component Analysis.
    
    PCA is a linear dimensionality reduction technique that:
    1. Finds directions of maximum variance
    2. Projects data onto these directions
    
    Why PCA works for gene expression:
    - Gene expression is often driven by a few biological factors
    - PCA captures these major sources of variation
    - First few PCs often correspond to biological signals
    
    Args:
        X: Preprocessed expression matrix (samples × genes)
        n_components: Number of dimensions for embedding
        
    Returns:
        embeddings: PCA embeddings
        pca: Fitted PCA object (for later analysis)
    """
    print("\n" + "=" * 60)
    print("METHOD 1: PCA Embeddings")
    print("=" * 60)
    
    # Fit PCA
    # Use min of n_components and n_samples
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    
    print(f"\nFitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(X)
    
    # Report variance explained
    var_explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(var_explained)
    
    print(f"\n✓ PCA complete")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"\nVariance explained:")
    print(f"  - PC1: {var_explained[0]*100:.1f}%")
    print(f"  - PC2: {var_explained[1]*100:.1f}%")
    print(f"  - PC3: {var_explained[2]*100:.1f}%")
    print(f"  - First 10 PCs: {cumsum[9]*100:.1f}%")
    print(f"  - All {n_components} PCs: {cumsum[-1]*100:.1f}%")
    
    return embeddings, pca


# ============================================================================
# METHOD 2: AUTOENCODER EMBEDDINGS
# ============================================================================

class GeneAutoencoder(nn.Module):
    """
    Simple autoencoder for gene expression data.
    
    Architecture:
        Encoder: input → 1024 → 512 → 256 → embedding_dim
        Decoder: embedding_dim → 256 → 512 → 1024 → output
    
    The bottleneck (embedding_dim) forces the model to learn
    a compressed representation of the data.
    """
    
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, embedding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def create_autoencoder_embeddings(X, embedding_dim=128, epochs=100):
    """
    Create embeddings using an autoencoder.
    
    How it works:
    1. Train autoencoder to reconstruct gene expression
    2. The bottleneck layer learns compressed representations
    3. Extract embeddings from the trained encoder
    
    Args:
        X: Preprocessed expression matrix
        embedding_dim: Size of the embedding
        epochs: Training epochs
        
    Returns:
        embeddings: Autoencoder embeddings
        model: Trained autoencoder
    """
    print("\n" + "=" * 60)
    print("METHOD 2: Autoencoder Embeddings")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Initialize model
    model = GeneAutoencoder(X.shape[1], embedding_dim).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining autoencoder for {epochs} epochs...")
    model.train()
    
    losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, _ = model(X_tensor)
        loss = criterion(x_recon, X_tensor)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Extract embeddings
    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_tensor)
        embeddings = embeddings.cpu().numpy()
    
    print(f"\n✓ Autoencoder training complete")
    print(f"  - Final loss: {losses[-1]:.6f}")
    print(f"  - Embedding shape: {embeddings.shape}")
    
    return embeddings, model, losses


# ============================================================================
# STEP 3: COMPARE METHODS
# ============================================================================

def compare_embeddings(pca_emb, ae_emb, metadata_df):
    """
    Compare PCA and autoencoder embeddings.
    """
    print("\n" + "=" * 60)
    print("Comparing Embedding Methods")
    print("=" * 60)
    
    from scipy.spatial.distance import cosine
    
    for name, emb in [("PCA", pca_emb), ("Autoencoder", ae_emb)]:
        print(f"\n{name} Embeddings:")
        
        # Within-group similarity
        print("  Average within-group similarity:")
        for disease in metadata_df['disease_abbrev'].unique():
            mask = metadata_df['disease_abbrev'] == disease
            disease_emb = emb[mask.values]
            
            if len(disease_emb) > 1:
                sims = []
                for i in range(len(disease_emb)):
                    for j in range(i + 1, len(disease_emb)):
                        sim = 1 - cosine(disease_emb[i], disease_emb[j])
                        sims.append(sim)
                print(f"    {disease}: {np.mean(sims):.4f}")


# ============================================================================
# STEP 4: SAVE EMBEDDINGS
# ============================================================================

def save_embeddings(pca_emb, ae_emb, metadata_df):
    """
    Save all embeddings to files.
    """
    print("\n" + "=" * 60)
    print("Saving Embeddings")
    print("=" * 60)
    
    # Save PCA embeddings
    pca_path = DATA_DIR / "expression_embeddings_pca.npy"
    np.save(pca_path, pca_emb)
    print(f"\n✓ Saved PCA embeddings to: {pca_path}")
    
    # Save autoencoder embeddings
    ae_path = DATA_DIR / "expression_embeddings_autoencoder.npy"
    np.save(ae_path, ae_emb)
    print(f"✓ Saved autoencoder embeddings to: {ae_path}")
    
    # Save as CSVs too
    sample_names = metadata_df['sample_title'].values
    
    pca_df = pd.DataFrame(
        pca_emb,
        index=sample_names,
        columns=[f"PC{i+1}" for i in range(pca_emb.shape[1])]
    )
    pca_df.to_csv(DATA_DIR / "expression_embeddings_pca.csv")
    
    ae_df = pd.DataFrame(
        ae_emb,
        index=sample_names,
        columns=[f"dim_{i}" for i in range(ae_emb.shape[1])]
    )
    ae_df.to_csv(DATA_DIR / "expression_embeddings_autoencoder.csv")
    
    print("✓ Saved CSV versions for easy inspection")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the expression embedding pipeline.
    """
    print("\n" + "=" * 60)
    print("GENE EXPRESSION EMBEDDING PIPELINE")
    print("=" * 60)
    
    # Load metadata for labels
    metadata_df = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    
    # Step 1: Load expression data
    expression_df = load_expression_data()
    
    # Step 2: Preprocess
    X, gene_names = preprocess_expression(expression_df)
    
    # Method 1: PCA embeddings
    pca_emb, pca_model = create_pca_embeddings(X, n_components=EMBEDDING_DIM)
    
    # Method 2: Autoencoder embeddings
    ae_emb, ae_model, losses = create_autoencoder_embeddings(
        X, 
        embedding_dim=EMBEDDING_DIM, 
        epochs=100
    )
    
    # Compare methods
    compare_embeddings(pca_emb, ae_emb, metadata_df)
    
    # Save embeddings
    save_embeddings(pca_emb, ae_emb, metadata_df)
    
    print("\n" + "=" * 60)
    print("EXPRESSION EMBEDDING COMPLETE!")
    print("=" * 60)
    print("\nNext step:")
    print("  Run 04_visualize_embeddings.py to visualize all embeddings")
    
    return pca_emb, ae_emb


if __name__ == "__main__":
    pca_embeddings, ae_embeddings = main()

