"""
Script 6: CellWhisperer-Style Contrastive Learning
==================================================

This script implements contrastive learning similar to how CellWhisperer
aligns text and expression embeddings.

Based on the actual CellWhisperer code from:
- src/cellwhisperer/jointemb/model.py
- src/cellwhisperer/jointemb/loss/losses.py

What CellWhisperer does:
1. Uses Geneformer/scGPT/UCE for transcriptome encoding
2. Uses BERT for text encoding  
3. Projects both to a shared space (2048 dims)
4. Uses CLIP loss to align them

What we do (simplified version):
1. Use our PCA embeddings for transcriptome
2. Use BioBERT embeddings for text
3. Train projection layers to align them
4. Use CLIP loss

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

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# Training settings
PROJECTION_DIM = 256  # CellWhisperer uses 2048, we use smaller for 25 samples
LEARNING_RATE = 0.001
EPOCHS = 200
TEMPERATURE = 0.07  # CLIP uses 0.07


# ============================================================================
# CONTRASTIVE LOSS (from CellWhisperer)
# ============================================================================

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    CLIP-style contrastive loss.
    
    From CellWhisperer's losses.py:
    The loss encourages matching pairs (diagonal) to have high similarity
    and non-matching pairs (off-diagonal) to have low similarity.
    """
    # Labels are just the diagonal indices (0, 1, 2, ..., N-1)
    labels = torch.arange(len(logits), device=logits.device)
    return F.cross_entropy(logits, labels)


class CLIPLoss(nn.Module):
    """
    CLIP Loss as implemented in CellWhisperer.
    
    Computes symmetric loss:
    - Loss from text perspective (which transcriptome matches this text?)
    - Loss from transcriptome perspective (which text matches this transcriptome?)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_embeds, transcriptome_embeds):
        # Normalize embeddings
        text_embeds = F.normalize(text_embeds, dim=-1)
        transcriptome_embeds = F.normalize(transcriptome_embeds, dim=-1)
        
        # Compute similarity matrix
        # Shape: (batch_size, batch_size)
        logits = torch.matmul(text_embeds, transcriptome_embeds.T) / self.temperature
        
        # Symmetric loss
        text_loss = contrastive_loss(logits)
        transcriptome_loss = contrastive_loss(logits.T)
        
        return (text_loss + transcriptome_loss) / 2.0


# ============================================================================
# PROJECTION MODEL (inspired by CellWhisperer's Discriminator)
# ============================================================================

class ProjectionHead(nn.Module):
    """
    Projects embeddings to shared space.
    
    CellWhisperer uses GlobalDiscriminatorDot with:
    - Separate projection for transcriptome (img_block)
    - Separate projection for text (text_block)
    - Batch layer normalization
    """
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
    """
    Simplified version of CellWhisperer's TranscriptomeTextDualEncoderModel.
    
    Takes pre-computed embeddings and learns to align them.
    """
    def __init__(self, text_dim, transcriptome_dim, projection_dim):
        super().__init__()
        
        # Projection heads (like CellWhisperer's discriminator)
        self.text_projection = ProjectionHead(text_dim, projection_dim)
        self.transcriptome_projection = ProjectionHead(transcriptome_dim, projection_dim)
        
        # Temperature parameter (learnable like in CLIP)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, text_embeds, transcriptome_embeds):
        # Project to shared space
        text_projected = self.text_projection(text_embeds)
        transcriptome_projected = self.transcriptome_projection(transcriptome_embeds)
        
        return text_projected, transcriptome_projected
    
    def get_temperature(self):
        return self.temperature.exp()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_embeddings():
    """Load pre-computed embeddings."""
    print("=" * 60)
    print("Loading Pre-computed Embeddings")
    print("=" * 60)
    
    # Load metadata
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    
    # Load embeddings
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    expr_emb = np.load(DATA_DIR / "expression_embeddings_pca.npy")
    
    print(f"✓ Text embeddings: {text_emb.shape}")
    print(f"✓ Expression embeddings: {expr_emb.shape}")
    print(f"✓ Samples: {len(metadata)}")
    
    return metadata, text_emb, expr_emb


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_alignment(text_emb, expr_emb, metadata, epochs=200):
    """
    Train the projection layers to align text and expression embeddings.
    
    This is a simplified version of CellWhisperer's training loop.
    """
    print("\n" + "=" * 60)
    print("Training Contrastive Alignment")
    print("(CellWhisperer-style)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Convert to tensors
    text_tensor = torch.FloatTensor(text_emb).to(device)
    expr_tensor = torch.FloatTensor(expr_emb).to(device)
    
    # Initialize model
    model = DualEncoderModel(
        text_dim=text_emb.shape[1],
        transcriptome_dim=expr_emb.shape[1],
        projection_dim=PROJECTION_DIM
    ).to(device)
    
    # Loss and optimizer
    criterion = CLIPLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    losses = []
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        text_proj, expr_proj = model(text_tensor, expr_tensor)
        
        # Compute loss
        loss = criterion(text_proj, expr_proj)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 50 == 0:
            tqdm.write(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final loss: {losses[-1]:.4f}")
    
    return model, losses


# ============================================================================
# EXTRACT ALIGNED EMBEDDINGS
# ============================================================================

def get_aligned_embeddings(model, text_emb, expr_emb):
    """Extract the aligned embeddings from the trained model."""
    device = next(model.parameters()).device
    
    text_tensor = torch.FloatTensor(text_emb).to(device)
    expr_tensor = torch.FloatTensor(expr_emb).to(device)
    
    model.eval()
    with torch.no_grad():
        text_aligned, expr_aligned = model(text_tensor, expr_tensor)
        
        # Normalize (like CellWhisperer does)
        text_aligned = F.normalize(text_aligned, dim=-1)
        expr_aligned = F.normalize(expr_aligned, dim=-1)
    
    return text_aligned.cpu().numpy(), expr_aligned.cpu().numpy()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_alignment(text_aligned, expr_aligned, metadata, losses):
    """Visualize the aligned embeddings."""
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    from sklearn.manifold import TSNE
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color map
    colors = {
        'OA': '#1f77b4', 'RA': '#d62728', 'SLE': '#9467bd',
        'MIC': '#2ca02c', 'SA': '#ff7f0e'
    }
    labels = metadata['disease_abbrev'].values
    
    # Plot 1: Training loss
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CLIP Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Aligned embeddings (combined)
    # Combine and reduce dimensions
    combined = np.vstack([text_aligned, expr_aligned])
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    n_samples = len(text_aligned)
    text_2d = combined_2d[:n_samples]
    expr_2d = combined_2d[n_samples:]
    
    # Plot text embeddings (circles)
    for disease in colors.keys():
        mask = labels == disease
        if mask.any():
            axes[1].scatter(text_2d[mask, 0], text_2d[mask, 1],
                          c=colors[disease], label=f'{disease} (text)',
                          marker='o', s=100, alpha=0.7)
    
    # Plot expression embeddings (triangles)
    for disease in colors.keys():
        mask = labels == disease
        if mask.any():
            axes[1].scatter(expr_2d[mask, 0], expr_2d[mask, 1],
                          c=colors[disease], label=f'{disease} (expr)',
                          marker='^', s=100, alpha=0.7)
    
    # Draw lines connecting matching pairs
    for i in range(n_samples):
        axes[1].plot([text_2d[i, 0], expr_2d[i, 0]],
                    [text_2d[i, 1], expr_2d[i, 1]],
                    'k-', alpha=0.2, linewidth=0.5)
    
    axes[1].set_title('Aligned Embeddings\n(circles=text, triangles=expression)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Similarity matrix
    similarity = np.dot(text_aligned, expr_aligned.T)
    im = axes[2].imshow(similarity, cmap='RdYlBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Text-Expression Similarity\n(diagonal should be high)')
    axes[2].set_xlabel('Expression samples')
    axes[2].set_ylabel('Text samples')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    # Save
    save_path = RESULTS_DIR / "aligned_embeddings.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    
    return fig


# ============================================================================
# CROSS-MODAL SEARCH DEMO
# ============================================================================

def demo_cross_modal_search(text_aligned, expr_aligned, metadata):
    """
    Demo: Search expression samples using text queries.
    
    This is what CellWhisperer enables!
    """
    print("\n" + "=" * 60)
    print("Cross-Modal Search Demo")
    print("(This is what CellWhisperer does!)")
    print("=" * 60)
    
    # Compute similarity matrix
    similarity = np.dot(text_aligned, expr_aligned.T)
    
    print("\nFor each text description, find matching expression sample:")
    print("-" * 60)
    
    correct = 0
    for i in range(len(metadata)):
        # Find most similar expression sample for this text
        best_match_idx = np.argmax(similarity[i])
        
        true_disease = metadata.iloc[i]['disease_abbrev']
        pred_disease = metadata.iloc[best_match_idx]['disease_abbrev']
        
        match = "✓" if i == best_match_idx else ("~" if true_disease == pred_disease else "✗")
        if i == best_match_idx:
            correct += 1
        
        print(f"  Text '{metadata.iloc[i]['sample_title']}' ({true_disease}) "
              f"→ Expr '{metadata.iloc[best_match_idx]['sample_title']}' ({pred_disease}) {match}")
    
    accuracy = correct / len(metadata) * 100
    print(f"\n  Exact match accuracy: {accuracy:.1f}%")
    print(f"  (Perfect alignment would be 100% - diagonal matches)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("CELLWHISPERER-STYLE CONTRASTIVE LEARNING")
    print("=" * 60)
    
    # Load embeddings
    try:
        metadata, text_emb, expr_emb = load_embeddings()
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("Please run scripts 01, 02, and 03 first!")
        return
    
    # Train alignment
    model, losses = train_alignment(text_emb, expr_emb, metadata, epochs=EPOCHS)
    
    # Get aligned embeddings
    text_aligned, expr_aligned = get_aligned_embeddings(model, text_emb, expr_emb)
    
    # Save aligned embeddings
    np.save(DATA_DIR / "text_embeddings_aligned.npy", text_aligned)
    np.save(DATA_DIR / "expression_embeddings_aligned.npy", expr_aligned)
    print(f"\n✓ Saved aligned embeddings to {DATA_DIR}")
    
    # Visualize
    visualize_alignment(text_aligned, expr_aligned, metadata, losses)
    
    # Demo cross-modal search
    demo_cross_modal_search(text_aligned, expr_aligned, metadata)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print("\nWhat we achieved:")
    print("  1. Trained projection layers (like CellWhisperer's discriminator)")
    print("  2. Used CLIP loss to align text and expression")
    print("  3. Now text and expression are in the SAME space!")
    print("  4. Can search expression samples using text queries!")
    print("\nThis is the core of how CellWhisperer works!")


if __name__ == "__main__":
    main()

