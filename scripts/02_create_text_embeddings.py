"""
Script 2: Create Text Embeddings with BioBERT
=============================================

This script creates text embeddings using BioBERT, a pre-trained
language model specifically designed for biomedical text.

What is BioBERT?
----------------
BioBERT is BERT pre-trained on:
- PubMed abstracts (4.5B words)
- PMC full-text articles (13.5B words)

This makes it excellent at understanding medical terminology like
"rheumatoid arthritis", "synovial biopsy", etc.

What this script does:
1. Loads sample metadata with text descriptions
2. Loads BioBERT model from Hugging Face
3. Converts each text description to a 768-dimensional embedding
4. Saves embeddings for visualization

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

# BioBERT model from Hugging Face
# Alternative models you can try:
# - "dmis-lab/biobert-v1.1" (original BioBERT)
# - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" (PubMedBERT)
# - "allenai/scibert_scivocab_uncased" (SciBERT)
MODEL_NAME = "dmis-lab/biobert-v1.1"


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_metadata():
    """
    Load sample metadata with text descriptions.
    """
    print("=" * 60)
    print("STEP 1: Loading Metadata")
    print("=" * 60)
    
    metadata_path = DATA_DIR / "sample_metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run 01_parse_geo_data.py first!"
        )
    
    df = pd.read_csv(metadata_path)
    print(f"\n✓ Loaded {len(df)} samples")
    print(f"\nText description example:")
    print(f"  {df['text_description'].iloc[0]}")
    
    return df


# ============================================================================
# STEP 2: LOAD BioBERT MODEL
# ============================================================================

def load_biobert_model():
    """
    Load BioBERT model and tokenizer from Hugging Face.
    
    The model will be downloaded automatically on first run
    and cached for future use.
    
    Returns:
        tokenizer: BioBERT tokenizer
        model: BioBERT model
        device: torch device (cuda or cpu)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Loading BioBERT Model")
    print("=" * 60)
    
    print(f"\nModel: {MODEL_NAME}")
    print("(This may take a few minutes on first run to download...)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Tokenizer loaded")
    
    # Load model
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("✓ Model loaded")
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Using device: {device}")
    
    return tokenizer, model, device


# ============================================================================
# STEP 3: CREATE EMBEDDINGS
# ============================================================================

def get_embedding(text, tokenizer, model, device):
    """
    Get the embedding for a single text.
    
    How it works:
    1. Tokenize: Convert text to token IDs
    2. Forward pass: Run through BioBERT
    3. Pool: Take the [CLS] token embedding as the sentence representation
    
    The [CLS] token is a special token at the beginning of each input
    that BERT uses to aggregate information from the entire sequence.
    
    Args:
        text: Input text string
        tokenizer: BioBERT tokenizer
        model: BioBERT model
        device: torch device
        
    Returns:
        numpy array: 768-dimensional embedding
    """
    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract [CLS] token embedding (first token)
    # Shape: (1, 768) -> (768,)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    
    return embedding


def create_all_embeddings(metadata_df, tokenizer, model, device):
    """
    Create embeddings for all samples.
    
    Returns:
        numpy array: Shape (n_samples, 768)
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Text Embeddings")
    print("=" * 60)
    
    embeddings = []
    
    print(f"\nProcessing {len(metadata_df)} samples...")
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        text = row['text_description']
        embedding = get_embedding(text, tokenizer, model, device)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    print(f"\n✓ Created embeddings with shape: {embeddings.shape}")
    print(f"  - Samples: {embeddings.shape[0]}")
    print(f"  - Dimensions: {embeddings.shape[1]}")
    
    return embeddings


# ============================================================================
# STEP 4: ANALYZE EMBEDDINGS
# ============================================================================

def analyze_embeddings(embeddings, metadata_df):
    """
    Basic analysis of the embeddings.
    
    This helps verify that the embeddings are meaningful:
    - Similar diseases should have similar embeddings
    - Different diseases should have different embeddings
    """
    print("\n" + "=" * 60)
    print("STEP 4: Analyzing Embeddings")
    print("=" * 60)
    
    from scipy.spatial.distance import cosine
    
    # Calculate pairwise similarities within each disease group
    print("\nAverage similarity within disease groups:")
    
    for disease in metadata_df['disease_abbrev'].unique():
        mask = metadata_df['disease_abbrev'] == disease
        disease_embeddings = embeddings[mask]
        
        if len(disease_embeddings) > 1:
            # Calculate average pairwise similarity
            similarities = []
            for i in range(len(disease_embeddings)):
                for j in range(i + 1, len(disease_embeddings)):
                    sim = 1 - cosine(disease_embeddings[i], disease_embeddings[j])
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            print(f"  {disease}: {avg_sim:.4f} (n={len(disease_embeddings)})")
    
    # Calculate average similarity between disease groups
    print("\nAverage similarity between disease groups:")
    diseases = metadata_df['disease_abbrev'].unique()
    
    for i, disease1 in enumerate(diseases):
        for disease2 in diseases[i+1:]:
            mask1 = metadata_df['disease_abbrev'] == disease1
            mask2 = metadata_df['disease_abbrev'] == disease2
            
            emb1 = embeddings[mask1]
            emb2 = embeddings[mask2]
            
            similarities = []
            for e1 in emb1:
                for e2 in emb2:
                    sim = 1 - cosine(e1, e2)
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            print(f"  {disease1} vs {disease2}: {avg_sim:.4f}")


# ============================================================================
# STEP 5: SAVE EMBEDDINGS
# ============================================================================

def save_embeddings(embeddings, metadata_df):
    """
    Save embeddings to file.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Saving Embeddings")
    print("=" * 60)
    
    # Save as numpy array
    npy_path = DATA_DIR / "text_embeddings.npy"
    np.save(npy_path, embeddings)
    print(f"\n✓ Saved embeddings to: {npy_path}")
    
    # Also save as CSV with sample IDs for easier inspection
    csv_path = DATA_DIR / "text_embeddings.csv"
    embedding_df = pd.DataFrame(
        embeddings,
        index=metadata_df['sample_title'],
        columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
    )
    embedding_df.to_csv(csv_path)
    print(f"✓ Saved embeddings CSV to: {csv_path}")
    
    return npy_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the complete embedding pipeline.
    """
    print("\n" + "=" * 60)
    print("TEXT EMBEDDING PIPELINE WITH BioBERT")
    print("=" * 60)
    
    # Step 1: Load metadata
    metadata_df = load_metadata()
    
    # Step 2: Load BioBERT model
    tokenizer, model, device = load_biobert_model()
    
    # Step 3: Create embeddings
    embeddings = create_all_embeddings(metadata_df, tokenizer, model, device)
    
    # Step 4: Analyze embeddings
    analyze_embeddings(embeddings, metadata_df)
    
    # Step 5: Save embeddings
    save_embeddings(embeddings, metadata_df)
    
    print("\n" + "=" * 60)
    print("TEXT EMBEDDING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run 03_create_expression_embeddings.py for gene embeddings")
    print("  2. Run 04_visualize_embeddings.py to visualize and compare")
    
    return embeddings


if __name__ == "__main__":
    embeddings = main()

