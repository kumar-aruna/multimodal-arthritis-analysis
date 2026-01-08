"""
Script 5: Similarity Search Demo
================================

This script demonstrates how to use embeddings for:
1. Finding similar samples based on expression
2. Searching samples by text query
3. Cross-modal search (text → expression, expression → text)

This is the core functionality that CellWhisperer provides:
- Query: "Show me samples with rheumatoid arthritis"
- Result: Returns samples with similar embeddings

Author: Aruna (Bioinformatics Project)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cdist
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():
    """
    Load metadata and embeddings.
    """
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load metadata
    metadata = pd.read_csv(DATA_DIR / "sample_metadata.csv")
    
    # Load embeddings
    text_emb = np.load(DATA_DIR / "text_embeddings.npy")
    pca_emb = np.load(DATA_DIR / "expression_embeddings_pca.npy")
    ae_emb = np.load(DATA_DIR / "expression_embeddings_autoencoder.npy")
    
    print(f"✓ Loaded {len(metadata)} samples")
    
    return metadata, text_emb, pca_emb, ae_emb


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return 1 - cosine(vec1, vec2)


def find_similar_samples(query_idx, embeddings, metadata, top_k=5):
    """
    Find the most similar samples to a query sample.
    
    Args:
        query_idx: Index of the query sample
        embeddings: Embedding matrix
        metadata: Sample metadata
        top_k: Number of similar samples to return
        
    Returns:
        DataFrame with similar samples and their similarities
    """
    query_emb = embeddings[query_idx]
    
    # Calculate similarities to all samples
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_emb, emb)
        similarities.append({
            'index': i,
            'sample': metadata.iloc[i]['sample_title'],
            'disease': metadata.iloc[i]['disease_abbrev'],
            'similarity': sim
        })
    
    # Sort by similarity (exclude self)
    results = pd.DataFrame(similarities)
    results = results[results['index'] != query_idx]
    results = results.sort_values('similarity', ascending=False).head(top_k)
    
    return results


def search_by_disease(disease_abbrev, embeddings, metadata):
    """
    Get the average embedding for a disease type.
    Can be used as a "prototype" for that disease.
    """
    mask = metadata['disease_abbrev'] == disease_abbrev
    disease_embeddings = embeddings[mask]
    return disease_embeddings.mean(axis=0)


def find_samples_by_disease_query(disease, embeddings, metadata, top_k=5):
    """
    Find samples most similar to a disease "prototype".
    
    This is like querying: "Show me samples similar to rheumatoid arthritis"
    """
    # Get disease prototype (average embedding)
    prototype = search_by_disease(disease, embeddings, metadata)
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(prototype, emb)
        similarities.append({
            'index': i,
            'sample': metadata.iloc[i]['sample_title'],
            'disease': metadata.iloc[i]['disease_abbrev'],
            'similarity': sim
        })
    
    results = pd.DataFrame(similarities)
    results = results.sort_values('similarity', ascending=False).head(top_k)
    
    return results


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_similar_samples(metadata, embeddings, embedding_name):
    """
    Demo: Find similar samples to a query sample.
    """
    print("\n" + "=" * 60)
    print(f"Demo 1: Find Similar Samples ({embedding_name})")
    print("=" * 60)
    
    # Pick a sample from each disease
    for disease in ['RA', 'OA', 'SLE']:
        mask = metadata['disease_abbrev'] == disease
        query_idx = mask[mask].index[0]
        query_sample = metadata.iloc[query_idx]
        
        print(f"\nQuery: {query_sample['sample_title']} ({query_sample['disease_abbrev']})")
        print("-" * 40)
        
        results = find_similar_samples(query_idx, embeddings, metadata, top_k=5)
        
        for _, row in results.iterrows():
            match_icon = "✓" if row['disease'] == disease else "✗"
            print(f"  {match_icon} {row['sample']}: {row['disease']} (sim: {row['similarity']:.4f})")


def demo_disease_search(metadata, embeddings, embedding_name):
    """
    Demo: Search for samples by disease type.
    """
    print("\n" + "=" * 60)
    print(f"Demo 2: Search by Disease ({embedding_name})")
    print("=" * 60)
    
    for query_disease in ['RA', 'OA', 'SLE']:
        print(f"\nQuery: 'Show me samples similar to {query_disease}'")
        print("-" * 40)
        
        results = find_samples_by_disease_query(
            query_disease, embeddings, metadata, top_k=7
        )
        
        correct = sum(results['disease'] == query_disease)
        print(f"  Results ({correct}/{len(results)} correct):")
        
        for _, row in results.iterrows():
            match_icon = "✓" if row['disease'] == query_disease else "✗"
            print(f"    {match_icon} {row['sample']}: {row['disease']} (sim: {row['similarity']:.4f})")


def demo_cross_modal_comparison(metadata, text_emb, expr_emb):
    """
    Demo: Compare text and expression embeddings.
    
    This shows whether samples that are similar in text description
    are also similar in gene expression.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Cross-Modal Comparison")
    print("=" * 60)
    
    print("\nComparing text similarity vs expression similarity:")
    print("(Do samples with similar descriptions have similar expression?)\n")
    
    # For each sample, find its nearest neighbors in both spaces
    agreements = []
    
    for i in range(len(metadata)):
        # Find top 3 similar in text space
        text_neighbors = find_similar_samples(i, text_emb, metadata, top_k=3)
        text_neighbor_idx = set(text_neighbors['index'].values)
        
        # Find top 3 similar in expression space
        expr_neighbors = find_similar_samples(i, expr_emb, metadata, top_k=3)
        expr_neighbor_idx = set(expr_neighbors['index'].values)
        
        # Calculate overlap
        overlap = len(text_neighbor_idx & expr_neighbor_idx)
        agreements.append(overlap)
    
    avg_agreement = np.mean(agreements)
    
    print(f"Average neighbor overlap: {avg_agreement:.2f} / 3")
    print(f"(3 = perfect agreement, 0 = no agreement)")
    
    # Show specific examples
    print("\nExamples of agreement/disagreement:")
    
    for i in [0, 5, 12]:  # Pick a few samples
        sample = metadata.iloc[i]
        print(f"\n  Sample: {sample['sample_title']} ({sample['disease_abbrev']})")
        
        text_neighbors = find_similar_samples(i, text_emb, metadata, top_k=3)
        expr_neighbors = find_similar_samples(i, expr_emb, metadata, top_k=3)
        
        print(f"    Text neighbors: {list(text_neighbors['sample'].values)}")
        print(f"    Expr neighbors: {list(expr_neighbors['sample'].values)}")


def demo_disease_classification(metadata, embeddings, embedding_name):
    """
    Demo: Simple classification accuracy using embeddings.
    
    For each sample, predict its disease based on nearest neighbor.
    """
    print("\n" + "=" * 60)
    print(f"Demo 4: Disease Classification ({embedding_name})")
    print("=" * 60)
    
    correct = 0
    total = len(metadata)
    
    predictions = []
    
    for i in range(total):
        true_disease = metadata.iloc[i]['disease_abbrev']
        
        # Find nearest neighbor (excluding self)
        results = find_similar_samples(i, embeddings, metadata, top_k=1)
        predicted_disease = results.iloc[0]['disease']
        
        predictions.append({
            'sample': metadata.iloc[i]['sample_title'],
            'true': true_disease,
            'predicted': predicted_disease,
            'correct': true_disease == predicted_disease
        })
        
        if true_disease == predicted_disease:
            correct += 1
    
    accuracy = correct / total * 100
    
    print(f"\nNearest Neighbor Classification Accuracy: {accuracy:.1f}%")
    print(f"({correct}/{total} correct predictions)")
    
    # Show confusion by disease
    print("\nPer-disease accuracy:")
    pred_df = pd.DataFrame(predictions)
    
    for disease in metadata['disease_abbrev'].unique():
        disease_mask = pred_df['true'] == disease
        disease_correct = pred_df[disease_mask]['correct'].sum()
        disease_total = disease_mask.sum()
        print(f"  {disease}: {disease_correct}/{disease_total} ({disease_correct/disease_total*100:.0f}%)")
    
    return pred_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Run all demos.
    """
    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH DEMO")
    print("=" * 60)
    
    try:
        metadata, text_emb, pca_emb, ae_emb = load_data()
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("Please run scripts 01, 02, and 03 first!")
        return
    
    # Demo 1: Find similar samples
    demo_similar_samples(metadata, pca_emb, "PCA")
    
    # Demo 2: Search by disease
    demo_disease_search(metadata, pca_emb, "PCA")
    
    # Demo 3: Cross-modal comparison
    demo_cross_modal_comparison(metadata, text_emb, pca_emb)
    
    # Demo 4: Classification accuracy
    print("\n" + "=" * 60)
    print("Classification Comparison")
    print("=" * 60)
    
    for name, emb in [("Text (BioBERT)", text_emb), 
                       ("PCA", pca_emb), 
                       ("Autoencoder", ae_emb)]:
        demo_disease_classification(metadata, emb, name)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Embeddings capture disease-related information")
    print("  2. Similar samples cluster together in embedding space")
    print("  3. Both text and expression embeddings are informative")
    print("\nThis is the foundation of how CellWhisperer works!")
    print("In practice, CellWhisperer trains these embeddings jointly")
    print("using contrastive learning for better alignment.")


if __name__ == "__main__":
    main()

