"""
Script 1: Parse GEO Series Matrix File
======================================

This script parses the GSE36700 series matrix file and extracts:
1. Sample metadata (disease, age, gender, treatment)
2. Gene expression matrix
3. Creates text descriptions for each sample

Author: Aruna (Bioinformatics Project)
Dataset: GSE36700 - Arthritis synovial biopsies
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
INPUT_FILE = PROJECT_DIR / "GSE36700_series_matrix.txt"

# Create output directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)


# ============================================================================
# STEP 1: PARSE METADATA
# ============================================================================

def parse_metadata(filepath):
    """
    Extract sample metadata from the series matrix file.
    
    The metadata lines start with '!' and contain information about:
    - Sample titles (e.g., "OA1", "RA2")
    - Disease type
    - Patient age
    - Patient gender
    - Treatment information
    
    Returns:
        pd.DataFrame: Metadata for each sample
    """
    print("=" * 60)
    print("STEP 1: Parsing Metadata")
    print("=" * 60)
    
    metadata = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Stop when we reach the expression data
            if line.startswith('!series_matrix_table_begin'):
                break
            
            # Parse relevant metadata fields
            if line.startswith('!Sample_title'):
                values = line.split('\t')[1:]
                metadata['sample_title'] = [v.strip('"') for v in values]
                
            elif line.startswith('!Sample_geo_accession'):
                values = line.split('\t')[1:]
                metadata['geo_accession'] = [v.strip('"') for v in values]
                
            elif line.startswith('!Sample_source_name_ch1'):
                values = line.split('\t')[1:]
                metadata['source_description'] = [v.strip('"') for v in values]
                
            elif line.startswith('!Sample_characteristics_ch1'):
                values = line.split('\t')[1:]
                values = [v.strip('"') for v in values]
                
                # Determine what type of characteristic this is
                if values and ':' in values[0]:
                    char_type = values[0].split(':')[0].strip()
                    char_values = [v.split(':')[1].strip() if ':' in v else v for v in values]
                    
                    if char_type == 'tissue':
                        metadata['tissue'] = char_values
                    elif char_type == 'disease':
                        metadata['disease'] = char_values
                    elif char_type == 'age':
                        metadata['age'] = [int(v) if v.isdigit() else None for v in char_values]
                    elif char_type == 'gender':
                        metadata['gender'] = char_values
                    elif char_type == 'treatment':
                        metadata['treatment'] = char_values
    
    # Create DataFrame
    df = pd.DataFrame(metadata)
    
    # Add disease abbreviation for easier grouping
    disease_map = {
        'Osteoarthritis': 'OA',
        'Rheumatoid arthritis': 'RA',
        'Systemic lupus erythematosus': 'SLE',
        'Microcrystalline arthritis': 'MIC',
        'Seronegative arthritis': 'SA'
    }
    df['disease_abbrev'] = df['disease'].map(disease_map)
    
    print(f"\n✓ Extracted metadata for {len(df)} samples")
    print(f"\nSample distribution by disease:")
    print(df['disease_abbrev'].value_counts())
    
    return df


# ============================================================================
# STEP 2: PARSE EXPRESSION DATA
# ============================================================================

def parse_expression_data(filepath):
    """
    Extract gene expression matrix from the series matrix file.
    
    The expression data is located between:
    - !series_matrix_table_begin
    - !series_matrix_table_end
    
    Returns:
        pd.DataFrame: Expression matrix (genes × samples)
    """
    print("\n" + "=" * 60)
    print("STEP 2: Parsing Expression Data")
    print("=" * 60)
    
    # Find the start and end of expression data
    data_lines = []
    in_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('!series_matrix_table_begin'):
                in_data = True
                continue
            
            if line.startswith('!series_matrix_table_end'):
                break
            
            if in_data:
                data_lines.append(line)
    
    # Parse header (column names = sample IDs)
    header = data_lines[0].split('\t')
    header = [h.strip('"') for h in header]
    
    # Parse expression values
    expression_data = []
    probe_ids = []
    
    for line in data_lines[1:]:
        parts = line.split('\t')
        probe_id = parts[0].strip('"')
        values = [float(v) for v in parts[1:]]
        
        probe_ids.append(probe_id)
        expression_data.append(values)
    
    # Create DataFrame
    df = pd.DataFrame(
        expression_data,
        index=probe_ids,
        columns=header[1:]  # Skip "ID_REF" column name
    )
    
    print(f"\n✓ Extracted expression matrix:")
    print(f"  - Probe IDs (genes): {len(df)}")
    print(f"  - Samples: {len(df.columns)}")
    print(f"\nExpression value statistics:")
    print(f"  - Min: {df.values.min():.2f}")
    print(f"  - Max: {df.values.max():.2f}")
    print(f"  - Mean: {df.values.mean():.2f}")
    
    return df


# ============================================================================
# STEP 3: CREATE TEXT DESCRIPTIONS
# ============================================================================

def create_text_descriptions(metadata_df):
    """
    Create natural language descriptions for each sample.
    
    This is similar to what CellWhisperer does - creating textual
    annotations that describe each sample's biological context.
    
    Example output:
    "Synovial biopsy sample from a patient with rheumatoid arthritis.
     Patient is a 52-year-old male, treated with NSAIDs."
    
    Returns:
        pd.Series: Text descriptions for each sample
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Text Descriptions")
    print("=" * 60)
    
    descriptions = []
    
    for _, row in metadata_df.iterrows():
        # Build description
        parts = []
        
        # Disease information
        parts.append(f"Synovial biopsy sample from a patient with {row['disease'].lower()}")
        
        # Patient demographics
        gender_word = "male" if row['gender'] == 'm' else "female"
        parts.append(f"Patient is a {row['age']}-year-old {gender_word}")
        
        # Treatment information
        if row['treatment'] and row['treatment'] != '-':
            parts.append(f"treated with {row['treatment']}")
        else:
            parts.append("with no current treatment")
        
        # Combine into full description
        description = ". ".join(parts) + "."
        descriptions.append(description)
    
    metadata_df['text_description'] = descriptions
    
    print("\n✓ Created text descriptions")
    print("\nExample descriptions:")
    for i in range(min(3, len(descriptions))):
        print(f"\n  Sample {metadata_df.iloc[i]['sample_title']}:")
        print(f"  {descriptions[i]}")
    
    return metadata_df


# ============================================================================
# STEP 4: SAVE PROCESSED DATA
# ============================================================================

def save_data(metadata_df, expression_df):
    """
    Save processed data to CSV files.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Saving Processed Data")
    print("=" * 60)
    
    # Save metadata
    metadata_path = DATA_DIR / "sample_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n✓ Saved metadata to: {metadata_path}")
    
    # Save expression matrix
    expression_path = DATA_DIR / "expression_matrix.csv"
    expression_df.to_csv(expression_path)
    print(f"✓ Saved expression matrix to: {expression_path}")
    
    # Also save a transposed version (samples × genes) for easier embedding
    expression_T_path = DATA_DIR / "expression_matrix_transposed.csv"
    expression_df.T.to_csv(expression_T_path)
    print(f"✓ Saved transposed matrix to: {expression_T_path}")
    
    return metadata_path, expression_path


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the complete parsing pipeline.
    """
    print("\n" + "=" * 60)
    print("GEO DATA PARSING PIPELINE")
    print("Dataset: GSE36700 - Arthritis Gene Expression")
    print("=" * 60)
    
    # Check if input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    print(f"\nInput file: {INPUT_FILE}")
    
    # Step 1: Parse metadata
    metadata_df = parse_metadata(INPUT_FILE)
    
    # Step 2: Parse expression data
    expression_df = parse_expression_data(INPUT_FILE)
    
    # Step 3: Create text descriptions
    metadata_df = create_text_descriptions(metadata_df)
    
    # Step 4: Save data
    save_data(metadata_df, expression_df)
    
    print("\n" + "=" * 60)
    print("PARSING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run 02_create_text_embeddings.py to create BioBERT embeddings")
    print("  2. Run 03_create_expression_embeddings.py for gene embeddings")
    print("  3. Run 04_visualize_embeddings.py to visualize results")
    
    return metadata_df, expression_df


if __name__ == "__main__":
    metadata, expression = main()

