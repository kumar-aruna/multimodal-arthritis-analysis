# Multimodal Learning for Arthritis Gene Expression Analysis

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Gene%20Expression-green.svg)](https://www.ncbi.nlm.nih.gov/geo/)

> **CellWhisperer-Inspired Multimodal Embedding Analysis**  
> Aligning text descriptions and gene expression data in a shared embedding space for arthritis subtype classification and analysis.

---

## Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [References](#-references)
- [License](#-license)

---

## Overview

This project implements a **multimodal learning system** that aligns clinical text descriptions with gene expression data for arthritis patients. Inspired by [CellWhisperer](https://www.nature.com/articles/s41587-025-02857-9) (Nature Biotechnology 2025), the system enables cross-modal search and analysis by projecting both modalities into a shared embedding space using contrastive learning.

### Problem Statement

Clinical data exists in two separate worlds:
- **Text descriptions**: "52-year-old male with rheumatoid arthritis, treated with NSAIDs"
- **Gene expression**: 54,675 gene measurements from microarray analysis

These modalities cannot be directly compared, limiting our ability to:
- Search expression data using natural language queries
- Connect clinical descriptions to molecular profiles
- Discover relationships between text and expression patterns

### Solution

We bridge this gap by:
1. Creating embeddings for both modalities (BioBERT for text, PCA/Autoencoder for expression)
2. Aligning them in a shared 256-dimensional space using contrastive learning
3. Achieving **100% cross-modal retrieval accuracy**
4. Discovering biological insights about disease relationships

---

##  Key Features

- ✅ **Multimodal Alignment**: Text and expression embeddings in shared space
- ✅ **100% Cross-Modal Retrieval**: Perfect matching between text and expression
- ✅ **Biological Insights**: Disease relationship analysis and clinical correlations
- ✅ **Multiple Methods**: Comparison of PCA vs Autoencoder for expression embeddings
- ✅ **Interactive Visualizations**: Plotly-based interactive HTML visualizations
- ✅ **Comprehensive Analysis**: End-to-end pipeline from raw data to insights

---

## Dataset

**Source**: [GSE36700](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36700) from NCBI GEO

- **Samples**: 25 synovial biopsies from arthritis patients
- **Platform**: Affymetrix Human Genome U133 Plus 2.0
- **Probes**: ~54,675 gene expression measurements
- **Diseases**:
  - OA (Osteoarthritis): 5 samples
  - RA (Rheumatoid Arthritis): 7 samples
  - SLE (Systemic Lupus Erythematosus): 4 samples
  - MIC (Microcrystalline Arthritis): 5 samples
  - SA (Seronegative Arthritis): 4 samples

**Metadata Available**:
- Disease type
- Patient age (19-73 years)
- Gender (male/female)
- Treatment status (NSAIDs, colchicine, or none)

---

## Project Structure

```
multimodal-arthritis-analysis/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── scripts/                 # Analysis pipeline
│   ├── 01_parse_geo_data.py
│   ├── 02_create_text_embeddings.py
│   ├── 03_create_expression_embeddings.py
│   ├── 04_visualize_embeddings.py
│   ├── 05_similarity_search.py
│   ├── 06_cellwhisperer_style_training.py
│   ├── 06b_compare_pca_vs_autoencoder.py
│   ├── 07_embedding_atlas_visualization.py
│   ├── 08_interactive_plotly_viz.py
│   └── 09_biological_analysis.py
│
├── data/                    # Processed data (gitignored)
│   ├── sample_metadata.csv
│   ├── expression_matrix.csv
│   ├── text_embeddings.npy
│   ├── expression_embeddings_pca.npy
│   ├── expression_embeddings_autoencoder.npy
│   ├── text_embeddings_aligned.npy
│   └── expression_embeddings_aligned.npy
│
├── results/                 # Output visualizations
│   ├── embedding_comparison_*.png
│   ├── aligned_embeddings.png
│   ├── finding*.png
│   ├── interactive_*.html
│   └── biological_analysis_report.md
│
└── docs/                   # Documentation (see docs/README.md)
    ├── COMPREHENSIVE_PROJECT_HANDBOOK.md
    ├── STORY_STYLE_HANDBOOK.md
    └── ...
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-arthritis-analysis.git
   cd multimodal-arthritis-analysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download `GSE36700_series_matrix.txt` from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36700)
   - Place it in the project root directory

---

## 💻 Usage

### Running the Complete Pipeline

Run scripts in numerical order:

```bash
# Step 1: Parse GEO data
python scripts/01_parse_geo_data.py

# Step 2: Create text embeddings (BioBERT)
python scripts/02_create_text_embeddings.py

# Step 3: Create expression embeddings (PCA & Autoencoder)
python scripts/03_create_expression_embeddings.py

# Step 4: Visualize embeddings (before alignment)
python scripts/04_visualize_embeddings.py

# Step 5: Similarity search demonstration
python scripts/05_similarity_search.py

# Step 6: Align embeddings (CellWhisperer-style)
python scripts/06_cellwhisperer_style_training.py

# Step 6b: Compare PCA vs Autoencoder
python scripts/06b_compare_pca_vs_autoencoder.py

# Step 7: Prepare for Embedding Atlas
python scripts/07_embedding_atlas_visualization.py

# Step 8: Create interactive visualizations
python scripts/08_interactive_plotly_viz.py

# Step 9: Biological analysis
python scripts/09_biological_analysis.py
```

### Quick Start (Minimal Pipeline)

For a quick demonstration, run only the essential steps:

```bash
python scripts/01_parse_geo_data.py
python scripts/02_create_text_embeddings.py
python scripts/03_create_expression_embeddings.py
python scripts/06_cellwhisperer_style_training.py
python scripts/09_biological_analysis.py
```

### Viewing Results

- **Static visualizations**: Check `results/*.png`
- **Interactive HTML**: Open `results/interactive_*.html` in your browser
- **Biological report**: See `results/biological_analysis_report.md`

---

## Results

### Alignment Performance

- ✅ **100% cross-modal retrieval accuracy**
- ✅ **122% improvement** in text embedding clustering after alignment
- ✅ **91% improvement** in expression embedding clustering after alignment

### Key Visualizations

| Visualization | Description |
|--------------|-------------|
| `aligned_embeddings.png` | Training loss, aligned embeddings, and similarity heatmap |
| `finding1_disease_relationships.png` | Disease similarity heatmap and hierarchical clustering |
| `finding2_alignment_quality.png` | Alignment quality by disease type |
| `finding3_clinical_correlations.png` | Age, sex, and treatment correlations |
| `finding4_method_comparison.png` | Clustering quality comparison across methods |
| `interactive_aligned_embeddings.html` | Interactive Plotly visualization |

### Interactive Visualizations

Open in your browser:
- `results/interactive_aligned_embeddings.html` - Main alignment visualization
- `results/interactive_method_comparison.html` - Method comparison
- `results/interactive_dashboard.html` - Complete dashboard

---

## 🔬 Methodology

### 1. Text Embeddings

- **Model**: BioBERT (`dmis-lab/biobert-v1.1`)
- **Dimensions**: 768
- **Input**: Natural language descriptions of samples
- **Output**: Dense vector representations capturing semantic meaning

### 2. Expression Embeddings

Two methods compared:

**PCA (Principal Component Analysis)**
- Linear dimensionality reduction
- 24 dimensions (limited by sample size)
- Fast and interpretable

**Autoencoder**
- Non-linear neural network compression
- 128 dimensions
- Can capture complex patterns

### 3. Alignment

- **Method**: Contrastive learning with CLIP loss
- **Architecture**: Dual encoder with projection heads
- **Shared Space**: 256 dimensions
- **Training**: 200 epochs with Adam optimizer

### 4. Evaluation

- Cross-modal retrieval accuracy
- Silhouette scores for clustering quality
- Disease relationship analysis
- Clinical correlation analysis

---

## 🔍 Key Findings

### 1. Disease Relationships

- **Most similar**: RA & SLE (distance: 0.744)
  - Both autoimmune diseases
  - May share molecular pathways
  - Potential for drug repurposing

- **Most different**: SLE & SA (distance: 0.951)
  - Distinct disease mechanisms
  - Require different treatment approaches

### 2. Alignment Quality

- **Best aligned**: SLE (0.4851)
  - Clinical description accurately reflects molecular profile
  - Consistent disease presentation

- **Worst aligned**: MIC (0.5053)
  - More heterogeneity between text and expression
  - May indicate disease subtypes

### 3. Clinical Correlations

- **Disease type** is the primary driver of expression patterns
- **Age** shows weak correlation (r=-0.245, not significant)
- **Sex** has some influence but disease type dominates

### 4. Method Validation

- Alignment significantly improves disease clustering
- Text (Aligned) achieves best Silhouette Score (0.078)
- Multimodal approach more effective than single-modality

---

## 🛠️ Technologies Used

### Core Libraries

- **PyTorch**: Neural network implementation (Autoencoder, alignment model)
- **Transformers**: BioBERT for text embeddings
- **scikit-learn**: PCA, t-SNE, UMAP, metrics
- **NumPy/Pandas**: Data processing and manipulation
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations

### Models & Tools

- **BioBERT**: Pre-trained biomedical language model
- **CLIP Loss**: Contrastive learning loss function
- **Embedding Atlas**: Interactive visualization tool (optional)

---

## References

### Papers

1. **CellWhisperer**: [Nature Biotechnology 2025](https://www.nature.com/articles/s41587-025-02857-9)
   - Original multimodal learning approach for single-cell data

2. **BioBERT**: [Lee et al., 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
   - Biomedical language representation model

3. **CLIP**: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
   - Contrastive Language-Image Pre-training

### Datasets

- **GSE36700**: [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36700)
  - Arthritis synovial biopsy gene expression data

### Tools

- **Embedding Atlas**: [Apple GitHub](https://github.com/apple/embedding-atlas)
  - Interactive embedding visualization


##  Author

**Aruna**

- Project: Multimodal Learning for Arthritis Analysis
- Inspired by: CellWhisperer (Nature Biotechnology 2025)
- Date: January 2025



