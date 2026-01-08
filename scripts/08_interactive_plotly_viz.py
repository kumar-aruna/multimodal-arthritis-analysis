"""
Script 8: Interactive Plotly Visualization
==========================================

An alternative to Embedding Atlas using Plotly for interactive
embedding visualization in your browser.

Author: Aruna (Bioinformatics Project)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def create_interactive_visualization():
    """Create interactive Plotly visualization."""
    print("=" * 60)
    print("Creating Interactive Plotly Visualization")
    print("=" * 60)
    
    # Load the aligned embeddings
    df_aligned = pd.read_parquet(RESULTS_DIR / "atlas_aligned_embeddings.parquet")
    print(f"✓ Loaded aligned embeddings: {len(df_aligned)} points")
    
    # ========================================================================
    # Visualization 1: Aligned Embeddings (Text vs Expression)
    # ========================================================================
    print("\nCreating aligned embeddings visualization...")
    
    fig1 = px.scatter(
        df_aligned,
        x='x',
        y='y',
        color='disease',
        symbol='modality',
        hover_data=['sample_id', 'disease_full', 'age', 'sex', 'modality'],
        title='🔬 CellWhisperer-Style Aligned Embeddings<br><sup>Circles=Text, Diamonds=Expression | Same color pairs should be close!</sup>',
        color_discrete_map={
            'OA': '#1f77b4',
            'RA': '#d62728',
            'SLE': '#9467bd',
            'MIC': '#2ca02c',
            'SA': '#ff7f0e'
        },
        symbol_map={
            'Text': 'circle',
            'Expression': 'diamond'
        }
    )
    
    fig1.update_traces(marker=dict(size=15, line=dict(width=1, color='white')))
    fig1.update_layout(
        width=900,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Draw lines connecting text and expression for same sample
    n_samples = len(df_aligned) // 2
    for i in range(n_samples):
        text_point = df_aligned.iloc[i]
        expr_point = df_aligned.iloc[i + n_samples]
        fig1.add_trace(go.Scatter(
            x=[text_point['x'], expr_point['x']],
            y=[text_point['y'], expr_point['y']],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig1.write_html(RESULTS_DIR / "interactive_aligned_embeddings.html")
    print(f"  ✓ Saved: interactive_aligned_embeddings.html")
    
    # ========================================================================
    # Visualization 2: All Embedding Methods Comparison
    # ========================================================================
    print("\nCreating method comparison visualization...")
    
    df_all = pd.read_parquet(RESULTS_DIR / "atlas_all_embeddings.parquet")
    
    fig2 = px.scatter(
        df_all,
        x='x',
        y='y',
        color='disease',
        facet_col='method',
        hover_data=['sample_id', 'age', 'sex'],
        title='📊 Embedding Methods Comparison',
        color_discrete_map={
            'OA': '#1f77b4',
            'RA': '#d62728',
            'SLE': '#9467bd',
            'MIC': '#2ca02c',
            'SA': '#ff7f0e'
        }
    )
    
    fig2.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
    fig2.update_layout(width=1200, height=500)
    
    fig2.write_html(RESULTS_DIR / "interactive_method_comparison.html")
    print(f"  ✓ Saved: interactive_method_comparison.html")
    
    # ========================================================================
    # Visualization 3: Disease Distribution Dashboard
    # ========================================================================
    print("\nCreating disease dashboard...")
    
    # Create subplot figure
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Aligned Embeddings', 
            'Samples by Disease', 
            'Age Distribution',
            'Treatment Status'
        ],
        specs=[
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "box"}, {"type": "bar"}]
        ]
    )
    
    # Plot 1: Scatter
    for disease in df_aligned['disease'].unique():
        mask = (df_aligned['disease'] == disease) & (df_aligned['modality'] == 'Expression')
        subset = df_aligned[mask]
        fig3.add_trace(
            go.Scatter(
                x=subset['x'],
                y=subset['y'],
                mode='markers',
                name=disease,
                marker=dict(size=12),
                text=subset['sample_id']
            ),
            row=1, col=1
        )
    
    # Plot 2: Pie chart
    disease_counts = df_aligned[df_aligned['modality'] == 'Expression']['disease'].value_counts()
    fig3.add_trace(
        go.Pie(
            labels=disease_counts.index,
            values=disease_counts.values,
            name="Disease"
        ),
        row=1, col=2
    )
    
    # Plot 3: Age by disease
    expr_data = df_aligned[df_aligned['modality'] == 'Expression']
    for disease in expr_data['disease'].unique():
        fig3.add_trace(
            go.Box(
                y=expr_data[expr_data['disease'] == disease]['age'],
                name=disease,
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Plot 4: Sex distribution
    sex_counts = expr_data.groupby(['disease', 'sex']).size().unstack(fill_value=0)
    for sex in sex_counts.columns:
        fig3.add_trace(
            go.Bar(
                x=sex_counts.index,
                y=sex_counts[sex],
                name=f'{sex}',
                showlegend=True
            ),
            row=2, col=2
        )
    
    fig3.update_layout(
        height=800,
        width=1000,
        title_text="🧬 Arthritis Dataset Dashboard"
    )
    
    fig3.write_html(RESULTS_DIR / "interactive_dashboard.html")
    print(f"  ✓ Saved: interactive_dashboard.html")
    
    return fig1, fig2, fig3


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("INTERACTIVE VISUALIZATION")
    print("=" * 60)
    
    create_interactive_visualization()
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"""
Files created in {RESULTS_DIR}:

1. interactive_aligned_embeddings.html
   → Shows text and expression in the SAME space (CellWhisperer-style!)
   → Lines connect matching text-expression pairs

2. interactive_method_comparison.html
   → Compare BioBERT, PCA, and Autoencoder side by side

3. interactive_dashboard.html
   → Full dashboard with disease distribution, age, sex

Open any HTML file in your browser for interactive exploration!

Example:
  open results/interactive_aligned_embeddings.html
""")


if __name__ == "__main__":
    main()

