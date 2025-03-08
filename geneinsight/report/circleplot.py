"""
Module for generating interactive circle plots using UMAP embeddings.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def generate_circle_plot(
    input_csv: str,
    output_html: str,
    headings_csv: Optional[str] = None,
    embedding_model: str = 'all-MiniLM-L6-v2',
    n_components: int = 2,
    extra_vectors_csv: Optional[str] = None
) -> None:
    """
    Generate an interactive circle plot of topic embeddings using UMAP and Plotly.
    
    Args:
        input_csv: Path to input CSV with 'Term' and 'Cluster' columns
        output_html: Path to save the output HTML visualization
        headings_csv: Path to CSV with cluster headings (columns: 'cluster' and 'heading')
        embedding_model: Name of the sentence transformer model to use
        n_components: Number of UMAP dimensions (2 or 3)
        extra_vectors_csv: Path to CSV with additional terms to include in UMAP (optional)
    """
    try:
        # Create the directory for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        
        # Import libraries here to make them optional dependencies
        try:
            import umap
            from sentence_transformers import SentenceTransformer
            import plotly.graph_objects as go
            import colorcet as cc
        except ImportError as e:
            logger.error(f"Required package not found: {e}")
            logger.error("Please install the required packages: pip install umap-learn sentence-transformers plotly colorcet")
            raise
        
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Load headings if available
        heading_map = {}
        if headings_csv and os.path.exists(headings_csv):
            logger.info(f"Loading headings from {headings_csv}")
            try:
                headings_df = pd.read_csv(headings_csv)
                if 'cluster' in headings_df.columns and 'heading' in headings_df.columns:
                    heading_map = headings_df.set_index("cluster")["heading"].to_dict()
                else:
                    logger.warning("Headings CSV does not have required columns 'cluster' and 'heading'")
            except Exception as e:
                logger.warning(f"Error loading headings: {e}")
        
        # If we don't have proper headings, use the terms as headings
        if not heading_map and 'Cluster' in df.columns and 'Term' in df.columns:
            # Create headings from terms, taking the first term for each cluster
            cluster_terms = df.groupby('Cluster')['Term'].first()
            heading_map = {cluster: term for cluster, term in cluster_terms.items()}
        
        # Load extra terms if available
        extra_terms = None
        if extra_vectors_csv and os.path.exists(extra_vectors_csv):
            logger.info(f"Loading extra terms from {extra_vectors_csv}")
            try:
                extra_df = pd.read_csv(extra_vectors_csv)
                if 'Term' in extra_df.columns:
                    extra_terms = extra_df["Term"].tolist()
                else:
                    logger.warning("Extra vectors CSV does not have required column 'Term'")
            except Exception as e:
                logger.warning(f"Error loading extra terms: {e}")
        
        # Step 1: Embed the main terms
        logger.info(f"Loading sentence transformer model: {embedding_model}")
        model = SentenceTransformer(embedding_model)
        
        if 'Term' not in df.columns:
            logger.error("Input CSV does not have required column 'Term'")
            raise ValueError("Input CSV must have a 'Term' column")
        
        main_terms = df['Term'].tolist()
        logger.info(f"Embedding {len(main_terms)} main terms")
        main_embeddings = model.encode(main_terms)

        # If extra terms are provided, encode them and combine with the main embeddings
        if extra_terms:
            logger.info(f"Embedding {len(extra_terms)} extra terms")
            extra_embeddings = model.encode(extra_terms)
            combined_embeddings = np.concatenate([main_embeddings, extra_embeddings], axis=0)
        else:
            combined_embeddings = main_embeddings

        # Step 2: Fit UMAP on the combined embeddings
        logger.info(f"Fitting UMAP with {n_components} components")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        combined_umap = reducer.fit_transform(combined_embeddings)

        # (Optional) Compute subtopic counts per cluster for display
        if 'Cluster' in df.columns:
            theme_sizes = df['Cluster'].value_counts()
        else:
            theme_sizes = {}

        # Step 3: Compute the UMAP coordinates for each heading
        logger.info("Computing UMAP coordinates for headings")
        sorted_cluster_ids = sorted(heading_map.keys())
        headings_list = [heading_map[cid] for cid in sorted_cluster_ids]
        heading_embeddings = model.encode(headings_list)
        heading_umap = reducer.transform(heading_embeddings)

        # Step 4: Create Plotly markers for each heading
        logger.info("Creating Plotly visualization")
        markers = []
        marker_size = 20
        colors = cc.glasbey_dark[:len(sorted_cluster_ids)]

        for i, cid in enumerate(sorted_cluster_ids):
            x, y = heading_umap[i]
            # Retrieve the number of subtopics for this cluster (if available)
            size = theme_sizes.get(cid, "N/A")
            heading_str = heading_map[cid]
            markers.append(
                go.Scatter(
                    x=[x],
                    y=[y],
                    marker=dict(
                        size=marker_size,
                        color=colors[i % len(colors)],
                        line=dict(width=2, color='black')
                    ),
                    name=f"Theme {cid + 1} - {heading_str}",
                    text=f"Theme {cid + 1} - {heading_str}<br>Subtopics: {size}",
                    textposition="top center",
                    hovertemplate=f"Theme {cid + 1} - {heading_str}<br>Subtopics: {size}<extra></extra>"
                )
            )

        layout = go.Layout(
            title="Gene Set Topic Map",
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            showlegend=True,
            height=900,
            width=700,
            legend=dict(
                title="",
                orientation="h",
                x=0.5,
                y=-0.2,  # Adjust vertical position if needed
                xanchor="center",
                font=dict(size=12),
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                bordercolor="black",
                borderwidth=2
            ),
            margin=dict(b=100)
        )

        fig = go.Figure(data=markers, layout=layout)
        
        # Save the figure to HTML
        logger.info(f"Saving circle plot to {output_html}")
        fig.write_html(output_html)
        
        logger.info("Circle plot generation complete")
        
    except Exception as e:
        logger.error(f"Error generating circle plot: {e}")
        import traceback
        traceback.print_exc()