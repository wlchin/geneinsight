import pandas as pd
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import colorcet as cc
import argparse

def plot_headings_plotly(df, output_html, heading_map, embedding_model='all-MiniLM-L6-v2',
                         n_components=2, extra_terms=None):
    """
    Plots the positions of the headings in a 2D UMAP embedding space using Plotly.
    
    The UMAP reducer is fit on the embeddings of the main terms (from df['Term'])
    combined with any additional terms provided. The same reducer is then used
    to transform the heading strings (from heading_map) into the UMAP space.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with 'Term' and 'Cluster' columns.
        output_html (str): Path to the output HTML file.
        heading_map (dict): Dictionary mapping cluster IDs to heading strings.
        embedding_model (str): Sentence transformer model to use for embedding text.
        n_components (int): Number of dimensions for UMAP reduction.
        extra_terms (list of str, optional): Additional terms (from extra CSV) to include in UMAP.
    
    Returns:
        None
    """
    # Step 1: Embed the main terms.
    model = SentenceTransformer(embedding_model)
    main_terms = df['Term'].tolist()
    main_embeddings = model.encode(main_terms)

    # If extra terms are provided, encode them and combine with the main embeddings.
    if extra_terms is not None:
        extra_embeddings = model.encode(extra_terms)
        combined_embeddings = np.concatenate([main_embeddings, extra_embeddings], axis=0)
    else:
        combined_embeddings = main_embeddings

    # Step 2: Fit UMAP on the combined embeddings.
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    combined_umap = reducer.fit_transform(combined_embeddings)
    # (If needed, you can extract main_umap = combined_umap[:len(main_embeddings)])

    # (Optional) Compute subtopic counts per cluster for display.
    theme_sizes = df['Cluster'].value_counts()

    # Step 3: Compute the UMAP coordinates for each heading.
    # Assume that heading_map's keys correspond to cluster IDs.
    sorted_cluster_ids = sorted(heading_map.keys())
    headings_list = [heading_map[cid] for cid in sorted_cluster_ids]
    heading_embeddings = model.encode(headings_list)
    heading_umap = reducer.transform(heading_embeddings)

    # Step 4: Create Plotly markers for each heading.
    markers = []
    marker_size = 20
    colors = cc.glasbey_dark[:len(sorted_cluster_ids)]

    for i, cid in enumerate(sorted_cluster_ids):
        x, y = heading_umap[i]
        # Retrieve the number of subtopics for this cluster (if available).
        size = theme_sizes.get(cid, "N/A")
        heading_str = heading_map[cid]
        markers.append(
            go.Scatter(
                x=[x],
                y=[y],
                marker=dict(
                    size=marker_size,
                    color=colors[i],
                    line=dict(width=2, color='black')
                ),
                name=f"Theme {cid + 1} - {heading_str}",
                text=f"Theme {cid + 1} - {heading_str}<br>Subtopics: {size}",
                textposition="top center",
                hovertemplate=f"Theme {cid + 1} - {heading_str}<br>Subtopics: {size}<extra></extra>"
            )
        )

    layout = go.Layout(
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
    fig.write_html(output_html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heading positions in a 2D UMAP space.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input CSV file containing 'Term' and 'Cluster' columns.")
    parser.add_argument("--headings_csv", type=str, required=True,
                        help="Path to the CSV file with cluster headings (columns: 'cluster' and 'heading').")
    parser.add_argument("--output_html", type=str, required=True,
                        help="Path to the output HTML file.")
    # Optional extra vectors CSV (with a "Term" column containing additional terms).
    parser.add_argument("--extra_vectors_csv", type=str, required=False,
                        help="Path to the CSV file containing additional terms in the 'Term' column to include in UMAP.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    headings_df = pd.read_csv(args.headings_csv)
    heading_map = headings_df.set_index("cluster")["heading"].to_dict()

    # Load extra terms if the CSV is provided.
    extra_terms = None
    if args.extra_vectors_csv:
        extra_df = pd.read_csv(args.extra_vectors_csv)
        extra_terms = extra_df["Term"].tolist()

    plot_headings_plotly(df, args.output_html, heading_map, extra_terms=extra_terms)
