#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse

from sentence_transformers import SentenceTransformer, util

###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

###############################################################################
# Utility functions for Cosine Similarity
###############################################################################
def compute_cosine_matrix(texts1, texts2, model):
    """
    Computes an M x N cosine similarity matrix between texts1 (size M) and texts2 (size N)
    using a SentenceTransformer.
    """
    logger.info(f"Encoding {len(texts1)} texts for the first set and {len(texts2)} texts for the second set...")
    emb1 = model.encode(texts1, convert_to_tensor=True)
    emb2 = model.encode(texts2, convert_to_tensor=True)
    logger.info("Calculating cosine similarity matrix...")
    cosine_sim_matrix = util.pytorch_cos_sim(emb1, emb2).cpu().numpy()
    return cosine_sim_matrix

def average_topk_from_matrix(cosine_matrix, top_k=1):
    """
    Given a precomputed cosine similarity matrix, for each column, select the top-k values,
    compute their average, and return the average of these values across all columns.
    """
    topk_vals_for_each = []
    for j in range(cosine_matrix.shape[1]):
        col = cosine_matrix[:, j]
        # Get indices of the top_k highest similarities in the column
        top_k_indices = np.argpartition(col, -top_k)[-top_k:]
        topk_vals = col[top_k_indices]
        avg_topk = np.mean(topk_vals)
        topk_vals_for_each.append(avg_topk)
    return float(np.mean(topk_vals_for_each))

###############################################################################
# Main processing
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Compute and plot cosine similarity metrics.")
    parser.add_argument("--sets_dir", type=str, default="1000geneset_benchmark/results/filtered_sets",
                        help="Directory containing gene set CSV files.")
    parser.add_argument("--topics_dir", type=str, default="1000geneset_benchmark/results/filtered_topics/samp_25",
                        help="Directory containing topic CSV files.")
    parser.add_argument("--scores_dir", type=str, default="scores_output_25",
                        help="Directory to save individual cosine score CSV files.")
    parser.add_argument("--results_csv", type=str, default="mover_score_results_25.csv",
                        help="CSV file to save aggregated results.")
    parser.add_argument("--plot_file", type=str, default="mover_score_scatter_25.png",
                        help="File name to save the scatter plot.")
    parser.add_argument("--num_gene_sets", type=int, default=10,
                        help="Number of gene sets to process (default: 10)")
    args = parser.parse_args()

    sets_dir = args.sets_dir
    topics_dir = args.topics_dir
    scores_dir = args.scores_dir

    # Patterns for CSV files
    sets_pattern = os.path.join(sets_dir, "*_filtered_gene_sets.csv")
    topics_pattern = os.path.join(topics_dir, "*_filtered_topics.csv")

    # Collect filenames
    sets_files = glob.glob(sets_pattern)
    topics_files = glob.glob(topics_pattern)

    # Build dictionaries mapping gene set names to file paths
    sets_dict = {}
    for f in sets_files:
        base = os.path.basename(f)
        gene_set_name = base.replace("_filtered_gene_sets.csv", "")
        sets_dict[gene_set_name] = f

    topics_dict = {}
    for f in topics_files:
        base = os.path.basename(f)
        gene_set_name = base.replace("_filtered_topics.csv", "")
        topics_dict[gene_set_name] = f

    # Process only gene sets that exist in both directories
    all_gene_sets = sorted(set(sets_dict.keys()).intersection(topics_dict.keys()))
    if not all_gene_sets:
        logger.warning("No matching gene set files found. Exiting.")
        return

    # Prepare a directory to store individual cosine similarity CSVs
    os.makedirs(scores_dir, exist_ok=True)

    # Define topK values for evaluation
    topK_values = [2, 5, 10, 25, 50]

    # Load SentenceTransformer model for cosine similarity
    model_name = 'all-MiniLM-L6-v2'
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    st_model = SentenceTransformer(model_name)

    results = []

    # Use the --num_gene_sets argument to determine how many gene sets to process.
    for gene_set_name in all_gene_sets[:args.num_gene_sets]:
        logger.info(f"Processing gene set: {gene_set_name}")

        set_file = sets_dict[gene_set_name]
        topic_file = topics_dict[gene_set_name]

        df_sets = pd.read_csv(set_file)
        df_topics = pd.read_csv(topic_file)

        source_list = df_sets["Term"].tolist()   # full gene set (source)
        summary_list = df_topics["Term"].tolist()  # reduced set (topics)

        # Compute compression ratio
        compression_ratio = len(source_list) / float(len(summary_list)) if summary_list else 0.0

        # Check if the cosine similarity file already exists
        cosine_outfile = os.path.join(scores_dir, f"{gene_set_name}_cosine_scores.csv")
        if os.path.exists(cosine_outfile):
            logger.info(f"Cosine score file exists for {gene_set_name}, loading it instead of recomputing.")
            # Load existing cosine matrix
            cosine_df = pd.read_csv(cosine_outfile, index_col=0)
            cosine_matrix = cosine_df.values
        else:
            # Compute cosine similarity matrix
            logger.info(f"Calculating cosine similarity matrix for gene set: {gene_set_name}")
            cosine_matrix = compute_cosine_matrix(source_list, summary_list, st_model)

            # Save the cosine similarity matrix to CSV
            cosine_df = pd.DataFrame(cosine_matrix, index=source_list, columns=summary_list)
            cosine_df.to_csv(cosine_outfile)
            logger.info(f"Saved cosine similarity matrix to: {cosine_outfile}")

        # Compute top-k recall for each top_k value using the matrix (either loaded or newly computed)
        gene_set_results = []
        for k in topK_values:
            recall_cosine = average_topk_from_matrix(cosine_matrix, top_k=k)
            logger.info(f"[{gene_set_name}] topK={k} | cosine similarity recall={recall_cosine:.4f}")

            result_entry = {
                "gene_set": gene_set_name,
                "top_k": k,
                "compression_ratio": compression_ratio,
                "recall_cosine": recall_cosine,
                "source_length": len(source_list),
                "summary_length": len(summary_list)
            }
            results.append(result_entry)
            gene_set_results.append(result_entry)
            
        # Create and save a DataFrame with all top-k values for this gene set
        topk_df = pd.DataFrame(gene_set_results)
        topk_outfile = os.path.join(scores_dir, f"{gene_set_name}_topk.csv")
        topk_df.to_csv(topk_outfile, index=False)
        logger.info(f"Saved top-k results for {gene_set_name} to: {topk_outfile}")

    # Aggregate results into a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.results_csv, index=False)
    logger.info(f"Aggregated results saved to: {args.results_csv}")

    # Plotting: Scatter plot of Top-k Cosine Similarity vs. Source Document Size
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results_df,
        x="source_length",
        y="recall_cosine",
        style="top_k",
        s=100
    )
    plt.title("Top-k Cosine Similarity vs. Source Document Size")
    plt.xlabel("Source Document Size (number of terms)")
    plt.ylabel("Top-k Cosine Similarity")
    plt.legend(title="top_k")
    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=300, bbox_inches="tight")
    logger.info(f"Scatter plot saved to: {args.plot_file}")

if __name__ == "__main__":
    main()