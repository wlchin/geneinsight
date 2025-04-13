#!/usr/bin/env python
import os
import glob
import pandas as pd
import numpy as np
import argparse
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import scipy.stats as stats
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_avg_distance(terms, model):
    """
    Embed the given list of terms using the provided model,
    compute pairwise cosine distances, and return the average distance.
    If fewer than 2 terms are provided, returns np.nan.
    """
    if len(terms) < 2:
        return np.nan
    # Get embeddings
    embeddings = model.encode(terms, show_progress_bar=False)
    # Compute pairwise cosine distances (distance = 1 - cosine similarity)
    distances = pairwise_distances(embeddings, metric='cosine')
    # Take the upper-triangle (excluding the diagonal)
    triu_indices = np.triu_indices_from(distances, k=1)
    avg_distance = distances[triu_indices].mean()
    return avg_distance

def main():
    parser = argparse.ArgumentParser(
        description="Compute average cosine distances for gene sets, topics, and enrichment documents using SentenceTransformer embeddings."
    )
    parser.add_argument("--sets_dir", type=str, default="1000geneset_benchmark/results/filtered_sets",
                        help="Directory containing gene sets CSV files.")
    parser.add_argument("--topics_dir", type=str, default="1000geneset_benchmark/results/filtered_topics/samp_75",
                        help="Directory containing topics CSV files.")
    parser.add_argument("--enrichment_dir", type=str, default="1000geneset_benchmark/results/enrichment_df_listmode",
                        help="Directory containing enrichment CSV files.")
    parser.add_argument("--max_gene_sets", type=int, default=1000,
                        help="Maximum number of gene sets to process.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name to use.")
    parser.add_argument("--output_stats_csv", type=str, default="avg_distances_stats_75.csv",
                        help="Output CSV file for statistics results.")
    # New argument to save individual average distances
    parser.add_argument("--output_distances_csv", type=str, default="avg_distances_75.csv",
                        help="Output CSV file for individual average distances.")
    args = parser.parse_args()

    # Define file patterns based on provided directories
    sets_pattern  = os.path.join(args.sets_dir, "*_filtered_gene_sets.csv")
    topics_pattern = os.path.join(args.topics_dir, "*_filtered_topics.csv")
    enrichment_pattern = os.path.join(args.enrichment_dir, "*__documents.csv")

    # Collect filenames using glob
    sets_files  = glob.glob(sets_pattern)
    topics_files = glob.glob(topics_pattern)
    enrichment_files = glob.glob(enrichment_pattern)

    # Build dictionaries mapping gene set name to file path
    sets_dict = {
        os.path.basename(f).replace("_filtered_gene_sets.csv", ""): f 
        for f in sets_files
    }
    topics_dict = {
        os.path.basename(f).replace("_filtered_topics.csv", ""): f 
        for f in topics_files
    }
    enrichment_dict = {
        os.path.basename(f).replace("__documents.csv", ""): f 
        for f in enrichment_files
    }

    # Use the intersection of gene set names that are available in both sets and topics.
    all_gene_sets = sorted(set(sets_dict.keys()).intersection(topics_dict.keys()))
    logger.info(f"Found {len(all_gene_sets)} gene sets in common between sets and topics.")

    # Load the SentenceTransformer model
    logger.info(f"Loading SentenceTransformer model: {args.model_name}")
    st_model = SentenceTransformer(args.model_name)

    # Collect results for each gene set
    results = []
    for gene_set in tqdm(all_gene_sets[:args.max_gene_sets], desc="Processing gene sets"):
        result = {'gene_set': gene_set}
        
        # Process gene sets file
        sets_file = sets_dict.get(gene_set)
        if sets_file and os.path.exists(sets_file):
            df_sets = pd.read_csv(sets_file)
            # Try to use "Term" column, otherwise use "description"
            if 'Term' in df_sets.columns:
                col = 'Term'
            elif 'description' in df_sets.columns:
                col = 'description'
            else:
                logger.warning(f"Neither 'Term' nor 'description' column found in {sets_file}")
                result['sets_avg_distance'] = np.nan
                col = None
            if col is not None:
                terms = df_sets[col].dropna().unique().tolist()
                result['sets_avg_distance'] = compute_avg_distance(terms, st_model)
        else:
            result['sets_avg_distance'] = np.nan
        
        # Process topics file
        topics_file = topics_dict.get(gene_set)
        if topics_file and os.path.exists(topics_file):
            df_topics = pd.read_csv(topics_file)
            if 'Term' in df_topics.columns:
                terms = df_topics['Term'].dropna().unique().tolist()
                result['topics_avg_distance'] = compute_avg_distance(terms, st_model)
            else:
                logger.warning(f"'Term' column not found in {topics_file}")
                result['topics_avg_distance'] = np.nan
        else:
            result['topics_avg_distance'] = np.nan
            
        # Process enrichment file
        enrichment_file = enrichment_dict.get(gene_set)
        if enrichment_file and os.path.exists(enrichment_file):
            df_enrichment = pd.read_csv(enrichment_file)
            # Check for "Term" column, if not found use "description"
            if 'Term' in df_enrichment.columns:
                col = 'Term'
            elif 'description' in df_enrichment.columns:
                col = 'description'
            else:
                logger.warning(f"Neither 'Term' nor 'description' column found in {enrichment_file}")
                result['enrichment_avg_distance'] = np.nan
                col = None
            if col is not None:
                terms = df_enrichment[col].dropna().unique().tolist()
                result['enrichment_avg_distance'] = compute_avg_distance(terms, st_model)
        else:
            result['enrichment_avg_distance'] = np.nan
            
        results.append(result)

    # Create the final dataframe with gene sets as rows
    results_df = pd.DataFrame(results).set_index('gene_set')
    logger.info("Completed computing average distances")
    print(results_df)

    # Save individual average distances to CSV using the new argument
    results_df.to_csv(args.output_distances_csv)
    logger.info(f"Individual average distances saved to {args.output_distances_csv}")

    # Compute paired t-tests for the average distances
    # Use a copy with dropped NA values for statistical tests
    results_df_stats = results_df.dropna()
    cols = results_df_stats.columns.tolist()
    results_stats = []
    # Iterate over all unique pairs of columns
    for col1, col2 in itertools.combinations(cols, 2):
        # Perform a paired t-test
        t_stat, p_val = stats.ttest_rel(results_df_stats[col1], results_df_stats[col2])
        # Calculate the mean difference (col1 - col2)
        mean_diff = results_df_stats[col1].mean() - results_df_stats[col2].mean()
        # Determine which column has the larger mean
        larger = col1 if mean_diff > 0 else col2 if mean_diff < 0 else "Equal"
        results_stats.append({
            "Comparison": f"{col1} vs {col2}",
            "t-statistic": t_stat,
            "p-value": p_val,
            "Mean Difference": mean_diff,
            "Larger": larger
        })
    
    # Create a tidy DataFrame with the statistics results
    stats_results_df = pd.DataFrame(results_stats)
    
    # Display the results
    print(stats_results_df)
    print(stats_results_df.to_markdown())
    
    # Save the stats results to CSV
    stats_results_df.to_csv(args.output_stats_csv)
    logger.info(f"Statistics results saved to {args.output_stats_csv}")

if __name__ == "__main__":
    main()
