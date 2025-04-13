#!/usr/bin/env python
import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def soft_cardinality(list1, list2, similarity_model, threshold=0.6, similarity_function="cosine"):
    embeddings1 = similarity_model.encode(list1, convert_to_tensor=True)
    embeddings2 = similarity_model.encode(list2, convert_to_tensor=True)

    if similarity_function == "cosine":
        similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2).cpu().numpy()
    elif similarity_function == "euclidean":
        similarity_matrix = -torch.cdist(embeddings1, embeddings2, p=2).cpu().numpy()
    else:
        raise ValueError("similarity_function must be 'cosine' or 'euclidean'")

    soft_card = 0
    for i in range(len(list1)):
        max_sim = np.max(similarity_matrix[i, :])
        if max_sim >= threshold:
            soft_card += max_sim
    return soft_card

def normalized_soft_cardinality(list1, list2, similarity_model, threshold=0.6, similarity_function="cosine"):
    sc = soft_cardinality(list1, list2, similarity_model, threshold, similarity_function)
    return sc / len(list1) if len(list1) > 0 else 0.0

def get_terms(file_path, cols_priority):
    """
    Reads a CSV file and returns a list of unique non-null terms from the first available column 
    in cols_priority. Returns an empty list if none of the columns are found.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []
    for col in cols_priority:
        if col in df.columns:
            return df[col].dropna().unique().tolist()
    logger.warning(f"None of the columns {cols_priority} found in {file_path}.")
    return []

def main(args):
    # Define directories and file patterns from command-line arguments
    sets_dir = args.sets_dir
    topics_dir = args.topics_dir
    enrichment_dir = args.enrichment_dir

    sets_pattern = os.path.join(sets_dir, "*_filtered_gene_sets.csv")
    topics_pattern = os.path.join(topics_dir, "*_filtered_topics.csv")
    enrichment_pattern = os.path.join(enrichment_dir, "*__documents.csv")

    # Collect filenames using glob
    sets_files = glob.glob(sets_pattern)
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

    # Build the union of gene set names from all three dictionaries
    all_gene_sets = sorted(set(sets_dict.keys()).union(topics_dict.keys()).union(enrichment_dict.keys()))
    logger.info(f"Processing soft cardinality for {len(all_gene_sets)} gene sets.")

    # Load the SentenceTransformer models
    logger.info(f"Loading SentenceTransformer model: {args.model_name}")
    st_model = SentenceTransformer(args.model_name)
    logger.info(f"Loading SentenceTransformer model for similarity computation: {args.similarity_model_name}")
    similarity_model = SentenceTransformer(args.similarity_model_name)

    results_sc = []

    # Process only the first max_gene_sets gene sets as specified by the argument
    for gene_set in all_gene_sets[:args.max_gene_sets]:
        res = {'gene_set': gene_set}
        
        # Extract terms from the gene sets file (priority: 'Term', then 'description')
        if gene_set in sets_dict and os.path.exists(sets_dict[gene_set]):
            terms_sets = get_terms(sets_dict[gene_set], ["Term", "description"])
        else:
            terms_sets = []
        
        # Extract terms from the topics file (uses 'Term')
        if gene_set in topics_dict and os.path.exists(topics_dict[gene_set]):
            terms_topics = get_terms(topics_dict[gene_set], ["Term"])
        else:
            terms_topics = []
        
        # Extract terms from the enrichment file (priority: 'Term', then 'description')
        if gene_set in enrichment_dict and os.path.exists(enrichment_dict[gene_set]):
            terms_enrichment = get_terms(enrichment_dict[gene_set], ["Term", "description"])
        else:
            terms_enrichment = []
        
        # Compute normalized soft cardinality for each pair if both term lists are non-empty
        if terms_sets and terms_topics:
            res["sets_topics_sc"] = normalized_soft_cardinality(terms_sets, terms_topics, similarity_model, threshold=args.threshold)
        else:
            res["sets_topics_sc"] = np.nan

        if terms_sets and terms_enrichment:
            res["sets_enrichment_sc"] = normalized_soft_cardinality(terms_sets, terms_enrichment, similarity_model, threshold=args.threshold)
        else:
            res["sets_enrichment_sc"] = np.nan

        if terms_topics and terms_enrichment:
            res["topics_enrichment_sc"] = normalized_soft_cardinality(terms_topics, terms_enrichment, similarity_model, threshold=args.threshold)
        else:
            res["topics_enrichment_sc"] = np.nan

        results_sc.append(res)

    # Create a dataframe with gene sets as rows and pairwise soft cardinality scores as columns
    soft_card_df = pd.DataFrame(results_sc).set_index('gene_set')
    logger.info("Completed computing pairwise soft cardinality scores")

    if args.output_csv:
        soft_card_df.to_csv(args.output_csv)
        logger.info(f"Results saved to {args.output_csv}")
    else:
        print(soft_card_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute soft cardinality scores between gene sets from various CSV sources."
    )
    parser.add_argument("--sets_dir", type=str, default="1000geneset_benchmark/results/filtered_sets",
                        help="Directory containing filtered gene sets CSV files.")
    parser.add_argument("--topics_dir", type=str, default="1000geneset_benchmark/results/filtered_topics",
                        help="Directory containing filtered topics CSV files.")
    parser.add_argument("--enrichment_dir", type=str, default="1000geneset_benchmark/results/enrichment_df_listmode",
                        help="Directory containing enrichment CSV files.")
    parser.add_argument("--max_gene_sets", type=int, default=1000,
                        help="Maximum number of gene sets to process.")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="Name of the SentenceTransformer model to load initially.")
    parser.add_argument("--similarity_model_name", type=str, default="paraphrase-MiniLM-L6-v2",
                        help="Name of the SentenceTransformer model for similarity computation.")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Threshold for soft cardinality similarity.")
    parser.add_argument("--output_csv", type=str, default="",
                        help="Path to output CSV file. If provided, results are saved to CSV instead of printed.")
    args = parser.parse_args()
    main(args)
