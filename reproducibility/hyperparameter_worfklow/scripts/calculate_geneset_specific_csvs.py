#!/usr/bin/env python

import glob
import os
import pandas as pd
import random
import argparse
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate soft cardinality overlap for gene sets')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--log_level', type=str, required=True, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    parser.add_argument('--file_pattern', type=str, default='*filtered_gene_sets_seed_*.csv',
                        help='Glob pattern to match CSV files')
    parser.add_argument('--model', type=str, default='paraphrase-MiniLM-L6-v2',
                        help='SentenceTransformer model to use')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.9],
                        help='Thresholds for soft cardinality calculation')
    parser.add_argument('--repetitions', type=int, default=5,
                        help='Number of repetitions for each sample size')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Maximum number of samples to use')
    parser.add_argument('--log_file', type=str, default='soft_cardinality_sampling.log',
                        help='Log file name')
    return parser.parse_args()

def soft_cardinality(embeddings1, embeddings2, threshold=0.6, similarity_function="cosine"):
    if similarity_function == "cosine":
        similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2).cpu().numpy()
    elif similarity_function == "euclidean":
        similarity_matrix = -torch.cdist(embeddings1, embeddings2, p=2).cpu().numpy()
    else:
        raise ValueError("similarity_function must be 'cosine' or 'euclidean'")
    soft_card = 0.0
    for i in range(len(embeddings1)):
        max_sim = np.max(similarity_matrix[i])
        if max_sim >= threshold:
            soft_card += max_sim
    return soft_card

def normalized_soft_cardinality(embeddings1, embeddings2, list1_length, threshold=0.6, similarity_function="cosine"):
    sc = soft_cardinality(embeddings1, embeddings2, threshold, similarity_function)
    return sc / list1_length if list1_length > 0 else 0.0

def run_sampling_experiments_for_gene_set(gene_set_files, gene_set_name, args):
    logging.info(f"Initializing SentenceTransformer model for gene set: {gene_set_name}")
    similarity_model = SentenceTransformer(args.model)
    
    file_to_terms = []
    all_unique_terms = set()
    for f_path in gene_set_files:
        df = pd.read_csv(f_path)
        unique_terms = set(df["Term"].unique())
        file_to_terms.append((f_path, unique_terms))
        all_unique_terms |= unique_terms
    all_unique_terms = list(all_unique_terms)
    
    logging.info(f"Gene set '{gene_set_name}' has {len(all_unique_terms)} unique terms.")
    all_term_embeddings = similarity_model.encode(all_unique_terms)
    term_to_index = {term: i for i, term in enumerate(all_unique_terms)}

    reference_embeddings = all_term_embeddings
    reference_length = len(all_unique_terms)

    max_samples = min(args.max_samples, len(gene_set_files))
    sample_sizes = range(1, max_samples + 1)
    repetitions = args.repetitions
    thresholds = args.thresholds

    all_results = []
    total_experiments = len(sample_sizes) * repetitions * len(thresholds)

    logging.info(f"Starting sampling for '{gene_set_name}' with "
                 f"{len(sample_sizes)} sample sizes, {repetitions} reps each, {len(thresholds)} thresholds.")
    
    with tqdm(total=total_experiments, desc=f"Processing {gene_set_name}") as pbar:
        for size in sample_sizes:
            for rep in range(1, repetitions + 1):
                sampled_files = random.sample(file_to_terms, size)
                sampled_terms_set = set()
                for _, terms_set in sampled_files:
                    sampled_terms_set |= terms_set
                sampled_terms = list(sampled_terms_set)

                sampled_indexes = [term_to_index[t] for t in sampled_terms]
                sampled_embeddings = reference_embeddings[sampled_indexes]

                for threshold in thresholds:
                    norm_sc = normalized_soft_cardinality(
                        embeddings1=reference_embeddings,
                        embeddings2=sampled_embeddings,
                        list1_length=reference_length,
                        threshold=threshold,
                        similarity_function="cosine"
                    )
                    all_results.append({
                        "gene_set": gene_set_name,
                        "sample_size": size,
                        "repetition": rep,
                        "threshold": threshold,
                        "soft_cardinality": norm_sc
                    })
                    pbar.update(1)

    results_df = pd.DataFrame(all_results)
    avg_results = (
        results_df.groupby(['sample_size', 'threshold'])['soft_cardinality']
        .mean()
        .reset_index(name='avg_soft_cardinality')
    )
    avg_results['gene_set'] = gene_set_name

    # Log some info
    for _, row in avg_results.iterrows():
        logging.info(f"{gene_set_name}: size={row['sample_size']}, "
                     f"threshold={row['threshold']}, avg_sc={row['avg_soft_cardinality']:.4f}")

    return results_df, avg_results

def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Script to create per-gene-set CSVs started.")
    logging.info(f"Arguments: {args}")

    # 1. Collect CSV files
    all_csv_files = glob.glob(os.path.join(args.input_dir, args.file_pattern))
    logging.info(f"Found {len(all_csv_files)} CSV files matching pattern.")
    if not all_csv_files:
        logging.error("No CSV files found. Exiting.")
        raise FileNotFoundError("No CSV files matching the pattern found.")

    # 2. Group by gene set name
    gene_sets = {}
    for file in all_csv_files:
        gene_set_name = os.path.basename(file).split("_filtered_gene_sets_seed_")[0]
        gene_sets.setdefault(gene_set_name, []).append(file)
    logging.info(f"Found {len(gene_sets)} distinct gene sets: {list(gene_sets.keys())}")

    # 3. Make output dir if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created directory: {args.output_dir}")

    # 4. Process each gene set
    for gene_set_name, gene_set_files in gene_sets.items():
        logging.info(f"Processing '{gene_set_name}' with {len(gene_set_files)} files.")
        detailed_results, avg_results = run_sampling_experiments_for_gene_set(
            gene_set_files, gene_set_name, args
        )

        # Save them in your nested folder
        safe_name = gene_set_name.replace(" ", "_").replace("/", "_")
        detailed_path = os.path.join(args.output_dir, f"{safe_name}_detailed.csv")
        avg_path = os.path.join(args.output_dir, f"{safe_name}_average.csv")
        detailed_results.to_csv(detailed_path, index=False)
        avg_results.to_csv(avg_path, index=False)
        logging.info(f"Saved to {detailed_path} and {avg_path}")

    logging.info("Finished creating per-gene-set CSVs.")
    print("Gene set–specific CSVs created successfully.")

if __name__ == "__main__":
    main()
