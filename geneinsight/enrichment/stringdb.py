#!/usr/bin/env python3
"""
Module for querying the StringDB API for gene enrichment.
"""

import argparse
import pandas as pd
import geneinsight.geneinsight.enrichment.stringdb as stringdb
import time
import traceback
import logging
import sys
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def read_gene_list(input_file: str) -> List[str]:
    """
    Read a gene list from a file.

    Args:
        input_file: Path to the input file

    Returns:
        List of gene names
    """
    genes = pd.read_csv(input_file, header=None).iloc[:, 0].tolist()
    return genes

def get_string_output(list_of_genes: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mode 1: Retrieves the enrichment data from StringDB for a given list of genes.

    Args:
        list_of_genes: A list of genes for which enrichment data is to be retrieved

    Returns:
        Tuple containing:
        - The enrichment data as a pandas DataFrame
        - A list of unique document descriptions
    """
    string_ids = stringdb.get_string_ids(list_of_genes)
    enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
    documents = enrichment_df["description"].unique().tolist()
    return enrichment_df, documents

def query_string_db_individual_genes(
    genes: List[str], 
    log_file: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mode 2: Queries the StringDB database for enrichment information for individual genes.

    Args:
        genes: A list of gene names
        log_file: Path to the log file for bad requests

    Returns:
        Tuple containing:
        - The enrichment data as a pandas DataFrame
        - A list of unique document descriptions
    """
    list_of_df = []
    logger.info("Querying StringDB for enrichment information.")

    num_genes = len(genes)
    num_bad_requests = 0
    bad_genes = []

    for gene in tqdm(genes, desc="Processing genes"):
        try:
            string_ids = stringdb.get_string_ids([gene])
            enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
            enrichment_df["gene_queried"] = gene
            list_of_df.append(enrichment_df)
            time.sleep(1)  # Rate limiting
        except:
            traceback.print_exc()
            num_bad_requests += 1
            bad_genes.append(gene)
            continue

    num_good_requests = num_genes - num_bad_requests
    percentage_good_requests = (num_good_requests / num_genes) * 100 if num_genes > 0 else 0

    logger.info(
        f"StringDB query complete. Number of genes with bad requests: {num_bad_requests}"
    )
    logger.info(f"Percentage of genes with good requests: {percentage_good_requests:.2f}%")

    if bad_genes:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            for gene in bad_genes:
                f.write(f"{gene}\n")
        logger.info(f"Bad requests logged to {log_file}")

    if len(list_of_df) == 0:
        # If no results returned at all, return empty frames
        return pd.DataFrame(), []

    total_df = pd.concat(list_of_df, ignore_index=True)
    documents = total_df["description"].unique().tolist()
    return total_df, documents

def process_gene_enrichment(
    input_file: str,
    output_dir: str,
    mode: str = "single"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process gene enrichment from StringDB based on the specified mode.
    
    Args:
        input_file: Path to input file containing gene list
        output_dir: Directory to store output CSV files
        mode: Query mode ('list' or 'single')
        
    Returns:
        Tuple containing:
        - The enrichment data as a pandas DataFrame
        - A list of unique document descriptions
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read genes
    genes = read_gene_list(input_file)

    # Determine basename for output files
    input_basename = os.path.splitext(os.path.basename(input_file))[0]

    if mode == "list":
        enrichment_df, documents = get_string_output(genes)
    else:
        log_file = os.path.join(output_dir, f"{input_basename}__bad_requests.log")
        enrichment_df, documents = query_string_db_individual_genes(genes, log_file)

    # Save output CSV files
    enrichment_output_path = os.path.join(output_dir, f"{input_basename}__enrichment.csv")
    documents_output_path = os.path.join(output_dir, f"{input_basename}__documents.csv")

    enrichment_df.to_csv(enrichment_output_path, index=False)
    pd.DataFrame({"description": documents}).to_csv(documents_output_path, index=False)
    
    logger.info(f"Enrichment data saved to {enrichment_output_path}")
    logger.info(f"Documents saved to {documents_output_path}")
    
    return enrichment_df, documents

def main():
    """Main entry point for running the script directly"""
    parser = argparse.ArgumentParser(description="Query gene enrichment from StringDB.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file containing gene list.")
    parser.add_argument("-m", "--mode", choices=["list", "single"], default="single",
                        help="Query mode: 'list' for batch enrichment from a gene list, 'single' for enrichment on individual genes. Default: single")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store output CSV files.")

    args = parser.parse_args()
    
    process_gene_enrichment(
        input_file=args.input,
        output_dir=args.output_dir,
        mode=args.mode
    )

if __name__ == "__main__":
    main()