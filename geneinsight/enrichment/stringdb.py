#!/usr/bin/env python3
"""
Module for querying the StringDB API for gene enrichment, with a user-specifiable species.
"""

import argparse
import pandas as pd
import stringdb
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
    Read a gene list from a file. Returns an empty list if the file has no data.
    """
    try:
        df = pd.read_csv(input_file, header=None)
        if df.empty:
            return []
        return df.iloc[:, 0].tolist()
    except pd.errors.EmptyDataError:
        # No columns to parse if file is empty
        return []


def get_string_output(
    list_of_genes: List[str], 
    species: int = 9606
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mode 1: Retrieves the enrichment data from StringDB for a given list of genes.
    """
    # >>> Minimal fix: If empty, return a DataFrame with columns
    if not list_of_genes:
        return pd.DataFrame(columns=["description"]), []

    # Pass the species parameter into get_string_ids and get_enrichment
    string_ids = stringdb.get_string_ids(list_of_genes, species=species)
    enrichment_df = stringdb.get_enrichment(
        string_ids.queryItem,
        species=species
    )
    documents = enrichment_df["description"].unique().tolist()
    return enrichment_df, documents


def query_string_db_individual_genes(
    genes: List[str], 
    log_file: str,
    species: int = 9606
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mode 2: Queries the StringDB database for enrichment information for individual genes.
    """
    # >>> Minimal fix: If empty, return a DataFrame with columns
    if not genes:
        return pd.DataFrame(columns=["description", "gene_queried"]), []

    list_of_df = []
    logger.info("Querying StringDB for enrichment information.")
    num_genes = len(genes)
    num_bad_requests = 0
    bad_genes = []

    for gene in tqdm(genes, desc="Processing genes"):
        try:
            # Pass the species parameter here as well
            string_ids = stringdb.get_string_ids([gene], species=species)
            enrichment_df = stringdb.get_enrichment(
                string_ids.queryItem,
                species=species
            )
            # We add 'gene_queried' to the columns below
            enrichment_df["gene_queried"] = gene
            
            # Only add non-empty DataFrames to the list
            if not enrichment_df.empty:
                list_of_df.append(enrichment_df)
            time.sleep(1)  # Rate limiting
        except KeyboardInterrupt:
            logger.info("Pipeline terminated by user via KeyboardInterrupt.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error querying gene '{gene}': {e}")
            num_bad_requests += 1
            bad_genes.append(gene)
            continue

    num_good_requests = num_genes - num_bad_requests
    percentage_good_requests = (num_good_requests / num_genes) * 100 if num_genes > 0 else 0

    logger.info(f"StringDB query complete. Number of genes with bad requests: {num_bad_requests}")
    logger.info(f"Percentage of genes with good requests: {percentage_good_requests:.2f}%")

    if bad_genes:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            for gene in bad_genes:
                f.write(f"{gene}\n")
        logger.info(f"Bad requests logged to {log_file}")

    # >>> Minimal fix: If no successes, return a DataFrame with columns
    if not list_of_df:
        return pd.DataFrame(columns=["description", "gene_queried"]), []

    # Filter out empty DataFrames before concatenation
    non_empty_dfs = [df for df in list_of_df if not df.empty]
    
    # Check if we have any non-empty DataFrames left
    if not non_empty_dfs:
        return pd.DataFrame(columns=["description", "gene_queried"]), []
        
    total_df = pd.concat(non_empty_dfs, ignore_index=True)
    documents = total_df["description"].unique().tolist()
    return total_df, documents


def process_gene_enrichment(
    input_file: str,
    output_dir: str,
    mode: str = "single",
    species: int = 9606
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process gene enrichment from StringDB based on the specified mode.
    
    Args:
        input_file: Path to input file containing gene list
        output_dir: Directory to store output CSV files
        mode: Query mode ('list' or 'single')
        species: NCBI species ID (default is 9606 for human)
        
    Returns:
        Tuple containing:
        - The enrichment data as a pandas DataFrame
        - A list of unique document descriptions
    """
    os.makedirs(output_dir, exist_ok=True)
    genes = read_gene_list(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]

    if mode == "list":
        enrichment_df, documents = get_string_output(genes, species=species)
    else:
        log_file = os.path.join(output_dir, f"{input_basename}__bad_requests.log")
        enrichment_df, documents = query_string_db_individual_genes(genes, log_file, species=species)

    enrichment_output_path = os.path.join(output_dir, f"{input_basename}__enrichment.csv")
    documents_output_path = os.path.join(output_dir, f"{input_basename}__documents.csv")

    enrichment_df.to_csv(enrichment_output_path, index=False)
    pd.DataFrame({"description": documents}).to_csv(documents_output_path, index=False)
    
    logger.info(f"Enrichment data saved to {enrichment_output_path}")
    logger.info(f"Documents saved to {documents_output_path}")
    
    return enrichment_df, documents


def main():
    parser = argparse.ArgumentParser(description="Query gene enrichment from StringDB.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input CSV file containing gene list.")
    parser.add_argument("-m", "--mode", choices=["list", "single"], default="single",
                        help="Query mode: 'list' for batch enrichment from a gene list, "
                             "'single' for enrichment on individual genes. Default: single")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory to store output CSV files.")
    # New species argument
    parser.add_argument("-s", "--species", type=int, default=9606,
                        help="NCBI species ID to use in the query (default: 9606, human).")

    args = parser.parse_args()
    process_gene_enrichment(
        input_file=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        species=args.species
    )


if __name__ == "__main__":
    main()
