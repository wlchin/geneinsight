#!/usr/bin/env python3
"""
Module for querying the StringDB API for gene enrichment, with a user-specifiable species - stringdb_local.py
"""

import argparse
import pandas as pd
import traceback
import logging
import sys
import os
import requests
from typing import List, Tuple, Optional, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_gene_list(input_file: str) -> List[str]:
    """
    Read a gene list from a file, removing duplicates.
    Returns an empty list if the file has no data.
    """
    try:
        df = pd.read_csv(input_file, header=None)
        if df.empty:
            return []
        
        # Get the gene list from the first column
        gene_list = df.iloc[:, 0].tolist()

        # Filter out NaN and empty strings
        gene_list = [
            gene for gene in gene_list
            if pd.notna(gene) and str(gene).strip()
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_genes = []
        for gene in gene_list:
            if gene not in seen:
                seen.add(gene)
                unique_genes.append(gene)
        
        # Log information about duplicates
        duplicate_count = len(gene_list) - len(unique_genes)
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate gene identifiers.")
            logger.info(f"Processing {len(unique_genes)} unique gene identifiers.")
        
        return unique_genes
    except pd.errors.EmptyDataError:
        # No columns to parse if file is empty
        return []


def load_species_data(species: int) -> pd.DataFrame:
    """
    Load the species-specific PKL files from a local .cache folder.
    Downloads from a public git repo if not found.
    """
    def pkl_file_names(sp: int) -> List[str]:
        if sp == 9606:
            return [f"human_{i}.pkl" for i in range(1, 5)]
        elif sp == 10090:
            return [f"mouse_{i}.pkl" for i in range(1, 5)]
        else:
            logger.error(f"No PKL mapping found for species {sp}")
            return []
    
    pkl_list = pkl_file_names(species)
    df_list = []
    for pkl_name in pkl_list:
        local_path = os.path.join(".cache", str(species), pkl_name)
        if not os.path.exists(local_path):
            logger.info(f"Local file {local_path} not found. Downloading...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            url = f"https://raw.githubusercontent.com/wlchin/string_db_enrichment_cache/main/{species}/{pkl_name}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Download for {pkl_name} completed.")
        df_list.append(pd.read_pickle(local_path))
    species_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    return species_df


def process_gene_enrichment(
    input_file: str,
    output_dir: str,
    species: int = 9606
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Process gene enrichment from StringDB based on the specified species.
    
    Args:
        input_file: Path to input file containing gene list
        output_dir: Directory to store output CSV files
        species: NCBI species ID (default is 9606 for human)
        
    Returns:
        Tuple containing:
        - The enrichment data as a pandas DataFrame
        - A list of unique document descriptions
    """
    os.makedirs(output_dir, exist_ok=True)
    species_df = load_species_data(species)

    genes = read_gene_list(input_file)
    if not species_df.empty:
        gene_name_series = species_df["gene_name"].dropna()
        if gene_name_series.empty:
            logger.warning("No gene names available in species data after dropping NaN.")
        else:
            example_gene = str(gene_name_series.iloc[0])
            if example_gene.isupper():
                genes = [g.upper() for g in genes]
            elif example_gene.islower():
                genes = [g.lower() for g in genes]
            elif example_gene.istitle():
                genes = [g.title() for g in genes]
    else:
        logger.warning("Species data is empty. Check PKL files or species argument.")

    filtered_df = species_df[species_df["gene_name"].isin(genes)]
    if filtered_df.empty:
        logger.warning("No rows retrieved for the given gene list.")

    documents = filtered_df["description"].unique().tolist() if not filtered_df.empty else []

    enrichment_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}__enrichment.csv")
    filtered_df.to_csv(enrichment_output_path, index=False)
    logger.info(f"Enrichment data saved to {enrichment_output_path}")
    return filtered_df, documents


def main():
    parser = argparse.ArgumentParser(description="Query gene enrichment locally.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input CSV file containing gene list.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Directory to store output CSV files.")
    parser.add_argument("-s", "--species", type=int, default=9606,
                        help="NCBI species ID to use in the query (default: 9606, human).")
    args = parser.parse_args()
    process_gene_enrichment(
        input_file=args.input,
        output_dir=args.output_dir,
        species=args.species
    )


if __name__ == "__main__":
    main()