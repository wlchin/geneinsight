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
import requests
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

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


def map_gene_identifiers(gene_list: List[str], species: int = 9606) -> Dict[str, str]:
    """
    Maps input gene identifiers to STRING identifiers using the STRING API.
    
    Args:
        gene_list: List of gene identifiers to map
        species: NCBI species ID (default is 9606 for human)
        
    Returns:
        Dictionary mapping original identifiers to STRING identifiers
    """
    if not gene_list:
        return {}
    
    string_api_url = "https://version-12-0.string-db.org/api"
    output_format = "tsv-no-header"
    method = "get_string_ids"
    
    # Set parameters for the API request
    params = {
        "identifiers": "\r".join(gene_list),
        "species": species,
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see input identifiers in the output
        "caller_identity": "string_db_enrichment_module"  # app name
    }
    
    # Construct URL
    request_url = "/".join([string_api_url, output_format, method])
    
    logger.info(f"Mapping {len(gene_list)} gene identifiers to STRING IDs")
    
    try:
        # Call STRING API
        results = requests.post(request_url, data=params)
        results.raise_for_status()  # Check for HTTP errors
        
        # Parse the results
        mapping = {}
        for line in results.text.strip().split("\n"):
            if not line:
                continue
            l = line.split("\t")
            if len(l) >= 3:  # Ensure we have enough columns
                input_identifier, string_identifier = l[0], l[2]
                mapping[input_identifier] = string_identifier
                
        logger.info(f"Successfully mapped {len(mapping)} out of {len(gene_list)} gene identifiers")
        return mapping
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during STRING ID mapping: {e}")
        return {}


def get_string_output(
    list_of_genes: List[str], 
    species: int = 9606
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mode 1: Retrieves the enrichment data from StringDB for a given list of genes.
    Note: This function assumes that list_of_genes contains only successfully mapped genes.
    """
    # If empty, return a DataFrame with columns
    if not list_of_genes:
        return pd.DataFrame(columns=["description", "original_gene"]), []
    
    # Get the STRING IDs for the mapped genes
    gene_mapping = map_gene_identifiers(list_of_genes, species=species)
    
    # Should have all genes mapped since we're pre-filtering, but check just in case
    if len(gene_mapping) != len(list_of_genes):
        logger.warning(f"Not all genes could be mapped: {len(gene_mapping)} mapped out of {len(list_of_genes)}")
        
    # Use only the STRING IDs for the query
    string_ids_list = list(gene_mapping.values())
    
    if not string_ids_list:
        logger.warning("No genes could be mapped to STRING IDs")
        return pd.DataFrame(columns=["description", "original_gene"]), []
    
    # Pass the species parameter into get_string_ids and get_enrichment
    string_ids = stringdb.get_string_ids(string_ids_list, species=species)
    enrichment_df = stringdb.get_enrichment(
        string_ids.queryItem,
        species=species
    )
    
    # Add a column for the original gene identifiers
    # Create a reverse mapping from STRING ID to original gene
    reverse_mapping = {v: k for k, v in gene_mapping.items()}
    
    # Add the original_gene column if we have results
    if not enrichment_df.empty and "preferredNames" in enrichment_df.columns:
        enrichment_df["original_gene"] = enrichment_df["preferredNames"].map(
            lambda x: reverse_mapping.get(x, "Unknown")
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
    Note: This function assumes that genes have already been successfully mapped.
    """
    # If empty, return a DataFrame with columns
    if not genes:
        return pd.DataFrame(columns=["description", "gene_queried", "original_gene"]), []
    
    # Get the mapping for these genes (they should all be mappable)
    gene_mapping = map_gene_identifiers(genes, species=species)
    
    list_of_df = []
    logger.info(f"Querying StringDB for enrichment information for {len(genes)} mapped genes.")
    num_genes = len(genes)
    num_bad_requests = 0
    bad_genes = []

    for gene in tqdm(genes, desc="Processing genes"):
        try:
            # Get the STRING ID for this gene
            if gene not in gene_mapping:
                logger.warning(f"Unexpectedly, gene '{gene}' could not be mapped to a STRING ID in the querying step")
                num_bad_requests += 1
                bad_genes.append(gene)
                continue
                
            string_id = gene_mapping[gene]
            
            # Query STRING DB with the STRING ID
            string_ids = stringdb.get_string_ids([string_id], species=species)
            enrichment_df = stringdb.get_enrichment(
                string_ids.queryItem,
                species=species
            )
            # We add both the STRING ID and original gene identifier to the columns
            enrichment_df["gene_queried"] = string_id
            enrichment_df["original_gene"] = gene
            
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

    # If no successes, return a DataFrame with columns
    if not list_of_df:
        return pd.DataFrame(columns=["description", "gene_queried", "original_gene"]), []

    # Filter out empty DataFrames before concatenation
    non_empty_dfs = [df for df in list_of_df if not df.empty]
    
    # Check if we have any non-empty DataFrames left
    if not non_empty_dfs:
        return pd.DataFrame(columns=["description", "gene_queried", "original_gene"]), []
        
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

    # First map gene identifiers to STRING IDs
    if not genes:
        logger.warning("Empty gene list provided. Cannot proceed with mapping or enrichment.")
        return pd.DataFrame(columns=["description"]), []
        
    gene_mapping = map_gene_identifiers(genes, species=species)
    
    # Save the mapping to a file
    if gene_mapping:
        mapping_output_path = os.path.join(output_dir, f"{input_basename}__gene_mapping.csv")
        pd.DataFrame({"original_gene": list(gene_mapping.keys()), 
                      "string_id": list(gene_mapping.values())}).to_csv(mapping_output_path, index=False)
        logger.info(f"Gene mapping saved to {mapping_output_path}")
    else:
        logger.warning("No genes could be mapped to STRING IDs. Cannot proceed with enrichment.")
        return pd.DataFrame(columns=["description"]), []
    
    # Only use successfully mapped genes
    mapped_genes = [gene for gene in genes if gene in gene_mapping]
    unmapped_genes = [gene for gene in genes if gene not in gene_mapping]
    
    # Log unmapped genes with explanation
    if unmapped_genes:
        unmapped_log_file = os.path.join(output_dir, f"{input_basename}__unmapped_genes.log")
        os.makedirs(os.path.dirname(unmapped_log_file), exist_ok=True)
        with open(unmapped_log_file, 'w') as f:
            f.write("# Unmapped genes that were filtered out\n")
            f.write("# These are usually non-protein coding genes, pseudogenes, or genes that do not have a corresponding entry in the STRING database\n")
            f.write("# If these genes are important for your analysis, you may need to use alternative identifiers\n\n")
            for gene in unmapped_genes:
                f.write(f"{gene}\n")
        logger.info(f"{len(unmapped_genes)} genes could not be mapped to STRING IDs and were filtered out.")
        logger.info(f"Unmapped genes are typically non-protein coding genes, pseudogenes, or genes without STRING database entries.")
        logger.info(f"List of unmapped genes saved to {unmapped_log_file}")
    
    if not mapped_genes:
        logger.warning("No genes could be mapped to STRING IDs. Cannot proceed with enrichment.")
        return pd.DataFrame(columns=["description"]), []
        
    logger.info(f"Proceeding with {len(mapped_genes)} successfully mapped genes out of {len(genes)} total")

    if mode == "list":
        enrichment_df, documents = get_string_output(mapped_genes, species=species)
    else:
        log_file = os.path.join(output_dir, f"{input_basename}__query_errors.log")
        enrichment_df, documents = query_string_db_individual_genes(mapped_genes, log_file, species=species)

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
    # Species argument
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