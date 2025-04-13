import argparse
import pandas as pd
import stringdb
import time
from tqdm import tqdm
import traceback
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def read_gene_list(input_file):
    """
    Read a gene list from a CSV file.

    Args:
        input_file (str): The path to the input CSV file.

    Returns:
        list: A list of genes read from the CSV file.
    """
    genes = pd.read_csv(input_file, header=None).iloc[:, 0].tolist()
    return genes

def get_string_output(list_of_genes):
    """
    Mode 1: Retrieves the enrichment data from StringDB for a given list of genes.

    Args:
        list_of_genes (list): A list of genes for which enrichment data is to be retrieved.

    Returns:
        pandas.DataFrame: The enrichment data as a pandas DataFrame.
        list: A list of unique document descriptions.
    """
    string_ids = stringdb.get_string_ids(list_of_genes)
    enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
    documents = enrichment_df["description"].unique().tolist()
    return enrichment_df, documents

def query_string_db_individual_genes(genes, log_file):
    """
    Mode 2: Queries the StringDB database for enrichment information for individual genes.

    Args:
        genes (list): A list of gene names.
        log_file (str): Path to the log file for bad requests.

    Returns:
        tuple: A tuple containing the enrichment dataframe and a list of unique document descriptions.
    """
    list_of_df = []
    logging.info("Querying StringDB for enrichment information.")

    num_genes = len(genes)
    num_bad_requests = 0
    bad_genes = []

    for gene in tqdm(genes):
        try:
            string_ids = stringdb.get_string_ids([gene])
            enrichment_df = stringdb.get_enrichment(string_ids.queryItem)
            enrichment_df["gene_queried"] = gene
            list_of_df.append(enrichment_df)
            time.sleep(1)
        except:
            traceback.print_exc()
            num_bad_requests += 1
            bad_genes.append(gene)
            continue

    num_good_requests = num_genes - num_bad_requests
    percentage_good_requests = (num_good_requests / num_genes) * 100

    logging.info(
        f"StringDB query complete. Number of genes with bad requests: {num_bad_requests}"
    )
    logging.info(f"Percentage of genes with good requests: {percentage_good_requests}%")

    if bad_genes:
        with open(log_file, 'w') as f:
            for gene in bad_genes:
                f.write(f"{gene}\n")
        logging.info(f"Bad requests logged to {log_file}")

    if len(list_of_df) == 0:
        # If no results returned at all, return empty frames
        return pd.DataFrame(), []

    total_df = pd.concat(list_of_df, ignore_index=True)
    documents = total_df["description"].unique().tolist()
    return total_df, documents

def main():
    parser = argparse.ArgumentParser(description="Query gene enrichment from StringDB.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file containing gene list.")
    parser.add_argument("-m", "--mode", choices=["list", "single"], default="single",
                        help="Query mode: 'list' for batch enrichment from a gene list, 'single' for enrichment on individual genes. Default: single")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store output CSV files.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Read genes
    genes = read_gene_list(args.input)

    # Determine basename for output files
    input_basename = os.path.splitext(os.path.basename(args.input))[0]

    if args.mode == "list":
        enrichment_df, documents = get_string_output(genes)
    else:
        log_file = os.path.join(args.output_dir, f"{input_basename}__bad_requests.log")
        enrichment_df, documents = query_string_db_individual_genes(genes, log_file)

    # Save output CSV files with the specified prefix
    enrichment_output_path = os.path.join(args.output_dir, f"{input_basename}__enrichment.csv")
    documents_output_path = os.path.join(args.output_dir, f"{input_basename}__documents.csv")

    enrichment_df.to_csv(enrichment_output_path, index=False)
    pd.DataFrame({"description": documents}).to_csv(documents_output_path, index=False)
    logging.info(f"Enrichment data saved to {enrichment_output_path}")
    logging.info(f"Documents saved to {documents_output_path}")

if __name__ == "__main__":
    main()
