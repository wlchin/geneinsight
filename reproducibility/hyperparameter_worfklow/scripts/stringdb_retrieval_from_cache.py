import argparse
import pandas as pd
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

def concatenate_gene_data(gene_list, cache_folder):
    """
    Iterate through the cache folder for each gene in the gene list and concatenate the data.

    Args:
        gene_list (list): List of genes.
        cache_folder (str): Path to the folder containing cached CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame of gene data.
    """
    concatenated_df = pd.DataFrame()
    for gene in gene_list:
        file_path = os.path.join(cache_folder, f"{gene}_enrichment.csv")
        if os.path.exists(file_path):
            gene_df = pd.read_csv(file_path)
            concatenated_df = pd.concat([concatenated_df, gene_df], ignore_index=True)
    return concatenated_df

def save_dataframes(concatenated_df, output_folder, prefix):
    """
    Save the concatenated DataFrame and a DataFrame containing only the "description" column.

    Args:
        concatenated_df (pd.DataFrame): The concatenated DataFrame of gene data.
        output_folder (str): Path to the folder where the DataFrames will be saved.
        prefix (str): Prefix for the saved file names.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    full_df_path = os.path.join(output_folder, f"{prefix}__enrichment.csv")
    description_df_path = os.path.join(output_folder, f"{prefix}__documents.csv")
    
    concatenated_df.to_csv(full_df_path, index=False)
    concatenated_df[['description']].to_csv(description_df_path, index=False)

def process_gene_list_file(gene_list_file, cache_folder, output_folder):
    """
    Process a gene list file: read the gene list, retrieve data from cache, and save the DataFrames.

    Args:
        gene_list_file (str): Path to the gene list file.
        cache_folder (str): Path to the folder containing cached CSV files.
        output_folder (str): Path to the folder where the DataFrames will be saved.
    """
    gene_list = read_gene_list(gene_list_file)
    concatenated_df = concatenate_gene_data(gene_list, cache_folder)
    prefix = os.path.basename(gene_list_file).split('.')[0]
    save_dataframes(concatenated_df, output_folder, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gene list file and retrieve data from cache.")
    parser.add_argument("gene_list_file", type=str, help="Path to the gene list file.")
    parser.add_argument("cache_folder", type=str, help="Path to the folder containing cached CSV files.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the DataFrames will be saved.")
    
    args = parser.parse_args()
    
    process_gene_list_file(args.gene_list_file, args.cache_folder, args.output_folder)


