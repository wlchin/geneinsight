"""
Module for generating gene heatmaps.
"""

import os
import random
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import ast
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class Geneplotter:
    """
    Class for plotting gene sets as heatmaps.
    """
    
    def __init__(self, data=None):
        """
        Initialize the Geneplotter.
        
        Args:
            data: Optional data to initialize with
        """
        self.data = data
        # Get colormap for the plots
        self.cmap = plt.get_cmap('viridis')
        # Get the colors for 0 and 1
        self.color_0 = self.cmap(0.0)
        self.color_1 = self.cmap(1.0)

    def generate_subsets(self, num_subsets: int, min_size: int, max_size: int) -> List[List[str]]:
        """
        Generate random subsets for testing.
        
        Args:
            num_subsets: Number of subsets to generate
            min_size: Minimum size of each subset
            max_size: Maximum size of each subset
            
        Returns:
            List of randomly generated subsets
        """
        subsets = []
        for _ in range(num_subsets):
            size = random.randint(min_size, max_size)
            subset = random.sample(string.ascii_lowercase, size)
            subsets.append(subset)
        return subsets
    
    def add_newlines(self, text: str, min_distance: int = 50) -> str:
        """
        Add newlines to long text for better display.
        
        Args:
            text: Input text
            min_distance: Minimum characters before adding a newline
            
        Returns:
            Formatted text with newlines
        """
        # Convert the input string into a list of characters
        char_list = list(text)
        
        # Initialize the last newline position to the beginning of the string
        last_newline_pos = 0
        
        # Iterate over the characters in the list
        for i in range(len(char_list)):
            # If the current character is a space and the minimum distance has been reached
            if char_list[i] == ' ' and (i - last_newline_pos) >= min_distance:
                # Insert a newline character after the space
                char_list[i] = '\n'
                # Update the last newline position
                last_newline_pos = i
        
        # Join the list back into a string and return it
        return ''.join(char_list)

    def plot_heatmap(
        self, 
        query_set: List[str], 
        ref_sets: List[List[str]], 
        ontology_sets: List[List[str]], 
        ref_labels: List[str], 
        ontology_labels: List[str], 
        savename: str
    ) -> None:
        """
        Plot a heatmap for the given gene sets.

        Args:
            query_set: The query set gene list
            ref_sets: List of reference dictionary gene sets
            ontology_sets: List of ontology dictionary gene sets
            ref_labels: The names of the reference dictionary gene sets
            ontology_labels: The names of the ontology dictionary gene sets
            savename: The name of the file to save the heatmap
        """
        from sklearn.preprocessing import MultiLabelBinarizer
        
        subsets = [query_set] + ref_sets + ontology_sets

        mlb = MultiLabelBinarizer()
        binarized_data = mlb.fit_transform(subsets)
        df = pd.DataFrame(binarized_data, columns=mlb.classes_)

        middle_rows_df = df.iloc[1:len(ref_sets)+1]  # reference sets
        last_rows_df = df.iloc[len(ref_sets)+1:]  # ontology sets

        ref_labels = [self.add_newlines(label) for label in ref_labels]
        ontology_labels = [self.add_newlines(label) for label in ontology_labels]

        ref_count = len(middle_rows_df)
        onto_count = len(last_rows_df)
        total_count = ref_count + onto_count
        height_ratios = [ref_count/total_count, onto_count/total_count]

        # calculate figsize width based on number of columns
        num_columns = len(mlb.classes_)
        figsize_width = num_columns * 0.5 + 7  # for the text

        fig, axes = plt.subplots(2, 1, figsize=(figsize_width, 10), sharex=True, gridspec_kw={'height_ratios': height_ratios})

        if ref_count > 0:
            sns.heatmap(middle_rows_df, annot=True, cmap='viridis', cbar=False, ax=axes[0], xticklabels=mlb.classes_, yticklabels=ref_labels)
            axes[0].set_title('StringDB Reference sets')
            axes[0].set_ylabel('')
        else:
            axes[0].text(0.5, 0.5, "No significant terms", ha="center", va="center")
            axes[0].axis("off")

        if onto_count > 0:
            sns.heatmap(last_rows_df, annot=True, cmap='viridis', cbar=False, ax=axes[1], xticklabels=mlb.classes_, yticklabels=ontology_labels)
            axes[1].set_title('Cross ontology reference sets')
            axes[1].set_xlabel('genes in references')
            axes[1].set_ylabel('')
        else:
            axes[1].text(0.5, 0.5, "No significant terms", ha="center", va="center")
            axes[1].axis("off")

        present_patch = mpatches.Patch(color=self.color_1, label='1: Gene is present in reference sets (or theme geneset)')
        not_present_patch = mpatches.Patch(color=self.color_0, label='0: Gene is not present in reference sets (or theme geneset)')

        # Add the legend to your plot
        legend = plt.legend(bbox_to_anchor=(0.5, -0.5), fontsize='large', loc='upper center', handles=[present_patch, not_present_patch])
        legend.set_title('Gene Presence Indicator', prop={'size': 'large'})

        for ax in axes:
            ax.set_xlabel('')

        plt.tight_layout()
        plt.savefig(savename)
        plt.close()

def dict_to_tuple(dict_str: str) -> Tuple[List[str], List[List[str]]]:
    """
    Convert a dictionary string into a tuple with a list of keys and a list of lists of genes.

    Args:
        dict_str: The dictionary string to convert

    Returns:
        A tuple containing a list of keys and a list of lists of genes
    """
    try:
        dict_obj = ast.literal_eval(dict_str)
        keys = list(dict_obj.keys())
        gene_lists = [genes.split(", ") for genes in dict_obj.values()]
        return keys, gene_lists
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing dictionary string: {e}")
        return [], []

def ontology_dict_to_tuple(dict_str: str) -> Tuple[List[str], List[List[str]]]:
    """
    Convert an ontology dictionary string into a tuple with a list of keys and a list of lists of genes.

    Args:
        dict_str: The ontology dictionary string to convert

    Returns:
        A tuple containing a list of keys and a list of lists of genes
    """
    try:
        dict_obj = ast.literal_eval(dict_str)
        keys = list(dict_obj.keys())
        gene_lists = [genes.split(";") for genes in dict_obj.values()]
        return keys, gene_lists
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing ontology dictionary string: {e}")
        return [], []

def unique_genes_to_keys(dict_str: str) -> List[str]:
    """
    Convert a unique genes dictionary string into a list of keys.

    Args:
        dict_str: The unique genes dictionary string to convert

    Returns:
        A list of keys
    """
    try:
        dict_obj = ast.literal_eval(dict_str)
        return list(dict_obj.keys())
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing unique genes dictionary string: {e}")
        return []

def generate_heatmaps(df_path: str, save_folder: str, log_file: Optional[str] = None) -> None:
    """
    Generate heatmaps for all entries in a DataFrame.
    
    Args:
        df_path: Path to the input CSV file containing the DataFrame
        save_folder: Folder where generated PNG files will be saved
        log_file: Path to the log file (optional)
    """
    # Configure logging if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
    
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    try:
        # Read the input DataFrame
        df = pd.read_csv(df_path)
        
        logger.info(f"Generating heatmaps for {len(df)} entries")
        
        # Loop over each row of the DataFrame and generate a heatmap for each entry
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating heatmaps"):
            try:
                query_name = row["query"]
                png_name = query_name.replace(" ", "_") + ".png"
                # replace special characters "/"
                png_name = png_name.replace("/", "_")
                savename = os.path.join(save_folder, png_name)
                
                query_set = unique_genes_to_keys(row["unique_genes"])
                
                # Check if the required columns exist
                if "ref_dict" in row and "ontology_dict" in row:
                    ref_keys, ref_lists = dict_to_tuple(row["ref_dict"])
                    ontology_keys, ontology_lists = ontology_dict_to_tuple(row["ontology_dict"])
                    
                    logger.info(f"Generating heatmap for query: {query_name}")
                    
                    gp = Geneplotter()
                    gp.plot_heatmap(
                        query_set=query_set,
                        ref_sets=ref_lists,
                        ontology_sets=ontology_lists,
                        ref_labels=ref_keys,
                        ontology_labels=ontology_keys,
                        savename=savename
                    )
                    
                    logger.info(f"Saved heatmap to: {savename}")
                else:
                    logger.warning(f"Skipping row {index}: Missing required columns")
            except Exception as e:
                logger.error(f"Error generating heatmap for row {index}: {e}")
                continue
        
        logger.info(f"Completed generating {len(df)} heatmaps")
        
    except Exception as e:
        logger.error(f"Error generating heatmaps: {e}")
        import traceback
        traceback.print_exc()