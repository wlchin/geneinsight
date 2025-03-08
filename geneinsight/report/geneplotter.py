import random
import string
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import ast
import argparse
import os
import logging
from tqdm import tqdm

cmap = plt.get_cmap('viridis')

# Get the colors for 0 and 1
color_0 = cmap(0.0)
color_1 = cmap(1.0)

class Geneplotter:
    def __init__(self, data = None):
        self.data = data

    def generate_subsets(self, num_subsets, min_size, max_size):
        subsets = []
        for _ in range(num_subsets):
            size = random.randint(min_size, max_size)
            subset = random.sample(string.ascii_lowercase, size)
            subsets.append(subset)
        return subsets
    
    def add_newlines(self, text, min_distance=50):
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

    def plot_heatmap(self, query_set, ref_sets, ontology_sets, ref_labels, ontology_labels, savename):
        """
        Plot a heatmap for the given gene sets.

        Parameters
        ----------
        query_set : list
            The query set gene list.
        ref_sets : list of lists
            List of 5 reference dictionary gene sets.
        ontology_sets : list of lists
            List of 5 ontology dictionary gene sets.
        ref_labels : list
            The names of the reference dictionary gene sets.
        ontology_labels : list
            The names of the ontology dictionary gene sets.
        savename : str
            The name of the file to save the heatmap.
        """
        subsets = [query_set] + ref_sets + ontology_sets

        mlb = MultiLabelBinarizer()
        binarized_data = mlb.fit_transform(subsets)
        df = pd.DataFrame(binarized_data, columns=mlb.classes_)

        middle_rows_df = df.iloc[1:-5]  # reference sets
        last_five_rows_df = df.iloc[-5:]  # ontology sets

        ref_labels = [self.add_newlines(label) for label in ref_labels]
        ontology_labels = [self.add_newlines(label) for label in ontology_labels]

        ref_count = len(middle_rows_df)
        onto_count = len(last_five_rows_df)
        total_count = ref_count + onto_count
        height_ratios = [ref_count/total_count, onto_count/total_count]

        # calculate figsize width based on number of columns
        num_columns = len(mlb.classes_)
        figsize_width = num_columns * 0.5 + 7  # for the text

        fig, axes = plt.subplots(2, 1, figsize=(figsize_width, 10), sharex=True, gridspec_kw={'height_ratios': height_ratios})

        sns.heatmap(middle_rows_df, annot=True, cmap='viridis', cbar=False, ax=axes[0], xticklabels=mlb.classes_, yticklabels=ref_labels)
        axes[0].set_title('StringDB Reference sets')
        axes[0].set_ylabel('')

        sns.heatmap(last_five_rows_df, annot=True, cmap='viridis', cbar=False, ax=axes[1], xticklabels=mlb.classes_, yticklabels=ontology_labels)
        axes[1].set_title('Cross ontology reference sets')
        axes[1].set_xlabel('genes in references')
        axes[1].set_ylabel('')

        present_patch = mpatches.Patch(color=color_1, label='1: Gene is present in reference sets (or theme geneset)')
        not_present_patch = mpatches.Patch(color=color_0, label='0: Gene is not present in reference sets (or theme geneset)')

        # Add the legend to your plot
        legend = plt.legend(bbox_to_anchor=(0.5, -0.5), fontsize='large', loc='upper center', handles=[present_patch, not_present_patch])
        legend.set_title('Gene Presence Indicator', prop={'size': 'large'})

        for ax in axes:
            ax.set_xlabel('')

        plt.tight_layout()
        plt.savefig(savename)
        plt.close()

def dict_to_tuple(dict_str):
    """
    Convert a dictionary string into a tuple with a list of keys and a list of lists of genes.

    Args:
        dict_str (str): The dictionary string to convert.

    Returns:
        tuple: A tuple containing a list of keys and a list of lists of genes.
    """
    dict_obj = ast.literal_eval(dict_str)
    keys = list(dict_obj.keys())
    gene_lists = [genes.split(", ") for genes in dict_obj.values()]
    return keys, gene_lists

def ontology_dict_to_tuple(dict_str):
    """
    Convert an ontology dictionary string into a tuple with a list of keys and a list of lists of genes.

    Args:
        dict_str (str): The ontology dictionary string to convert.

    Returns:
        tuple: A tuple containing a list of keys and a list of lists of genes.
    """
    dict_obj = ast.literal_eval(dict_str)
    keys = list(dict_obj.keys())
    gene_lists = [genes.split(";") for genes in dict_obj.values()]
    return keys, gene_lists

def unique_genes_to_keys(dict_str):
    """
    Convert a unique genes dictionary string into a list of keys.

    Args:
        dict_str (str): The unique genes dictionary string to convert.

    Returns:
        list: A list of keys.
    """
    dict_obj = ast.literal_eval(dict_str)
    return list(dict_obj.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df",
        required=True,
        help="Path to the input CSV file containing the DataFrame."
    )
    parser.add_argument(
        "--save_folder",
        default="images",
        help="Folder where generated PNG files will be saved. Defaults to 'images'."
    )
    parser.add_argument(
        "--log_file",
        default="geneplotter.log",
        help="Path to the log file. Defaults to 'geneplotter.log'."
    )
    args = parser.parse_args()

    # Configure logging with timestamps and log to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )

    os.makedirs(args.save_folder, exist_ok=True)

    # Read the input DataFrame
    df = pd.read_csv(args.df)

    # Loop over each row of the DataFrame and generate a heatmap for each entry
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating heatmaps"):
        query_name = row["query"]
        png_name = query_name.replace(" ", "_") + ".png"
        # replace special characters "/"
        png_name = png_name.replace("/", "_")
        savename = os.path.join(args.save_folder, png_name)

        query_set = unique_genes_to_keys(row["unique_genes"])
        ref_keys, ref_lists = dict_to_tuple(row["ref_dict"])
        ontology_keys, ontology_lists = ontology_dict_to_tuple(row["ontology_dict"])

        logging.info(f"Generating heatmap for query: {query_name}")

        gp = Geneplotter()
        gp.plot_heatmap(
            query_set=query_set,
            ref_sets=ref_lists,
            ontology_sets=ontology_lists,
            ref_labels=ref_keys,
            ontology_labels=ontology_keys,
            savename=savename
        )

        logging.info(f"Saved heatmap to: {savename}")