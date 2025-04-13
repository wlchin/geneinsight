import pandas as pd
import glob
import re
import argparse
import os
import numpy as np
from scipy.stats import pearsonr

def compute_t_statistic(r, n):
    """Compute t-statistic from Pearson r and sample size n."""
    # Avoid division by zero if r is exactly 1 or -1.
    if abs(r) == 1:
        return np.inf
    return r * np.sqrt((n - 2) / (1 - r**2))

def main(folder, mover_pattern, cosine_pattern, output_md):
    # Build full glob patterns for mover and cosine files.
    mover_glob = os.path.join(folder, mover_pattern)
    cosine_glob = os.path.join(folder, cosine_pattern)
    
    mover_files = glob.glob(mover_glob)
    cosine_files = glob.glob(cosine_glob)
    
    if not mover_files:
        print(f"No mover files found matching pattern: {mover_glob}")
        return
    if not cosine_files:
        print(f"No cosine files found matching pattern: {cosine_glob}")
        return

    # Build dictionaries mapping file label (numeric part) to DataFrame.
    mover_dict = {}
    cosine_dict = {}
    label_regex = re.compile(r"(\d+)")
    
    for file in mover_files:
        basename = os.path.basename(file)
        match = label_regex.search(basename)
        if match:
            label = int(match.group(1))
            mover_dict[label] = pd.read_csv(file)
        else:
            print(f"Warning: Could not extract label from {file}")
    
    for file in cosine_files:
        basename = os.path.basename(file)
        match = label_regex.search(basename)
        if match:
            label = int(match.group(1))
            cosine_dict[label] = pd.read_csv(file)
        else:
            print(f"Warning: Could not extract label from {file}")
    
    # Process only labels common to both mover and cosine datasets.
    common_labels = sorted(set(mover_dict.keys()).intersection(cosine_dict.keys()))
    
    # Dictionary to store results per label.
    # Each entry will be a dict: {"Pearson": r, "t_statistic": t, "p_value": p}
    results = {}
    
    for label in common_labels:
        mover_df = mover_dict[label]
        cosine_df = cosine_dict[label]
        
        # Merge on common keys. Adjust the key names if needed.
        merged = pd.merge(
            mover_df[['gene_set', 'top_k', 'recall_sentence']],
            cosine_df[['gene_set', 'top_k', 'recall_cosine']],
            on=["gene_set", "top_k"]
        )
        
        n = len(merged)
        if n < 3:
            # Not enough data points to compute a correlation.
            results[label] = {"Pearson": "NA", "t_statistic": "NA", "p_value": "NA"}
        else:
            # Compute Pearson correlation and p-value.
            r, p = pearsonr(merged['recall_sentence'], merged['recall_cosine'])
            t = compute_t_statistic(r, n)
            results[label] = {"Pearson": f"{r:.3f}", "t_statistic": f"{t:.3f}", "p_value": f"{p:.3e}"}
    
    # Create a DataFrame: rows are file labels, columns are metrics.
    table = pd.DataFrame(results).T
    table.index.name = "Label"
    
    markdown_table = table.to_markdown()
    
    with open(output_md, "w") as f:
        f.write(markdown_table)
    
    print(f"Markdown table written to {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Pearson correlation, t-statistic, and p-value for paired CSVs "
                    "using columns 'recall_sentence' and 'recall_cosine'. "
                    "Rows in the markdown table correspond to the numeric label (e.g., 25, 50, 75, 100)."
    )
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing the CSV files.")
    parser.add_argument("--mover_pattern", type=str, required=True,
                        help="File pattern for mover score CSV files (e.g., 'mover_score_results_*.csv').")
    parser.add_argument("--cosine_pattern", type=str, required=True,
                        help="File pattern for cosine score CSV files (e.g., 'cosine_score_results_*.csv').")
    parser.add_argument("--output", type=str, required=True,
                        help="Output markdown file path.")
    
    args = parser.parse_args()
    main(args.folder, args.mover_pattern, args.cosine_pattern, args.output)
