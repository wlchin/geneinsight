import glob
import re
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import os

# ----------------------------------------------------------------------
# 1. Helper functions
# ----------------------------------------------------------------------

def parse_filename(filename):
    """
    Extracts gene set and number of samples from filenames like:
    GSE6674_UNSTIM_VS_PL2_3_STIM_BCELL_UP_key_topics_10_samples.csv
    Mirna-30E-5P_Innate_Immunity_Viral_GSE130005_2_key_topics_6_samples.csv
    Quantitative_Transcriptomes_Mtorc2-Suppressed_Glioblastoma_GSE138475_1_key_topics_20_samples.csv
    
    Returns (gene_set, num_samples) or None if format doesn't match.
    """
    # Extract just the filename from the full path
    base_filename = os.path.basename(filename)
    
    # Generic regex to capture any name followed by _key_topics_X_samples.csv
    match = re.search(r"(.+)_key_topics_(\d+)_samples\.csv$", base_filename)
    if match:
        gene_set, num_samples = match.groups()
        return gene_set, int(num_samples)
    return None

def kendall_distance(rank1, rank2):
    """
    Compute the Kendall distance between two ranked lists.
    rank1, rank2: lists or tuples of the same items in rank order.
    """
    position_in_rank2 = {item: i for i, item in enumerate(rank2)}
    distance = 0
    n = len(rank1)
    for i in range(n):
        for j in range(i + 1, n):
            if position_in_rank2[rank1[i]] > position_in_rank2[rank1[j]]:
                distance += 1
    return distance

def rbo_score(r1, r2, p=0.7):
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists r1, r2.
    The parameter p (0 < p < 1) controls the weight assigned at each depth.
    """
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
        
    r1_set, r2_set = set(), set()
    rbo = 0.0
    
    for i in range(min(len(r1), len(r2))):
        r1_set.add(r1[i])
        r2_set.add(r2[i])
        current_overlap = len(r1_set & r2_set)
        rbo += (p ** i) * (current_overlap / (i + 1))
        
    return (1 - p) * rbo

def compare_ranks(df, ground_truth_col, test_col):
    """
    Given a DataFrame with columns for ground truth ranks and test ranks,
    compute Kendall distance and RBO for the 'Term' column.
    Returns a dict with both values.
    """
    sorted_gt = df.sort_values(ground_truth_col)["Term"].tolist()
    sorted_test = df.sort_values(test_col)["Term"].tolist()
    kd = kendall_distance(sorted_gt, sorted_test)
    rbo_val = rbo_score(sorted_gt, sorted_test)
    return {"kendall_distance": kd, "rbo": rbo_val}


def process_data(file_paths=None):
    # ----------------------------------------------------------------------
    # 2. Gather & organize file information
    # ----------------------------------------------------------------------

    # Use provided file paths or use glob if none provided
    all_files = file_paths if file_paths else glob.glob("results/key_topics/*_key_topics_*_samples.csv")
    print(all_files)

    # Create a list of tuples: (filepath, gene_set, num_samples)
    file_info = []
    for filepath in all_files:
        parsed = parse_filename(filepath)
        if parsed:
            gene_set, num_samples = parsed
            file_info.append((filepath, gene_set, num_samples))

    df_info = pd.DataFrame(file_info, columns=["filepath", "gene_set", "num_samples"])

    # ----------------------------------------------------------------------
    # 3. Main loop: per gene set, compare ground truth vs. test files
    # ----------------------------------------------------------------------

    results = []

    # Group by gene_set
    grouped = df_info.groupby("gene_set")

    for gene_set_name, gene_set_df in grouped:
        # 1) Identify the ground-truth file for this gene set (max num_samples)
        #max_samples = gene_set_df["num_samples"].max()
        max_samples = 25
        ground_truth_entry = gene_set_df[gene_set_df["num_samples"] == max_samples]
        if ground_truth_entry.empty:
            # If there's no file with max samples in the group, skip
            continue

        ground_truth_path = ground_truth_entry["filepath"].values[0]
        gt_df = pd.read_csv(ground_truth_path)
        
        # Check if "Count" column exists, adjust if needed (might be named "Weight" or similar)
        rank_column = "Count" if "Count" in gt_df.columns else "Weight"
        if rank_column not in gt_df.columns:
            print(f"Warning: Couldn't find rank column in {ground_truth_path}")
            print(f"Available columns: {gt_df.columns.tolist()}")
            continue
            
        gt_df["rank"] = gt_df[rank_column].rank(ascending=False, method="first")

        # 2) Identify all other files in this gene set
        for _, row in gene_set_df.iterrows():
            test_filepath = row["filepath"]
            test_samples = row["num_samples"]

            test_df = pd.read_csv(test_filepath)
            test_df["rank"] = test_df[rank_column].rank(ascending=False, method="first")

            # 3) Merge on Term
            # Check if "Term" column exists, it might be "Gene" or similar
            term_column = "Term" if "Term" in gt_df.columns else "Gene"
            if term_column not in gt_df.columns:
                print(f"Warning: Couldn't find term column in {ground_truth_path}")
                print(f"Available columns: {gt_df.columns.tolist()}")
                continue
                
            merged_df = pd.merge(
                gt_df, test_df, on=term_column, suffixes=("_gt", "_test")
            )

            # 4) Calculate custom Kendall distance & RBO
            rank_scores = compare_ranks(merged_df, "rank_gt", "rank_test")

            # 5) Calculate built-in Kendall tau
            tau, p_value = kendalltau(merged_df["rank_gt"], merged_df["rank_test"])

            # 6) Store results
            results.append({
                "gene_set": gene_set_name,              # e.g., "GSE6674_UNSTIM_VS_PL2_3_STIM_BCELL_UP"
                "num_samples": test_samples,            # number of samples
                "max_samples": max_samples,             # reference number of samples
                "rbo": rank_scores["rbo"],
                "kendall_distance": rank_scores["kendall_distance"],
                "kendall_tau": tau,
                "kendall_pvalue": p_value
            })

    # ----------------------------------------------------------------------
    # 4. Compile & save final results to CSV
    # ----------------------------------------------------------------------

    # Filter out rows where num_samples is max (since this is the reference)
    filtered_results = []
    for result in results:
        if result["num_samples"] != result["max_samples"]:
            filtered_results.append(result)

    # Compile final results
    final_df = pd.DataFrame(filtered_results)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save to CSV for plotting script to use
    final_df.to_csv("results/topic_modelling_metrics.csv", index=False)
    
    print("Data processing complete. Results saved to 'results/topic_modelling_metrics.csv'")
    
    return final_df

if __name__ == "__main__":
    # When run directly, use the default glob pattern
    process_data()