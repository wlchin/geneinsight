#!/usr/bin/env python3
"""
Consolidate multiple gene set CSV files into a single filtered gene set CSV
by reading the same base filename with different random seed indices.

Example usage:
    python consolidate_gene_sets.py \
        --input_file results/filtered_sets/cad_filtered_gene_sets_seed_0.csv \
        --n_samples 5 \
        --output_file results/filtered_sets/cad_filtered_gene_sets_consolidated.csv
"""

import argparse
import os
import re
import pandas as pd


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Concatenate multiple CSV files that share a similar base name, varying only by seed index."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the initial CSV file (with 'seed_0'), used to infer the pattern for other files."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of seed-based CSV files to concatenate (default: 5)."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to save the concatenated CSV file."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Extract directory and base filename
    input_dir = os.path.dirname(args.input_file)
    base_filename = os.path.basename(args.input_file)

    # We expect something like "results/filtered_sets/<gene_set>_filtered_gene_sets_seed_0.csv"
    # Use a regex to capture the part before "_seed_X.csv"
    match = re.match(r"(.*)_seed_\d+\.csv", base_filename)
    if not match:
        raise ValueError(
            f"Input file '{base_filename}' does not match the pattern '*_seed_#.csv'"
        )

    prefix = match.group(1)

    # Read all files [0..n_samples-1] and store in list
    dataframes = []
    for i in range(args.n_samples):
        filename = f"{prefix}_seed_{i}.csv"
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file does not exist: {path}")

        df = pd.read_csv(path)
        dataframes.append(df)

    # Concatenate all dataframes
    final_df = pd.concat(dataframes, ignore_index=True)

    # Save to output
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    print(f"Concatenated {args.n_samples} files into '{args.output_file}'.")
    print(f"Final dataframe shape: {final_df.shape}")


if __name__ == "__main__":
    main()
