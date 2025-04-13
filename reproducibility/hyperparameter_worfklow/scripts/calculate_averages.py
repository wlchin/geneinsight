#!/usr/bin/env python

import glob
import os
import pandas as pd
import argparse
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Combine per-gene-set CSVs into aggregated CSVs.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing the *_detailed.csv and *_average.csv files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the combined results')
    parser.add_argument('--log_level', type=str, required=True,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log_file', type=str, default='combine_averages.log',
                        help='Log file name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Script to combine CSVs started.")
    logging.info(f"Arguments: {args}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created directory: {args.output_dir}")

    # Search for gene set–specific CSVs in input_dir
    detailed_csvs = glob.glob(os.path.join(args.input_dir, "*_detailed.csv"))
    avg_csvs = glob.glob(os.path.join(args.input_dir, "*_average.csv"))
    logging.info(f"Found {len(detailed_csvs)} detailed CSVs, {len(avg_csvs)} average CSVs.")

    combined_detailed_path = os.path.join(args.output_dir, "all_gene_sets_detailed.csv")
    combined_avg_path = os.path.join(args.output_dir, "all_gene_sets_average.csv")

    # Combine detailed
    if detailed_csvs:
        dfs_detailed = [pd.read_csv(f) for f in detailed_csvs]
        combined_detailed = pd.concat(dfs_detailed, ignore_index=True)
        combined_detailed.to_csv(combined_detailed_path, index=False)
        logging.info(f"Combined detailed -> {combined_detailed_path}")
    else:
        logging.info("No detailed CSV files found to combine.")
        combined_detailed = pd.DataFrame()

    # Combine averages
    if avg_csvs:
        dfs_avg = [pd.read_csv(f) for f in avg_csvs]
        combined_avg = pd.concat(dfs_avg, ignore_index=True)
        combined_avg.to_csv(combined_avg_path, index=False)
        logging.info(f"Combined average -> {combined_avg_path}")
    else:
        logging.info("No average CSV files found to combine.")
        combined_avg = pd.DataFrame()

    logging.info("Finished combining all CSVs.")
    print("Combination of CSVs completed successfully.")
    
    if not combined_avg.empty:
        print("Summary of combined averages:")
        for gene_set in combined_avg['gene_set'].unique():
            subset = combined_avg[combined_avg['gene_set'] == gene_set]
            print(f"  Gene set: {gene_set}")
            for _, row in subset.iterrows():
                print(f"    sample_size={row['sample_size']}, threshold={row['threshold']}, "
                      f"avg_sc={row['avg_soft_cardinality']:.4f}")

if __name__ == "__main__":
    main()
