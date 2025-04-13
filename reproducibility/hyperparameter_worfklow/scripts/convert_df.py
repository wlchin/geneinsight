#!/usr/bin/env python

import pandas as pd

def convert_df(input_csv, output_csv):
    # Read the first dataframe
    df1 = pd.read_csv(input_csv)
    
    # Rename columns according to the stated correspondence:
    # sample_size           <- topic_modelling_runs
    # threshold             <- threshold
    # avg_soft_cardinality  <- norm_soft_card_score
    # gene_set              <- csv_file
    df2 = df1.rename(
        columns={
            "topic_modelling_runs": "sample_size",
            "threshold": "threshold",
            "norm_soft_card_score": "avg_soft_cardinality",
            "csv_file": "gene_set"
        }
    )
    
    # Reorder columns to match the desired second dataframe
    df2 = df2[["sample_size", "threshold", "avg_soft_cardinality", "gene_set"]]
    
    # Write out the transformed dataframe
    df2.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert the first dataframe to the second by renaming and reordering columns."
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file (df1).")
    parser.add_argument("--output", required=True, help="Path to the output CSV file (df2).")
    args = parser.parse_args()

    convert_df(args.input, args.output)
