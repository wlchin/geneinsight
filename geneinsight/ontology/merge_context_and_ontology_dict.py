import pandas as pd
import argparse

def main(ontology_dict_csv, subheadings_csv, output_csv):
    # Read the input CSV files
    ontology_dict_df = pd.read_csv(ontology_dict_csv)
    subheadings_df = pd.read_csv(subheadings_csv)

    # Merge the DataFrames on the "query" column
    merged_df = pd.merge(subheadings_df, ontology_dict_df, on="query", how="inner")

    if merged_df.empty:
        print("Warning: Merge resulted in empty DataFrame. No matching 'query' values found.")

    # Save the merged DataFrame to the output CSV file
    merged_df.to_csv(output_csv, index=False)
    print(merged_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ontology dictionary and subheadings CSV on the 'query' column.")
    parser.add_argument("--ontology_dict_csv", required=True, help="Path to the ontology dictionary CSV file.")
    parser.add_argument("--subheadings_csv", required=True, help="Path to the subheadings CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    main(args.ontology_dict_csv, args.subheadings_csv, args.output_csv)
