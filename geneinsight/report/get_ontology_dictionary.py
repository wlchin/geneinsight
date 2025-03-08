import pandas as pd
import ast
import argparse

def main(input_csv, output_csv):
    data_frame = pd.read_csv(input_csv)

    # Initialize a list to store the results
    results = []

    # Iterate through all rows of the DataFrame
    for index, row in data_frame.iterrows():
        query = row["query"]
        enrichr_df_filtered = row["enrichr_df_filtered"]
        
        # Parse the string of the list of dicts into a list of dicts
        enrichr_list = ast.literal_eval(enrichr_df_filtered)
        
        # Convert the list of dicts into a DataFrame
        enrichr_df = pd.DataFrame(enrichr_list)
        
        # Convert “Term” and “Genes” columns into a dictionary
        my_dict = dict(zip(enrichr_df["Term"], enrichr_df["Genes"]))
        
        # Append the result to the list
        results.append({"query": query, "ontology_dict": my_dict})

    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to the output CSV file
    results_df.to_csv(output_csv, index=False)
    print(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ontology enrichment results.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)



