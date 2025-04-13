import os
import pandas as pd
import argparse
from collections import Counter

def count_strings_most_common(single_list, top_n=None):
    # Count the occurrences of each string in the list
    counter = Counter(single_list)
    
    if top_n is not None:
        return counter.most_common(top_n)
    return counter.most_common()  # Return all counts if top_n is not specified

def main(input_file, output_file, top_n):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Extract the terms from the DataFrame
    terms_list = df[df["Representative_document"] == True]["Document"].tolist()

    # Count the occurrences of each term and get the top N terms
    top_terms = count_strings_most_common(terms_list, top_n)

    # Convert the top terms to a DataFrame
    top_terms_df = pd.DataFrame(top_terms, columns=["Term", "Count"])

    # Save the top terms DataFrame to a CSV file
    top_terms_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count top terms in topic modeling results.")
    parser.add_argument("input_file", type=str, help="Path to the input topic modeling CSV file.")
    parser.add_argument("output_file", type=str, help="Path to the output top terms CSV file.")
    parser.add_argument("--top_n", type=int, default=None, help="Number of top terms to output.")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.top_n)
