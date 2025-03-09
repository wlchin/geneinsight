"""
Module for converting ontology enrichment results into dictionary format.
"""
import pandas as pd
import ast
import logging
import json
from typing import Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def process_ontology_enrichment(input_data):
    """
    Process the ontology enrichment results and convert to a dictionary format.
    
    Parameters
    ----------
    input_data : str or pd.DataFrame
        Path to the input CSV file containing enrichment results, or a DataFrame directly.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with queries and their associated ontology dictionaries.
    """
    logger.info(f"Processing enrichment results")
    
    # Check if input is a file path or DataFrame
    if isinstance(input_data, str):
        logger.info(f"Reading enrichment results from file: {input_data}")
        data_frame = pd.read_csv(input_data)
    else:
        logger.info("Using provided DataFrame directly")
        data_frame = input_data
    
    # Initialize a list to store the results
    results = []
    
    # Iterate through all rows of the DataFrame
    for index, row in data_frame.iterrows():
        query = row["query"]
        logger.debug(f"Processing query: {query}")
        
        # Safely check if enrichr_df_filtered is empty
        try:
            filtered_value = row["enrichr_df_filtered"]
            # Handle different types of filtered_value
            if isinstance(filtered_value, str):
                # It's a string representation
                is_empty = filtered_value in ('[]', '', 'nan', 'None', 'NaN')
            elif isinstance(filtered_value, list):
                # It's already a list
                is_empty = len(filtered_value) == 0
            elif pd.isna(filtered_value).any() if hasattr(filtered_value, '__iter__') else pd.isna(filtered_value):
                # It's a NaN or None value
                is_empty = True
            else:
                # Something else
                is_empty = False
                
            if is_empty:
                logger.warning(f"No enrichment results for query: {query}")
                results.append({"query": query, "ontology_dict": {}})
                continue
                
        except (KeyError, TypeError) as e:
            logger.error(f"Error checking enrichr_df_filtered for query {query}: {e}")
            results.append({"query": query, "ontology_dict": {}})
            continue
            
        # Parse the string of the list of dicts into a list of dicts
        try:
            # Handle different types for enrichr_df_filtered
            if isinstance(filtered_value, str):
                try:
                    enrichr_list = ast.literal_eval(filtered_value)
                except (SyntaxError, ValueError):
                    try:
                        enrichr_list = json.loads(filtered_value)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse string as list for query {query}: {filtered_value[:100]}...")
                        results.append({"query": query, "ontology_dict": {}})
                        continue
            elif isinstance(filtered_value, list):
                enrichr_list = filtered_value
            else:
                logger.error(f"Unsupported type for enrichr_df_filtered: {type(filtered_value)}")
                results.append({"query": query, "ontology_dict": {}})
                continue
            
            # Skip if the list is empty
            if not enrichr_list:
                logger.warning(f"Empty enrichment results for query: {query}")
                results.append({"query": query, "ontology_dict": {}})
                continue
                
            # Convert the list of dicts into a DataFrame
            enrichr_df = pd.DataFrame(enrichr_list)
            
            # Convert "Term" and "Genes" columns into a dictionary
            # Ensure these columns exist
            if "Term" not in enrichr_df.columns or "Genes" not in enrichr_df.columns:
                required_cols = [col for col in ["Term", "Genes"] if col not in enrichr_df.columns]
                logger.warning(
                    f"Missing required columns {required_cols} for query {query}. "
                    f"Available columns: {enrichr_df.columns.tolist()}"
                )
                # Try to use alternative column names if available
                if "Term" not in enrichr_df.columns and "term" in enrichr_df.columns:
                    enrichr_df["Term"] = enrichr_df["term"]
                if "Genes" not in enrichr_df.columns and "genes" in enrichr_df.columns:
                    enrichr_df["Genes"] = enrichr_df["genes"]
                    
            # If we still don't have the required columns, create an empty dict
            if "Term" not in enrichr_df.columns or "Genes" not in enrichr_df.columns:
                results.append({"query": query, "ontology_dict": {}})
                continue
                
            # Create the dictionary mapping terms to genes
            my_dict = dict(zip(enrichr_df["Term"], enrichr_df["Genes"]))
            
            # Append the result to the list
            results.append({"query": query, "ontology_dict": my_dict})
            
        except Exception as e:
            logger.error(f"Error processing enrichment results for query {query}: {e}")
            results.append({"query": query, "ontology_dict": {}})
    
    # Convert the results list into a DataFrame
    results_df = pd.DataFrame(results)
    logger.info(f"Processed {len(results)} queries into ontology dictionaries")
    
    return results_df


def save_ontology_dictionary(data_frame: pd.DataFrame, output_csv: str) -> None:
    """
    Save the ontology dictionary DataFrame to a CSV file.
    
    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame containing the ontology dictionaries.
    output_csv : str
        Path to the output CSV file.
    """
    data_frame.to_csv(output_csv, index=False)
    logger.info(f"Saved ontology dictionary to: {output_csv}")


def main(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Process ontology enrichment results and save as a dictionary.
    
    Parameters
    ----------
    input_csv : str
        Path to the input CSV file with enrichment results.
    output_csv : str
        Path to the output CSV file for the dictionary.
        
    Returns
    -------
    pd.DataFrame
        The processed ontology dictionary DataFrame.
    """
    results_df = process_ontology_enrichment(input_csv)
    save_ontology_dictionary(results_df, output_csv)
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process ontology enrichment results.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    
    main(args.input_csv, args.output_csv)