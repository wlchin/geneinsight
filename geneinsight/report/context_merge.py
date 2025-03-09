"""
Context Merge Module for GeneInsight

This module handles merging context data with ontology dictionary data.
"""

import os
import logging
import pandas as pd

def merge_context_ontology(subheadings_path, ontology_dict_path, output_path):
    """
    Merge subheadings data with ontology dictionary data.
    
    Args:
        subheadings_path (str): Path to the subheadings CSV file
        ontology_dict_path (str): Path to the ontology dictionary CSV file
        output_path (str): Path to save the merged data
    """
    logging.info(f"Merging context with ontology data")
    logging.info(f"  Subheadings path: {subheadings_path}")
    logging.info(f"  Ontology dict path: {ontology_dict_path}")
    
    try:
        # Read input files
        subheadings_df = pd.read_csv(subheadings_path)
        ontology_df = pd.read_csv(ontology_dict_path)
        
        # Merge on 'query' column
        merged_df = pd.merge(subheadings_df, ontology_df, on="query", how="inner")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save merged data
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Saved merged data to {output_path}")
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Error merging context with ontology data: {e}")
        raise