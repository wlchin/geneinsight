"""
Module for generating summary statistics and JSON files.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def generate_json_summary(
    enrichment_file: str,
    topic_model_file: str,
    api_file: str,
    clustered_topics_file: str,
    output_file: str
) -> Dict[str, Any]:
    """
    Generate a summary of the gene set analysis in JSON format.
    
    Args:
        enrichment_file: Path to the enrichment CSV file
        topic_model_file: Path to the topic model CSV file
        api_file: Path to the API calls CSV file
        clustered_topics_file: Path to the clustered topics CSV file
        output_file: Path to save the output JSON file
        
    Returns:
        Dictionary containing the summary statistics
    """
    try:
        # Create the directory for the output file if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        logger.info("Loading data for summary generation")
        
        # Load the required CSV files
        enrichment_df = pd.read_csv(enrichment_file)
        topic_model_df = pd.read_csv(topic_model_file)
        api_df = pd.read_csv(api_file)
        clustered_topics_df = pd.read_csv(clustered_topics_file)
        
        # Calculate metrics
        number_of_genes_considered = len(enrichment_df["gene_queried"].unique()) if "gene_queried" in enrichment_df.columns else 0
        documents_considered = enrichment_df.shape[0]
        
        unique_seeds = topic_model_df['seed'].nunique() if 'seed' in topic_model_df.columns else 1
        
        if 'Topic' in topic_model_df.columns and 'seed' in topic_model_df.columns:
            max_topics_per_seed = topic_model_df.groupby('seed')['Topic'].max()
            average_topics = int(max_topics_per_seed.sum() / unique_seeds) if unique_seeds > 0 else 0
            min_max_topics = max_topics_per_seed.min() if len(max_topics_per_seed) > 0 else 0
            max_max_topics = max_topics_per_seed.max() if len(max_topics_per_seed) > 0 else 0
            range_of_max_topics = f"{min_max_topics} to {max_max_topics}"
        else:
            average_topics = 0
            range_of_max_topics = "N/A"
        
        api_calls_made = api_df.shape[0]
        number_of_themes_after_filtering = clustered_topics_df.shape[0]
        
        number_of_clusters = 0
        if 'Cluster' in clustered_topics_df.columns:
            number_of_clusters = clustered_topics_df['Cluster'].nunique()
        
        compression_ratio = round(documents_considered / number_of_themes_after_filtering, 2) if number_of_themes_after_filtering > 0 else 0
        
        # Create summary dictionary
        summary = {
            "number_of_genes_considered": number_of_genes_considered,
            "documents_considered": documents_considered,
            "average_topics": average_topics,
            "range_of_max_topics": range_of_max_topics,
            "api_calls_made": api_calls_made,
            "number_of_themes_after_filtering": number_of_themes_after_filtering,
            "number_of_clusters": number_of_clusters,
            "compression_ratio": compression_ratio,
            "time_of_analysis": datetime.now().isoformat()
        }
        
        # Save to JSON file
        logger.info(f"Saving summary to {output_file}")
        with open(output_file, "w") as json_file:
            json.dump(summary, json_file, indent=4)
        
        logger.info("Summary generation complete")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal summary in case of error
        return {
            "error": str(e),
            "time_of_analysis": datetime.now().isoformat()
        }