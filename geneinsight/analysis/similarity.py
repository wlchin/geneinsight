#!/usr/bin/env python3
"""
Module for filtering terms based on semantic similarity.
"""

import pandas as pd
import argparse
import logging
import sys
import os
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import optuna
from optuna.samplers import TPESampler

sampler = TPESampler(seed=10)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value (1 - cosine distance)
    """
    return 1 - cosine(vec1, vec2)

def find_best_similarity_threshold(embeddings, target_rows: int) -> float:
    """
    Find optimal similarity threshold using Optuna optimization when Count column is not available.
    
    Args:
        embeddings: Pre-computed embeddings for terms
        target_rows: Desired number of rows after filtering
        
    Returns:
        Optimal similarity threshold
    """
    logger.info(f"Finding optimal similarity threshold with target rows: {target_rows}")
    
    def objective(trial) -> float:
        # Parameter to optimize
        t = trial.suggest_float('threshold', 0.01, 0.99)  # similarity threshold
        
        # Filter by similarity
        used_indices = set()
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            for j in range(i+1, len(embeddings)):
                if j not in used_indices and 1 - cosine(embeddings[i], embeddings[j]) >= t:
                    used_indices.add(j)
        
        # Get filtered count
        filtered_count = len(embeddings) - len(used_indices)
        
        # Return distance from target
        return abs(filtered_count - target_rows)

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)
    
    # Get best parameter
    best_threshold = study.best_params['threshold']
    
    logger.info(f"Best similarity threshold found: {best_threshold:.4f}")
    return best_threshold

def find_best_params(embeddings, df: pd.DataFrame, target_rows: int) -> Tuple[float, float]:
    """
    Find optimal filtering parameters using Optuna optimization.
    
    Args:
        embeddings: Pre-computed embeddings for terms
        df: DataFrame containing terms and counts
        target_rows: Desired number of rows after filtering
        
    Returns:
        Tuple of (threshold, count_proportion) parameters
    """
    logger.info(f"Finding optimal parameters with target rows: {target_rows}")
    
    def objective(trial) -> float:
        # Parameters to optimize
        t = trial.suggest_float('threshold', 0.01, 0.99)  # similarity threshold
        c = trial.suggest_float('count_prop', 0.01, 0.99)  # count proportion threshold
        
        # Filter by similarity
        used_indices = set()
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            for j in range(i+1, len(embeddings)):
                if j not in used_indices and 1 - cosine(embeddings[i], embeddings[j]) >= t:
                    used_indices.add(j)
        
        # Filter by count proportion
        temp_df = df.iloc[list(set(range(len(df))) - used_indices)]
        max_count = temp_df['Count'].max()
        temp_df = temp_df[temp_df['Count'] >= c * max_count]
        
        # Return distance from target
        return abs(len(temp_df) - target_rows)

    # Create and run Optuna study
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)
    
    # Get best parameters
    best_threshold = study.best_params['threshold']
    best_count_prop = study.best_params['count_prop']
    
    logger.info(f"Best parameters found: threshold={best_threshold:.4f}, count_prop={best_count_prop:.4f}")
    return best_threshold, best_count_prop

def filter_terms_by_similarity(
    input_csv: str,
    output_csv: str,
    target_rows: int = 100,
    model_name: str = 'paraphrase-MiniLM-L6-v2'
) -> pd.DataFrame:
    """
    Filter terms by semantic similarity and term frequency.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        target_rows: Desired number of rows after filtering
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Loading data from {input_csv}")
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    logger.info(f"Loaded {len(df)} terms from input CSV")
    
    # Initialize the model
    logger.info(f"Initializing SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Embed the terms
    logger.info("Embedding terms...")
    embeddings = model.encode(df['Term'].tolist(), show_progress_bar=True)
    
    # Check if 'Count' column exists
    has_count_column = 'Count' in df.columns
    
    if has_count_column:
        logger.info("Found 'Count' column - using similarity and count for filtering")
        # Find optimal parameters for both similarity and count
        best_threshold, best_count_prop = find_best_params(embeddings, df, target_rows)
        
        # Filter terms based on cosine similarity
        logger.info(f"Filtering terms with similarity threshold: {best_threshold:.4f}")
        used_indices = set()
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            for j in range(i + 1, len(embeddings)):
                if cosine_similarity(embeddings[i], embeddings[j]) >= best_threshold:
                    used_indices.add(j)
        
        # Create a new DataFrame with the filtered terms
        filtered_df = df.iloc[list(set(range(len(df))) - used_indices)]
        
        # Filter terms based on count proportion
        logger.info(f"Filtering terms with count proportion: {best_count_prop:.4f}")
        max_count = filtered_df['Count'].max()
        filtered_df = filtered_df[filtered_df['Count'] >= best_count_prop * max_count]
    else:
        logger.info("No 'Count' column found - using only similarity for filtering")
        # Find optimal parameters for similarity only
        best_threshold = find_best_similarity_threshold(embeddings, target_rows)
        
        # Filter terms based on cosine similarity
        logger.info(f"Filtering terms with similarity threshold: {best_threshold:.4f}")
        used_indices = set()
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            for j in range(i + 1, len(embeddings)):
                if cosine_similarity(embeddings[i], embeddings[j]) >= best_threshold:
                    used_indices.add(j)
        
        # Create a new DataFrame with the filtered terms
        filtered_df = df.iloc[list(set(range(len(df))) - used_indices)]
    
    logger.info(f"Final filtered dataset contains {len(filtered_df)} terms")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save the filtered terms to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    logger.info(f"Filtered terms saved to {output_csv}")
    
    return filtered_df

def main():
    """Main entry point for running the script directly"""
    parser = argparse.ArgumentParser(description='Filter terms by auto-optimized similarity.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file')
    parser.add_argument('target_rows', type=int, help='Desired approximate number of filtered rows', nargs='?', default=100)
    parser.add_argument('--model', type=str, default='paraphrase-MiniLM-L6-v2', help='SentenceTransformer model to use')
    
    args = parser.parse_args()
    
    filter_terms_by_similarity(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        target_rows=args.target_rows,
        model_name=args.model
    )

if __name__ == "__main__":
    main()