"""
Module for performing topic modeling on filtered gene sets (meta-analysis).
"""

import os
import logging
import sys
import pandas as pd
import numpy as np
import importlib.resources as resources
from typing import List, Tuple, Dict, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try to import required packages
try:
    from bertopic import BERTopic
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

def get_embedding_model():
    """
    Load the SentenceTransformer model from the package's embedding_model folder
    
    Returns:
    SentenceTransformer: The loaded model
    """
    if not DEPS_AVAILABLE:
        logger.error("SentenceTransformer not available. Please install required packages.")
        return None
        
    try:
        # Get the path to the embedding_model directory in the package
        model_path = str(resources.files('geneinsight').joinpath('embedding_model'))
        
        # Verify the model directory exists
        if not os.path.exists(model_path):
            logger.error(f"Embedding model directory not found at {model_path}")
            logger.info("Falling back to online model...")
            return SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info(f"Loading embedding model from {model_path}")
        
        # Load the model from the package directory
        model = SentenceTransformer(model_path)
        
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        logger.info("Falling back to online model...")
        return SentenceTransformer("all-MiniLM-L6-v2")

def load_csv_data(file_path: str) -> List[str]:
    """
    Load term data from a CSV file.
    
    Args:
        file_path: Path to the CSV file with filtered gene sets
        
    Returns:
        List of terms
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check if "Term" column exists
        if "Term" in df.columns:
            return df["Term"].tolist()
        else:
            # Try to find a suitable column for terms
            text_columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
            if text_columns:
                logger.warning(f"'Term' column not found. Using '{text_columns[0]}' instead.")
                return df[text_columns[0]].tolist()
            else:
                logger.error("No suitable text column found in CSV.")
                return []
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return []

def initialize_bertopic(
    documents: List[str], 
    num_topics: Optional[int] = 10, 
    embeddings: Optional[List] = None,
    sentence_model = None
) -> Tuple:
    """
    Initialize a BERTopic model and fit the given documents.
    
    Args:
        documents: List of documents to fit
        num_topics: Number of topics to extract
        embeddings: Pre-computed embeddings (optional)
        sentence_model: SentenceTransformer model (optional)
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    if not DEPS_AVAILABLE:
        logger.error("BERTopic dependencies not available. Please install required packages.")
        return None, [], []
    
    if not documents:
        logger.error("No documents provided for topic modeling.")
        return None, [], []
    
    logger.info(f"Initializing BERTopic model with {num_topics} topics")
    
    try:
        if sentence_model is not None:
            logger.info("Using provided SentenceTransformer model for BERTopic")
            topic_model = BERTopic(embedding_model=sentence_model, nr_topics=num_topics)
        else:
            topic_model = BERTopic(nr_topics=num_topics)
        
        if embeddings is not None:
            topics, probs = topic_model.fit_transform(documents, embeddings)
        else:
            topics, probs = topic_model.fit_transform(documents)
            
        return topic_model, topics, probs
    except Exception as e:
        logger.error(f"Error initializing BERTopic model: {e}")
        return None, [], []

def initialize_kmeans_topic_model(
    documents: List[str], 
    num_topics: int = 10, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    embeddings: Optional[List] = None,
    sentence_model = None
) -> Tuple:
    """
    Initialize a KMeans-based topic model and fit the given documents.
    
    Args:
        documents: List of documents to fit
        num_topics: Number of topics to extract
        ncomp: Number of components for dimensionality reduction
        seed_value: Random seed for reproducibility
        embeddings: Pre-computed embeddings (optional)
        sentence_model: SentenceTransformer model (optional)
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available. Please install required packages.")
        return None, [], []
    
    if not documents:
        logger.error("No documents provided for topic modeling.")
        return None, [], []
    
    logger.info(f"Initializing KMeans topic model with {num_topics} topics, {ncomp} components, and seed {seed_value}")
    
    try:
        vectorizer_model = CountVectorizer(stop_words="english")
        cluster_model = KMeans(n_clusters=num_topics, random_state=seed_value)
        dim_model = PCA(n_components=ncomp)
        
        if sentence_model is not None:
            topic_model = BERTopic(
                embedding_model=sentence_model,
                hdbscan_model=cluster_model, 
                umap_model=dim_model, 
                vectorizer_model=vectorizer_model
            )
        else:
            topic_model = BERTopic(
                hdbscan_model=cluster_model, 
                umap_model=dim_model, 
                vectorizer_model=vectorizer_model
            )
        
        if embeddings is not None:
            topics, probs = topic_model.fit_transform(documents, embeddings)
        else:
            topics, probs = topic_model.fit_transform(documents)
            
        return topic_model, topics, probs
    except Exception as e:
        logger.error(f"Error initializing KMeans topic model: {e}")
        return None, [], []

def run_multiple_seed_topic_modeling(
    input_file: str, 
    output_file: str, 
    method: str = "bertopic",
    num_topics: Optional[int] = None, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    n_samples: int = 5,
    use_local_model: bool = True
) -> pd.DataFrame:
    """
    Run topic modeling multiple times with different seeds on filtered gene sets.
    
    Args:
        input_file: Path to the input CSV file with filtered gene sets
        output_file: Path to the output CSV file
        method: Topic modeling method ("bertopic" or "kmeans")
        num_topics: Number of topics to extract
        ncomp: Number of components for dimensionality reduction
        seed_value: Initial random seed value
        n_samples: Number of models to run with different seeds
        use_local_model: Whether to use the locally packaged model (default: True)
        
    Returns:
        DataFrame with the combined results
    """
    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available. Please install required packages.")
        dummy_df = pd.DataFrame({
            "Document": ["Dependencies not available"] * 5,
            "Topic": [-1] * 5,
            "Name": ["Error"] * 5,
            "Probability": [0.0] * 5,
            "Representative_document": [False] * 5,
            "Top_n_words": ["dependencies not available"] * 5,
            "seed": [seed_value] * 5,
        })
        
        # Save to CSV if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            dummy_df.to_csv(output_file, index=False)
            
        return dummy_df
    
    logger.info(f"Running multiple seed topic modeling with {n_samples} samples starting from seed {seed_value}")
    
    # Generate seed values (spaced out to ensure diversity)
    seed_values = [seed_value + i for i in range(n_samples)]
    
    # Load terms
    terms = load_csv_data(input_file)
    
    if not terms:
        logger.error("No terms found in the input file.")
        empty_df = pd.DataFrame(columns=["Document", "Topic", "Name", "Probability", "Representative_document", "Top_n_words", "seed"])
        
        # Save to CSV if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            empty_df.to_csv(output_file, index=False)
            
        return empty_df
    
    logger.info(f"Loaded {len(terms)} terms from {input_file}")
    
    # Create a temporary CSV with terms in 'description' column for compatibility
    temp_df = pd.DataFrame({"description": terms})
    temp_file = os.path.join(os.path.dirname(output_file), "temp_terms.csv")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    temp_df.to_csv(temp_file, index=False)
    
    # Get the sentence transformer model
    if use_local_model:
        logger.info("Using locally packaged SentenceTransformer model")
        sentence_model = get_embedding_model()
    else:
        logger.info("Using default SentenceTransformer model 'all-MiniLM-L6-v2'")
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Precompute embeddings for all terms
    try:
        logger.info("Precomputing sentence embeddings...")
        embeddings = sentence_model.encode(terms, show_progress_bar=True)
        logger.info("Embeddings computed successfully.")
    except Exception as e:
        logger.error(f"Error computing embeddings: {e}")
        embeddings = None
    
    # Run topic modeling with each seed
    dfs = []
    for seed in seed_values:
        logger.info(f"Running topic modeling with seed {seed}")
        
        if method == "kmeans":
            model, topics, probs = initialize_kmeans_topic_model(
                documents=terms,
                num_topics=num_topics,
                ncomp=ncomp,
                seed_value=seed,
                embeddings=embeddings,
                sentence_model=sentence_model
            )
        else:  # bertopic
            model, topics, probs = initialize_bertopic(
                documents=terms,
                num_topics=num_topics,
                embeddings=embeddings,
                sentence_model=sentence_model
            )
        
        if model is None:
            logger.error(f"Failed to initialize model with seed {seed}")
            continue
        
        # Get document info
        try:
            df_seed = model.get_document_info(terms)
            # Add seed to the dataframe
            df_seed["seed"] = seed
            dfs.append(df_seed)
        except Exception as e:
            logger.error(f"Error getting document info for seed {seed}: {e}")
    
    # Combine results
    if not dfs:
        logger.error("No successful topic models. Returning empty DataFrame.")
        empty_df = pd.DataFrame(columns=["Document", "Topic", "Name", "Probability", "Representative_document", "Top_n_words", "seed"])
        
        # Save to CSV if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            empty_df.to_csv(output_file, index=False)
            
        return empty_df
    
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Create directory if it doesn't exist
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        final_df.to_csv(output_file, index=False)
        logger.info(f"Saved combined results to {output_file}")
    
    # Clean up temporary file
    try:
        os.remove(temp_file)
    except OSError:
        pass
    
    return final_df

def main():
    """Command-line interface for the meta analysis module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run topic modeling on filtered gene sets.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file with terms.")
    parser.add_argument("--output_file", required=True, help="Path to save the results as CSV.")
    parser.add_argument("--method", choices=["bertopic", "kmeans"], default="bertopic", help="Topic modeling method.")
    parser.add_argument("--num_topics", type=int, default=None, help="Number of topics to extract.")
    parser.add_argument("--ncomp", type=int, default=2, help="Number of components for dimensionality reduction (KMeans only).")
    parser.add_argument("--seed_value", type=int, default=0, help="Initial random seed value.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of models to run with different seeds.")
    parser.add_argument("--use_external_model", action="store_true", help="Use external model instead of locally packaged model.")
    
    args = parser.parse_args()
    
    run_multiple_seed_topic_modeling(
        input_file=args.input_file,
        output_file=args.output_file,
        method=args.method,
        num_topics=args.num_topics,
        ncomp=args.ncomp,
        seed_value=args.seed_value,
        n_samples=args.n_samples,
        use_local_model=not args.use_external_model  # Default to local model unless specified
    )

if __name__ == "__main__":
    main()