#!/usr/bin/env python3
"""
BERTopic model implementation for topic modeling
"""

import argparse
import os
import sys
import logging
import signal
import pandas as pd
import importlib.resources as resources
from typing import List, Tuple, Optional, Union
from datetime import datetime

# Third-party imports
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", message="No sentence-transformers model found with name")

class SpecificWarningFilter(logging.Filter):
    def filter(self, record):
        # Exclude the specific warning message
        if "No sentence-transformers model found with name" in record.getMessage():
            return False
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sentence_transformer_logger = logging.getLogger('sentence_transformers')
sentence_transformer_logger.setLevel(logging.ERROR)

def get_embedding_model():
    """
    Load the SentenceTransformer model from the package's embedding_model folder
    
    Returns:
    SentenceTransformer: The loaded model
    """
    # Get the path to the embedding_model directory in the package
    model_path = str(resources.files('geneinsight').joinpath('embedding_model'))
    
    # Verify the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Embedding model directory not found at {model_path}")
    
    logger.info(f"Loading embedding model from {model_path}")
    
    # Load the model from the package directory
    model = SentenceTransformer(model_path)
    
    return model

def load_csv_data(file_path: str) -> List[str]:
    """
    Loads data from a CSV file.
    Assumes that there's a column named 'description'.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of document descriptions
    """
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)["description"].tolist()

def initialize_bertopic(
    documents: List[str], 
    num_topics: Optional[int] = None, 
    embeddings: Optional[List] = None,
    sentence_model = None
) -> Tuple[BERTopic, List[int], List[float]]:
    """
    Initializes a BERTopic model and fits the given documents.
    
    Args:
        documents: List of documents to fit
        num_topics: Number of topics to extract
        embeddings: Pre-computed embeddings (optional)
        sentence_model: SentenceTransformer model (optional)
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    logger.info(f"Initializing BERTopic model with {num_topics} topics")
    
    if sentence_model is not None:
        topic_model = BERTopic(embedding_model=sentence_model, nr_topics=num_topics)
    else:
        topic_model = BERTopic(nr_topics=num_topics)
        
    if embeddings is not None:
        topics, probs = topic_model.fit_transform(documents, embeddings)
    else:
        topics, probs = topic_model.fit_transform(documents)
        
    return topic_model, topics, probs

def initialize_kmeans_topic_model(
    documents: List[str], 
    num_topics: int = 10, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    embeddings: Optional[List] = None
) -> Tuple[BERTopic, List[int], List[float]]:
    """
    Initializes a KMeans-based topic model and fits the given documents.
    Uses KMeans for clustering and PCA for dimensionality reduction.
    
    Args:
        documents: List of documents to fit
        num_topics: Number of topics to extract
        ncomp: Number of components for PCA dimensionality reduction
        seed_value: Random seed for reproducibility
        embeddings: Pre-computed embeddings (optional)
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    logger.info(f"Initializing KMeans topic model with {num_topics} topics, {ncomp} components, and seed {seed_value}")
    vectorizer_model = CountVectorizer(stop_words="english")
    cluster_model = KMeans(n_clusters=num_topics, random_state=seed_value)
    dim_model = PCA(n_components=ncomp)
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

def initialize_model_and_fit_documents(
    documents: List[str], 
    method: str = "bertopic", 
    num_topics: Optional[int] = 10, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    embeddings: Optional[List] = None,
    sentence_model = None
) -> Tuple[BERTopic, List[int], List[float]]:
    """
    Initializes and fits a topic model using the specified method (BERTopic or KMeans).
    
    Args:
        documents: List of documents to fit
        method: Topic modeling method ("bertopic" or "kmeans")
        num_topics: Number of topics to extract
        ncomp: Number of components for dimensionality reduction (KMeans only)
        seed_value: Random seed for reproducibility
        embeddings: Pre-computed embeddings (optional)
        sentence_model: SentenceTransformer model (optional)
        
    Returns:
        Tuple of (topic_model, topics, probabilities)
    """
    logger.info(f"Initializing model with method {method}")
    if method == "kmeans":
        return initialize_kmeans_topic_model(documents, num_topics=num_topics, ncomp=ncomp, seed_value=seed_value, embeddings=embeddings)
    elif method == "bertopic":
        return initialize_bertopic(documents, num_topics=num_topics, embeddings=embeddings, sentence_model=sentence_model)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'bertopic' and 'kmeans'.")

def run_topic_modeling_return_df(
    documents: List[str], 
    method: str = "bertopic", 
    num_topics: Optional[int] = 10, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    embeddings: Optional[List] = None,
    sentence_model = None
) -> pd.DataFrame:
    """
    Runs the specified topic modeling method and returns the document info as a dataframe.
    
    Args:
        documents: List of documents to fit
        method: Topic modeling method ("bertopic" or "kmeans")
        num_topics: Number of topics to extract
        ncomp: Number of components for dimensionality reduction (KMeans only)
        seed_value: Random seed for reproducibility
        embeddings: Pre-computed embeddings (optional)
        sentence_model: SentenceTransformer model (optional)
        
    Returns:
        DataFrame containing document information and topic assignments
    """
    logger.info(f"Running topic modeling with method {method}, {num_topics} topics, {ncomp} components, and seed {seed_value}")
    model, topics, probs = initialize_model_and_fit_documents(
        documents,
        method=method,
        num_topics=num_topics,
        ncomp=ncomp,
        seed_value=seed_value,
        embeddings=embeddings,
        sentence_model=sentence_model
    )
    df_info = model.get_document_info(documents)
    return df_info

def run_multiple_seed_topic_modeling(
    input_file: str, 
    output_file: str, 
    method: str = "bertopic",
    num_topics: Optional[int] = None, 
    ncomp: int = 2, 
    seed_value: int = 0, 
    n_samples: int = 1,
    use_sentence_embeddings: bool = True,
    use_local_model: bool = True
) -> pd.DataFrame:
    """
    Runs the topic modeling N times with different seeds,
    concatenates the resulting dataframes, and saves to CSV.
    
    Args:
        input_file: Path to the input CSV file with a 'description' column
        output_file: Path to the output CSV file
        method: Topic modeling method ("bertopic" or "kmeans")
        num_topics: Number of topics to extract
        ncomp: Number of components for dimensionality reduction (KMeans only)
        seed_value: Initial random seed value
        n_samples: Number of different seeds to use
        use_sentence_embeddings: Whether to use SentenceTransformer embeddings
        use_local_model: Whether to use the locally packaged model (default: True)
        
    Returns:
        Concatenated DataFrame containing document information and topic assignments
    """
    logger.info(f"Running multiple seed topic modeling with {n_samples} samples starting from seed {seed_value}")
    seed_values = [seed_value + 10*i for i in range(n_samples)]

    # Load documents
    documents = load_csv_data(input_file)
    
    # Get SentenceTransformer model and precompute embeddings if specified
    precomputed_embeddings = None
    sentence_model = None
    
    if use_sentence_embeddings:
        if use_local_model:
            logger.info("Using locally packaged SentenceTransformer model")
            sentence_model = get_embedding_model()
        else:
            logger.info("Using default SentenceTransformer model 'all-MiniLM-L6-v2'")
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("Precomputing embeddings...")
        precomputed_embeddings = sentence_model.encode(documents, show_progress_bar=True)

    # Run topic modeling with each seed
    dfs = []
    for seed in seed_values:
        logger.info(f"Running topic modeling with seed {seed}")
        df_seed = run_topic_modeling_return_df(
            documents, 
            method=method, 
            num_topics=num_topics, 
            ncomp=ncomp, 
            seed_value=seed,
            embeddings=precomputed_embeddings,
            sentence_model=sentence_model
        )
        # Store which seed was used in a new column
        df_seed["seed"] = seed
        dfs.append(df_seed)
    
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    logger.info(f"Saving concatenated results to {output_file}")
    final_df.to_csv(output_file, index=False)
    
    return final_df

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info('Terminating the script...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    """Main entry point for running the script directly"""
    parser = argparse.ArgumentParser(description="Run topic modeling using BERTopic or KMeans multiple times.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file with a 'description' column.")
    parser.add_argument("--output_file", required=True, help="Path to the output CSV file for concatenated document info.")
    parser.add_argument("--method", choices=["bertopic", "kmeans"], default="bertopic", help="Topic modeling method.")
    parser.add_argument("--num_topics", type=int, default=None, help="Number of topics.")
    parser.add_argument("--ncomp", type=int, default=2, help="Number of components for dimensionality reduction (only for KMeans).")
    parser.add_argument("--seed_value", type=int, default=0, help="Initial seed value.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of topic models to run with different seeds.")
    parser.add_argument("--use_sentence_embeddings", action="store_true", default=True,
                        help="Whether to use SentenceTransformer embeddings for BERTopic (default: True).")
    parser.add_argument("--use_external_model", action="store_true", 
                        help="Use external model instead of the locally packaged model.")

    args = parser.parse_args()

    run_multiple_seed_topic_modeling(
        input_file=args.input_file,
        output_file=args.output_file,
        method=args.method,
        num_topics=args.num_topics,
        ncomp=args.ncomp,
        seed_value=args.seed_value,
        n_samples=args.n_samples,
        use_sentence_embeddings=args.use_sentence_embeddings,
        use_local_model=not args.use_external_model  # Default to local model unless specified
    )

if __name__ == "__main__":
    main()