import argparse
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import signal
import sys
import logging
from datetime import datetime
# Import GPU-accelerated components
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_csv_data(file_path):
    """
    Loads data from a CSV file.
    Assumes that there's a column named 'description'.
    """
    logging.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)["description"].tolist()

def initialize_bertopic(documents, num_topics=10, umap_components=5, umap_neighbors=15, 
                       umap_min_dist=0.0, hdbscan_min_samples=10):
    """
    Initializes a BERTopic model with GPU acceleration and fits the given documents.
    """
    logging.info(f"Initializing BERTopic model with GPU acceleration")
    logging.info(f"UMAP parameters: components={umap_components}, neighbors={umap_neighbors}, min_dist={umap_min_dist}")
    logging.info(f"HDBSCAN parameters: min_samples={hdbscan_min_samples}")
    
    # Create instances of GPU-accelerated UMAP and HDBSCAN
    umap_model = UMAP(
        n_components=umap_components, 
        n_neighbors=umap_neighbors, 
        min_dist=umap_min_dist
    )
    
    hdbscan_model = HDBSCAN(
        min_samples=hdbscan_min_samples, 
        gen_min_span_tree=True, 
        prediction_data=True
    )
    
    # Pass the GPU-accelerated models to BERTopic
    topic_model = BERTopic(
        nr_topics=num_topics,
        umap_model=umap_model, 
        hdbscan_model=hdbscan_model
    )
    
    # Fit and transform documents
    topics, probs = topic_model.fit_transform(documents)
    return topic_model, topics, probs

def initialize_kmeans_topic_model(documents, num_topics=10, ncomp=2, seed_value=0):
    """
    Initializes a KMeans-based topic model and fits the given documents.
    Uses KMeans for clustering and PCA for dimensionality reduction.
    """
    logging.info(f"Initializing KMeans topic model with {num_topics} topics, {ncomp} components, and seed {seed_value}")
    vectorizer_model = CountVectorizer(stop_words="english")
    cluster_model = KMeans(n_clusters=num_topics, random_state=seed_value)
    dim_model = PCA(n_components=ncomp)
    topic_model = BERTopic(
        hdbscan_model=cluster_model, 
        umap_model=dim_model, 
        vectorizer_model=vectorizer_model
    )
    topics, probs = topic_model.fit_transform(documents)
    return topic_model, topics, probs

def normalize_embeddings(embeddings):
    """
    Normalize embeddings using cuML's GPU-accelerated normalize function
    """
    logging.info("Normalizing embeddings using GPU acceleration")
    return normalize(embeddings)

def initialize_model_and_fit_documents(documents, method="bertopic", num_topics=10, ncomp=2, seed_value=0,
                                      umap_components=5, umap_neighbors=15, umap_min_dist=0.0, 
                                      hdbscan_min_samples=10):
    """
    Initializes and fits a topic model using the specified method (BERTopic with GPU or KMeans).
    """
    logging.info(f"Initializing model with method {method}")
    if method == "kmeans":
        return initialize_kmeans_topic_model(documents, num_topics=num_topics, ncomp=ncomp, seed_value=seed_value)
    elif method == "bertopic":
        return initialize_bertopic(
            documents, 
            num_topics=num_topics,
            umap_components=umap_components,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            hdbscan_min_samples=hdbscan_min_samples
        )
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'bertopic' and 'kmeans'.")

def run_topic_modeling_return_df(input_file, method="bertopic", num_topics=10, ncomp=2, seed_value=0,
                                umap_components=5, umap_neighbors=15, umap_min_dist=0.0, 
                                hdbscan_min_samples=10):
    """
    Loads the data, runs the specified topic modeling method with GPU acceleration if bertopic,
    and returns the document info as a dataframe.
    """
    logging.info(f"Running topic modeling on {input_file} with method {method}")
    topic_list = load_csv_data(input_file)
    model, topics, probs = initialize_model_and_fit_documents(
        topic_list, 
        method=method, 
        num_topics=num_topics, 
        ncomp=ncomp, 
        seed_value=seed_value,
        umap_components=umap_components,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        hdbscan_min_samples=hdbscan_min_samples
    )
    df_info = model.get_document_info(topic_list)
    return df_info

def run_multiple_seed_topic_modeling(input_file, output_file, method="bertopic",
                                    num_topics=10, ncomp=2, seed_value=0, n_samples=1,
                                    umap_components=5, umap_neighbors=15, umap_min_dist=0.0,
                                    hdbscan_min_samples=10):
    """
    Runs the topic modeling N times with different seeds using GPU acceleration,
    concatenates the resulting dataframes, saves to CSV,
    and returns the final concatenated dataframe.
    """
    logging.info(f"Running multiple seed topic modeling with {n_samples} samples starting from seed {seed_value}")
    seed_values = [seed_value + i for i in range(n_samples)]

    dfs = []
    for seed in seed_values:
        logging.info(f"Running topic modeling with seed {seed}")
        df_seed = run_topic_modeling_return_df(
            input_file, 
            method=method, 
            num_topics=num_topics, 
            ncomp=ncomp, 
            seed_value=seed,
            umap_components=umap_components,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            hdbscan_min_samples=hdbscan_min_samples
        )
        # Store which seed was used in a new column
        df_seed["seed"] = seed
        dfs.append(df_seed)
    
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Saving concatenated results to {output_file}")
    final_df.to_csv(output_file, index=False)
    return final_df

def signal_handler(sig, frame):
    logging.info('Terminating the script...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    parser = argparse.ArgumentParser(description="Run topic modeling using BERTopic with GPU acceleration or KMeans multiple times.")
    parser.add_argument("--input_file", required=True, help="Path to the input CSV file with a 'description' column.")
    parser.add_argument("--output_file", required=True, help="Path to the output CSV file for concatenated document info.")
    parser.add_argument("--method", choices=["bertopic", "kmeans"], default="bertopic", help="Topic modeling method.")
    parser.add_argument("--num_topics", type=int, default=None, help="Number of topics.")
    parser.add_argument("--ncomp", type=int, default=2, help="Number of components for dimensionality reduction (only for KMeans).")
    parser.add_argument("--seed_value", type=int, default=0, help="Initial seed value.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of topic models to run with different seeds.")
    # Add GPU-specific parameters
    parser.add_argument("--umap_components", type=int, default=5, help="Number of components for UMAP.")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="Number of neighbors for UMAP.")
    parser.add_argument("--umap_min_dist", type=float, default=0.0, help="Minimum distance for UMAP.")
    parser.add_argument("--hdbscan_min_samples", type=int, default=10, help="Minimum samples for HDBSCAN.")

    args = parser.parse_args()

    run_multiple_seed_topic_modeling(
        input_file=args.input_file,
        output_file=args.output_file,
        method=args.method,
        num_topics=args.num_topics,
        ncomp=args.ncomp,
        seed_value=args.seed_value,
        n_samples=args.n_samples,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_samples=args.hdbscan_min_samples
    )

if __name__ == "__main__":
    main()