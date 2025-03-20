import pandas as pd
import os
import logging
import pkg_resources
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import optuna
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Load the SentenceTransformer model from the package's embedding_model folder
    
    Returns:
    SentenceTransformer: The loaded model
    """
    try:
        # Get the path to the embedding_model directory in the package
        model_path = pkg_resources.resource_filename('geneinsight', 'embedding_model')
        
        # Verify the model directory exists
        if not os.path.exists(model_path):
            logger.warning(f"Embedding model directory not found at {model_path}")
            logger.info("Falling back to online model...")
            return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        logger.info(f"Loading embedding model from {model_path}")
        
        # Load the model from the package directory
        model = SentenceTransformer(model_path)
        
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        logger.info("Falling back to online model...")
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def run_clustering(
    input_csv: str, 
    output_csv: str, 
    min_clusters: int = 5, 
    max_clusters: int = 10, 
    n_trials: int = 100,
    use_local_model: bool = True
) -> None:
    """
    Cluster terms using sentence embeddings and optimization.

    Args:
        input_csv (str): Path to the input CSV file containing a 'Term' column.
        output_csv (str): Path where the output CSV (with cluster labels) will be saved.
        min_clusters (int, optional): Minimum number of clusters to try. Defaults to 5.
        max_clusters (int, optional): Maximum number of clusters to try. Defaults to 10.
        n_trials (int, optional): Number of trials for optimization. Defaults to 100.
        use_local_model (bool, optional): Whether to use the locally packaged model. Defaults to True.
    """
    logger.info(f"Starting clustering of terms from {input_csv}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Read the input CSV
    try:
        x = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(x)} terms from input CSV")
    except Exception as e:
        logger.error(f"Error reading input CSV: {e}")
        return
    
    # Handle special case for single term
    if len(x) == 1:
        x['Cluster'] = 0
        x.to_csv(output_csv, index=False)
        logger.info("Single term detected. Assigned to cluster 0.")
        print("Optimal clustering algorithm: N/A (single term)")
        print("Optimal number of clusters: 1")
        return

    # Load the embedding model (local or online)
    try:
        if use_local_model:
            logger.info("Using locally packaged sentence transformer model")
            embedder = get_embedding_model()
        else:
            logger.info("Using online sentence transformer model")
            embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
        logger.info("Encoding terms with sentence transformer model")
        embeddings = embedder.encode(x['Term'].tolist())
    except Exception as e:
        logger.error(f"Error during term embedding: {e}")
        return
    
    # Adjust min_clusters based on the number of samples
    # davies_bouldin_score requires 1 < n_clusters < n_samples
    actual_min_clusters = max(min_clusters, 2)
    actual_max_clusters = min(max_clusters, len(x) - 1)
    
    logger.info(f"Adjusted clustering range: {actual_min_clusters} to {actual_max_clusters} clusters")
    
    # If we can't satisfy the constraints, default to 2 clusters 
    # (or however many samples we have if less than 3)
    if actual_min_clusters > actual_max_clusters or len(x) < 3:
        if len(x) < 3:
            # For 2 samples, we can only have 1 cluster
            x['Cluster'] = 0
            x.to_csv(output_csv, index=False)
            logger.info("Too few samples for clustering. All terms assigned to cluster 0.")
            print("Optimal clustering algorithm: N/A (too few samples)")
            print("Optimal number of clusters: 1")
            return
        else:
            # Force exactly 2 clusters if we can't satisfy the constraints
            actual_min_clusters = actual_max_clusters = 2
            logger.info("Constraints can't be satisfied. Forcing 2 clusters.")
    
    def objective(trial):
        # Suggest a clustering algorithm and number of clusters
        clustering_algorithm = trial.suggest_categorical('clustering_algorithm', ['hierarchical', 'kmeans', 'spectral'])
        N = trial.suggest_int('n_clusters', actual_min_clusters, actual_max_clusters)
        
        if clustering_algorithm == 'hierarchical':
            cluster_model = AgglomerativeClustering(n_clusters=N)
        elif clustering_algorithm == 'kmeans':
            cluster_model = KMeans(n_clusters=N)
        elif clustering_algorithm == 'spectral':
            # For spectral clustering, we need to ensure n_neighbors doesn't exceed the number of samples
            n_neighbors = min(10, len(x) - 1)  # Default is 10, but can't exceed n_samples - 1
            cluster_model = SpectralClustering(n_clusters=N, affinity='nearest_neighbors', n_neighbors=n_neighbors)
        
        labels = cluster_model.fit_predict(embeddings)
        # Compute scores: lower is better
        db_score = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        return db_score - ch_score

    # Optimize clustering parameters using Optuna
    logger.info(f"Starting Optuna optimization with {n_trials} trials")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    optimal_clustering_algorithm = study.best_params['clustering_algorithm']
    optimal_N = study.best_params['n_clusters']
    
    logger.info(f'Optimal clustering algorithm: {optimal_clustering_algorithm}')
    logger.info(f'Optimal number of clusters: {optimal_N}')
    
    print(f'Optimal clustering algorithm: {optimal_clustering_algorithm}')
    print(f'Optimal number of clusters: {optimal_N}')
    
    # Cluster the terms using the optimal parameters
    if optimal_clustering_algorithm == 'hierarchical':
        cluster_model = AgglomerativeClustering(n_clusters=optimal_N)
    elif optimal_clustering_algorithm == 'kmeans':
        cluster_model = KMeans(n_clusters=optimal_N)
    elif optimal_clustering_algorithm == 'spectral':
        # Need to apply the same n_neighbors limit when using the optimal parameters
        n_neighbors = min(10, len(x) - 1)
        cluster_model = SpectralClustering(n_clusters=optimal_N, affinity='nearest_neighbors', n_neighbors=n_neighbors)
    
    logger.info(f"Clustering terms with {optimal_clustering_algorithm}")
    labels = cluster_model.fit_predict(embeddings)
    x['Cluster'] = labels
    
    # Sort the dataframe by cluster for easier analysis
    x = x.sort_values(by='Cluster')
    
    # Save the clustered data to the specified output CSV file
    try:
        x.to_csv(output_csv, index=False)
        logger.info(f"Saved clustered data to {output_csv}")
    except Exception as e:
        logger.error(f"Error saving output CSV: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster terms using sentence embeddings and optimization.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--min_clusters', type=int, default=5, help='Minimum number of clusters.')
    parser.add_argument('--max_clusters', type=int, default=10, help='Maximum number of clusters.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for optimization.')
    parser.add_argument('--use_external_model', action='store_true', help='Use the external model instead of the local one.')
    args = parser.parse_args()
    
    run_clustering(
        args.input_csv, 
        args.output_csv, 
        args.min_clusters, 
        args.max_clusters, 
        args.n_trials,
        use_local_model=not args.use_external_model  # Default to local model unless specified
    )