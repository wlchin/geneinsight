import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import optuna
import argparse

def run_clustering(input_csv: str, output_csv: str, min_clusters: int = 5, max_clusters: int = 10, n_trials: int = 100) -> None:
    """
    Cluster terms using sentence embeddings and optimization.

    Args:
        input_csv (str): Path to the input CSV file containing a 'Term' column.
        output_csv (str): Path where the output CSV (with cluster labels) will be saved.
        min_clusters (int, optional): Minimum number of clusters to try. Defaults to 5.
        max_clusters (int, optional): Maximum number of clusters to try. Defaults to 10.
        n_trials (int, optional): Number of trials for optimization. Defaults to 100.
    """
    # Read the input CSV
    x = pd.read_csv(input_csv)
    
    # (Optional) Filter topics with low count if needed
    # max_count = x['Count'].max()
    # x = x[x['Count'] >= 0.5 * max_count]

    # Load the MiniLM model for sentence embeddings
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.encode(x['Term'].tolist())
    
    def objective(trial):
        # Suggest a clustering algorithm and number of clusters
        clustering_algorithm = trial.suggest_categorical('clustering_algorithm', ['hierarchical', 'kmeans', 'spectral'])
        N = trial.suggest_int('n_clusters', min_clusters, max_clusters)
        
        if clustering_algorithm == 'hierarchical':
            cluster_model = AgglomerativeClustering(n_clusters=N)
        elif clustering_algorithm == 'kmeans':
            cluster_model = KMeans(n_clusters=N)
        elif clustering_algorithm == 'spectral':
            cluster_model = SpectralClustering(n_clusters=N, affinity='nearest_neighbors')
        
        labels = cluster_model.fit_predict(embeddings)
        # Compute scores: lower is better
        db_score = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        return db_score - ch_score

    # Optimize clustering parameters using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    optimal_clustering_algorithm = study.best_params['clustering_algorithm']
    optimal_N = study.best_params['n_clusters']
    
    print(f'Optimal clustering algorithm: {optimal_clustering_algorithm}')
    print(f'Optimal number of clusters: {optimal_N}')
    
    # Cluster the terms using the optimal parameters
    if optimal_clustering_algorithm == 'hierarchical':
        cluster_model = AgglomerativeClustering(n_clusters=optimal_N)
    elif optimal_clustering_algorithm == 'kmeans':
        cluster_model = KMeans(n_clusters=optimal_N)
    elif optimal_clustering_algorithm == 'spectral':
        cluster_model = SpectralClustering(n_clusters=optimal_N, affinity='nearest_neighbors')
    
    labels = cluster_model.fit_predict(embeddings)
    x['Cluster'] = labels
    
    # Sort the dataframe by cluster for easier analysis
    x = x.sort_values(by='Cluster')
    
    # Save the clustered data to the specified output CSV file
    x.to_csv(output_csv, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster terms using sentence embeddings and optimization.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--min_clusters', type=int, default=5, help='Minimum number of clusters.')
    parser.add_argument('--max_clusters', type=int, default=10, help='Maximum number of clusters.')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for optimization.')
    args = parser.parse_args()
    
    run_clustering(args.input_csv, args.output_csv, args.min_clusters, args.max_clusters, args.n_trials)