import pandas as pd
import logging
import sys
import argparse
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_csv_data(file_path):
    logging.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)["description"].tolist()

def initialize_bertopic(documents, num_topics=10):
    logging.info(f"Initializing BERTopic model with {num_topics} topics")
    topic_model = BERTopic(nr_topics=num_topics)
    topics, probs = topic_model.fit_transform(documents)
    return topic_model, topics, probs

def initialize_kmeans_topic_model(documents, num_topics=10, ncomp=2, seed_value=0):
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

def initialize_model_and_fit_documents(documents, method="bertopic", num_topics=10, ncomp=2, seed_value=0):
    logging.info(f"Initializing model with method {method}")
    if method == "kmeans":
        return initialize_kmeans_topic_model(documents, num_topics=num_topics, ncomp=ncomp, seed_value=seed_value)
    elif method == "bertopic":
        return initialize_bertopic(documents, num_topics=num_topics)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'bertopic' and 'kmeans'.")

def run_topic_modeling_return_df(input_file, method="bertopic", num_topics=10, ncomp=2, seed_value=0):
    logging.info(f"Running topic modeling on {input_file} with method {method}, {num_topics} topics, {ncomp} components, and seed {seed_value}")
    topic_list = load_csv_data(input_file)
    model, topics, probs = initialize_model_and_fit_documents(
        topic_list, 
        method=method, 
        num_topics=num_topics, 
        ncomp=ncomp, 
        seed_value=seed_value
    )
    df_info = model.get_document_info(topic_list)
    return df_info

def run_multiple_seed_topic_modeling(input_file, output_file, method="bertopic",
                                     num_topics=10, ncomp=2, seed_value=0, n_samples=1):
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
            seed_value=seed
        )
        df_seed["seed"] = seed
        dfs.append(df_seed)
    
    final_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Saving concatenated results to {output_file}")
    final_df.to_csv(output_file, index=False)
    return final_df

def main():
    parser = argparse.ArgumentParser(description="Run topic modeling on filtered gene sets.")
    parser.add_argument("--input_file", default="results/filtered_gene_sets.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_file", default="results/final_topic_modeling_results.csv", help="Path to the output CSV file.")
    parser.add_argument("--method", choices=["bertopic", "kmeans"], default="bertopic", help="Topic modeling method.")
    parser.add_argument("--num_topics", type=int, default=None, help="Number of topics.")
    parser.add_argument("--ncomp", type=int, default=2, help="Number of components for dimensionality reduction (only for KMeans).")
    parser.add_argument("--seed_value", type=int, default=0, help="Initial seed value.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of topic models to run with different seeds.")
    
    args = parser.parse_args()

    logging.info(f"Reading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    terms = df["Term"].tolist()

    temp_input_file = "temp_terms.csv"
    pd.DataFrame({"description": terms}).to_csv(temp_input_file, index=False)

    final_df = run_multiple_seed_topic_modeling(
        input_file=temp_input_file,
        output_file=args.output_file,
        method=args.method,
        num_topics=args.num_topics,
        ncomp=args.ncomp,
        seed_value=args.seed_value,
        n_samples=args.n_samples
    )

    # remove the temp_input_file
    os.remove(temp_input_file)

    logging.info(f"Final topic modeling results saved to {args.output_file}")

if __name__ == "__main__":
    main()