import pandas as pd
import json
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a summary of the mesothelioma workflow results.")
    parser.add_argument("--enrichment_file", type=str, required=True, help="Path to the enrichment CSV file.")
    parser.add_argument("--topic_model_file", type=str, required=True, help="Path to the topic model CSV file.")
    parser.add_argument("--api_file", type=str, required=True, help="Path to the API calls CSV file.")
    parser.add_argument("--clustered_topics_file", type=str, required=True, help="Path to the clustered topics CSV file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    return parser.parse_args()

def read_data(enrichment_file, topic_model_file, api_file, clustered_topics_file):
    enrichment_df = pd.read_csv(enrichment_file)
    topic_model_df = pd.read_csv(topic_model_file)
    api_df = pd.read_csv(api_file)
    clustered_topics = pd.read_csv(clustered_topics_file)
    return enrichment_df, topic_model_df, api_df, clustered_topics

def calculate_metrics(enrichment_df, topic_model_df, api_df, clustered_topics):
    number_of_genes_considered = len(enrichment_df["gene_queried"].unique())
    documents_considered = enrichment_df.shape[0]
    unique_seeds = topic_model_df['seed'].nunique()
    max_topics_per_seed = topic_model_df.groupby('seed')['Topic'].max()
    average_topics = int(max_topics_per_seed.sum() / unique_seeds)
    min_max_topics = max_topics_per_seed.min()
    max_max_topics = max_topics_per_seed.max()
    api_calls_made = api_df.shape[0]
    number_of_themes_after_filtering = clustered_topics.shape[0]
    number_of_clusters = clustered_topics['Cluster'].nunique()
    compression_ratio = round(documents_considered / number_of_themes_after_filtering, 2)
    return {
        "number_of_genes_considered": number_of_genes_considered,
        "documents_considered": documents_considered,
        "average_topics": average_topics,
        "range_of_max_topics": f"{min_max_topics} to {max_max_topics}",
        "API_calls_made": api_calls_made,
        "number_of_themes_after_filtering": number_of_themes_after_filtering,
        "number_of_clusters": number_of_clusters,
        "compression_ratio": compression_ratio,
        "time_of_analysis": datetime.now().isoformat()
    }

def save_results(results, output_file):
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results have been saved to {output_file}")

def main():
    args = parse_arguments()
    enrichment_df, topic_model_df, api_df, clustered_topics = read_data(
        args.enrichment_file, args.topic_model_file, args.api_file, args.clustered_topics_file
    )
    results = calculate_metrics(enrichment_df, topic_model_df, api_df, clustered_topics)
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()