import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import argparse
import optuna

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def find_best_params(embeddings, df, target_rows):
    def objective(trial):
        t = trial.suggest_float('threshold', 0.01, 0.99)
        c = trial.suggest_float('count_prop', 0.01, 0.99)
        used_indices = set()
        for i in range(len(embeddings)):
            if i in used_indices:
                continue
            for j in range(i+1, len(embeddings)):
                if 1 - cosine(embeddings[i], embeddings[j]) >= t:
                    used_indices.add(j)
        temp_df = df.iloc[list(set(range(len(df))) - used_indices)]
        max_count = temp_df['Count'].max()
        temp_df = temp_df[temp_df['Count'] >= c * max_count]
        return abs(len(temp_df) - target_rows)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    best_threshold = study.best_params['threshold']
    best_count_prop = study.best_params['count_prop']
    return best_threshold, best_count_prop

def main(input_csv, output_csv, target_rows):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Initialize the model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Embed the terms
    embeddings = model.encode(df['Term'].tolist())

    best_threshold, best_count_prop = find_best_params(embeddings, df, target_rows)

    # Filter terms based on cosine similarity
    used_indices = set()
    for i in range(len(embeddings)):
        if i in used_indices:
            continue
        for j in range(i + 1, len(embeddings)):
            if cosine_similarity(embeddings[i], embeddings[j]) >= best_threshold:
                used_indices.add(j)

    # Create a new DataFrame with the filtered terms and counts
    filtered_df = df.iloc[list(set(range(len(df))) - used_indices)]

    # Filter terms based on count proportion
    max_count = filtered_df['Count'].max()
    filtered_df = filtered_df[filtered_df['Count'] >= best_count_prop * max_count]

    # Save the filtered terms to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter terms by auto-optimized similarity.')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('output_csv', type=str, help='Path to the output CSV file')
    parser.add_argument('target_rows', type=int, help='Desired approximate number of filtered rows', nargs='?', default=100)
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.target_rows)