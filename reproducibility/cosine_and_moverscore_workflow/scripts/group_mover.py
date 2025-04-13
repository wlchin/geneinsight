import pandas as pd
import glob
import re
import argparse
import os

def main(folder, pattern, output_md):
    # Construct the full glob pattern.
    glob_pattern = os.path.join(folder, pattern)
    files = glob.glob(glob_pattern)
    
    if not files:
        print(f"No files found matching pattern: {glob_pattern}")
        return

    results = {}

    for file in files:
        # Extract a numeric label from the filename (e.g., 25, 50, etc.)
        match = re.search(r"(\d+)", os.path.basename(file))
        if not match:
            print(f"Warning: No numeric label found in {file}. Skipping.")
            continue
        file_label = int(match.group(1))
        
        df = pd.read_csv(file)
        grouped = df.groupby("top_k")["recall_sentence"]
        mean_recall = grouped.mean()
        sem_recall = grouped.sem()
        
        # Combine mean and SEM into a formatted string for each top_k.
        combined = pd.Series({k: f"{mean_recall[k]:.3f} ± {sem_recall[k]:.3f}" 
                              for k in mean_recall.index})
        results[file_label] = combined

    # Create a DataFrame where rows are the file labels and columns are the top_k values.
    table = pd.DataFrame(results).T
    table.sort_index(inplace=True)
    table = table.reindex(sorted(table.columns), axis=1)
    
    # Convert the DataFrame to a markdown table.
    markdown_table = table.to_markdown()
    
    with open(output_md, "w") as f:
        f.write(markdown_table)
    
    print(f"Markdown table written to {output_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a markdown table of average recall_sentence ± SEM per top_k from CSV files."
    )
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Folder containing the CSV files."
    )
    parser.add_argument(
        "--pattern", type=str, required=True,
        help="File pattern (e.g. 'mover_score_results_*.csv')."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output markdown file path."
    )
    
    args = parser.parse_args()
    main(args.folder, args.pattern, args.output)
