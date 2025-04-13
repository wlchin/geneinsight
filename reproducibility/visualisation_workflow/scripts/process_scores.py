import pandas as pd
import glob
import re
import argparse
import os

def main(folder, pattern, output_md, output_csv=None):
    # Construct the full glob pattern.
    glob_pattern = os.path.join(folder, pattern)
    files = glob.glob(glob_pattern)
    
    if not files:
        print(f"No files found matching pattern: {glob_pattern}")
        return

    results = {}
    # Store raw results for CSV output
    raw_results = {}

    for file in files:
        # Extract numeric label from the filename (e.g., 25, 50, etc.)
        match = re.search(r"(\d+)", os.path.basename(file))
        if not match:
            print(f"Warning: No numeric label found in {file}. Skipping.")
            continue
        file_label = int(match.group(1))
        
        # Read the CSV file.
        df = pd.read_csv(file)
        
        # Determine which recall metric to use
        recall_metric = "recall_cosine" if "recall_cosine" in df.columns else "recall_sentence"
        
        if recall_metric not in df.columns:
            print(f"Warning: Neither recall_cosine nor recall found in {file}. Skipping.")
            continue
            
        # Group by top_k and calculate the mean and SEM of the chosen recall metric
        grouped = df.groupby("top_k")[recall_metric]

        mean_recall = grouped.mean()
        sem_recall = grouped.sem()
        
        # Store raw mean and SEM values for CSV output
        raw_results[file_label] = {
            "mean": mean_recall,
            "sem": sem_recall
        }
        
        # Combine the mean and SEM into a formatted string for markdown
        combined = pd.Series({
            k: f"{mean_recall[k]:.3f} ± {sem_recall[k]:.3f}" 
            for k in mean_recall.index
        })
        results[file_label] = combined

    # Assemble a DataFrame with rows as file labels and columns as top_k values.
    if not results:
        print("No valid data found in any of the files.")
        return
        
    # Create markdown table
    table = pd.DataFrame(results).T
    table.sort_index(inplace=True)
    table = table.reindex(sorted(table.columns), axis=1)
    markdown_table = table.to_markdown()

    # Write the markdown table to the specified output file.
    with open(output_md, "w") as f:
        f.write(markdown_table)
    print(f"Markdown table written to {output_md}")
    
    # Generate CSV output if requested
    if output_csv:
        # Create dataframes for means and SEMs
        all_means = {}
        all_sems = {}
        
        for file_label, data in raw_results.items():
            for top_k in data["mean"].index:
                if top_k not in all_means:
                    all_means[top_k] = {}
                    all_sems[top_k] = {}
                all_means[top_k][file_label] = data["mean"][top_k]
                all_sems[top_k][file_label] = data["sem"][top_k]
        
        # Create CSV with proper structure
        csv_rows = []
        
        # Header row with file labels
        file_labels = sorted(raw_results.keys())
        header = ["top_k"] + [f"{label}_mean" for label in file_labels] + [f"{label}_sem" for label in file_labels]
        csv_rows.append(header)
        
        # Data rows
        for top_k in sorted(all_means.keys()):
            row = [top_k]
            # Add mean values
            for label in file_labels:
                row.append(all_means[top_k].get(label, ""))
            # Add SEM values
            for label in file_labels:
                row.append(all_sems[top_k].get(label, ""))
            csv_rows.append(row)
        
        # Write to CSV file
        with open(output_csv, "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        
        print(f"CSV data written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a markdown table and optional CSV of average recall metrics ± SEM per top_k from CSV files."
    )
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing the CSV files.")
    parser.add_argument("--pattern", type=str, required=True,
                        help="File pattern (e.g. 'cosine_score_results_*.csv').")
    parser.add_argument("--output", type=str, required=True,
                        help="Output markdown file path.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional output CSV file path.")
    
    args = parser.parse_args()
    main(args.folder, args.pattern, args.output, args.csv)