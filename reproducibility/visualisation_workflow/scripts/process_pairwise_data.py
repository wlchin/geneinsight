import os
import pandas as pd
import logging
import argparse
import time
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Process data for pairwise comparison plots')
    parser.add_argument('--filtered_dir', default="1000geneset_benchmark/results/filtered_sets",
                        help='Directory containing filtered gene sets')
    parser.add_argument('--enrichment_dir', default="1000geneset_benchmark/results/enrichment_df_listmode",
                        help='Directory containing enrichment data')
    parser.add_argument('--output_dir', default="results",
                        help='Directory for output files')
    parser.add_argument('--filtered_suffix', default="_filtered_gene_sets.csv",
                        help='Suffix for filtered gene set files')
    parser.add_argument('--enrichment_suffix', default="__enrichment.csv",
                        help='Suffix for enrichment files')
    return parser.parse_args()

def read_row_counts(directory, suffix):
    """Read all files in directory with suffix, return {gene_set: row_count}"""
    row_counts = {}
    if not os.path.isdir(directory):
        logging.warning(f"Directory not found -> {directory}")
        return row_counts
    
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            gene_set = filename.replace(suffix, "")
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            row_counts[gene_set] = len(df)
    
    logging.info(f"Read {len(row_counts)} files from {directory}")
    return row_counts

def prepare_comparison_data(counts_a, counts_b, label_a, label_b):
    """Prepare data for plotting from two sets of counts"""
    common_sets = set(counts_a.keys()).intersection(counts_b.keys())
    
    if not common_sets:
        logging.warning(f"No common gene sets found for {label_a} vs. {label_b}.")
        return None
    
    plot_data = pd.DataFrame({
        'gene_set': list(common_sets),
        f'{label_a}_rows': [counts_a[gs] for gs in common_sets],
        f'{label_b}_rows': [counts_b[gs] for gs in common_sets]
    })
    
    # Calculate correlation
    corr = plot_data[f'{label_a}_rows'].corr(plot_data[f'{label_b}_rows'])
    plot_data['correlation'] = corr
    
    return plot_data

def save_combined_data(output_dir):
    """Combine data from individual CSVs into a single reference file"""
    combined_data_path = os.path.join(output_dir, "all_pairwise_comparisons.csv")
    
    all_csvs = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 
               f.startswith('data_') and f != "all_pairwise_comparisons.csv"]
    
    if all_csvs:
        dfs = []
        for csv_file in all_csvs:
            df = pd.read_csv(os.path.join(output_dir, csv_file))
            comparison_name = os.path.splitext(csv_file)[0].replace('data_', '')
            df['comparison'] = comparison_name
            dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_csv(combined_data_path, index=False)
            logging.info(f"Saved combined data: {combined_data_path}")
            return combined_df
    
    return None

def process_pairwise_data(args):
    """Process all pairwise comparisons and save data"""
    from itertools import combinations
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    logging.info(f"Starting data processing at {datetime.datetime.now()}")
    
    # Read base counts without filtering
    filtered_counts = read_row_counts(args.filtered_dir, args.filtered_suffix)
    enr_counts = read_row_counts(args.enrichment_dir, args.enrichment_suffix)
    
    data_sources = {
        "Filtered": filtered_counts,
        "Enrichment": enr_counts,
    }
    
    comparison_data = {}
    
    # Generate pairwise comparison data
    logging.info("Processing pairwise comparisons")
    for (label_a, label_b) in combinations(data_sources.keys(), 2):
        counts_a = data_sources[label_a]
        counts_b = data_sources[label_b]
        
        plot_data = prepare_comparison_data(counts_a, counts_b, label_a, label_b)
        if plot_data is not None:
            comparison_key = f"{label_a.lower()}_vs_{label_b.lower()}"
            comparison_data[comparison_key] = plot_data
            
            # Save individual comparison data
            csv_path = os.path.join(args.output_dir, f"data_{comparison_key}.csv")
            plot_data.to_csv(csv_path, index=False)
            logging.info(f"Saved data: {csv_path}")
    
    # Save combined data
    combined_data = save_combined_data(args.output_dir)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Data processing completed in {elapsed_time:.2f} seconds")
    
    return comparison_data

def main():
    args = parse_args()
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "pairwise_processing.log")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    process_pairwise_data(args)

if __name__ == "__main__":
    main()