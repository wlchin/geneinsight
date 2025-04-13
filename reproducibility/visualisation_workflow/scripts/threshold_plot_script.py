import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate threshold-based comparison plots')
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

# Set up publication-style formatting
def setup_plot_style():
    plt.rcParams.update({
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        
        # Figure
        'figure.figsize': (4.5, 4),
        'figure.dpi': 300,
        
        # Axes
        'axes.linewidth': 0.8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.spines.right': False,
        'axes.spines.top': False,
        
        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Grid
        'grid.linewidth': 0.4,
        'grid.alpha': 0.3,
        
        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,
        
        # Savefig
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def read_row_counts_with_filter(directory, suffix, filter_column, filter_threshold):
    """Read files and filter by threshold, return {gene_set: filtered_row_count}"""
    row_counts = {}
    if not os.path.isdir(directory):
        logging.warning(f"Directory not found -> {directory}")
        return row_counts
    
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            gene_set = filename.replace(suffix, "")
            path = os.path.join(directory, filename)
            df = pd.read_csv(path)
            # Filter rows
            if filter_column in df.columns:
                filtered_df = df[df[filter_column] < filter_threshold]
                row_counts[gene_set] = len(filtered_df)
            else:
                logging.warning(f"Column '{filter_column}' not found in {filename}")
                row_counts[gene_set] = 0
    
    logging.info(f"Read and filtered {len(row_counts)} files from {directory} with threshold {filter_threshold}")
    return row_counts

def plot_pairwise_counts_with_filtering(args):
    """Create 3x3 grid of plots comparing data at different thresholds"""
    thresholds = [0.001, 0.01, 0.05]
    filtered_column = "Adjusted P-value"
    enrichment_column = "fdr"
    
    start_time = time.time()
    logging.info(f"Starting threshold analysis at {datetime.datetime.now()}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Using thresholds: {thresholds}")
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    logging.info("Created figure with 3x3 grid of subplots")
    
    stats_data = []
    
    for i, filtered_th in enumerate(thresholds):
        for j, enr_th in enumerate(thresholds):
            logging.info(f"Processing threshold pair: Filtered < {filtered_th}, Enrichment < {enr_th}")
            
            filtered_counts_th = read_row_counts_with_filter(
                args.filtered_dir,
                args.filtered_suffix,
                filtered_column,
                filtered_th
            )
            
            enr_counts_th = read_row_counts_with_filter(
                args.enrichment_dir,
                args.enrichment_suffix,
                enrichment_column,
                enr_th
            )
            
            ax = axes[i, j]
            common_sets = set(filtered_counts_th.keys()).intersection(enr_counts_th.keys())
            logging.info(f"Found {len(common_sets)} common gene sets between filtered and enrichment")
            
            if not common_sets:
                logging.warning(f"No common sets found for threshold pair (F<{filtered_th}, E<{enr_th})")
                ax.text(0.5, 0.5, "No common sets", va='center', ha='center')
                ax.set_title(f"F<{filtered_th}, E<{enr_th}")
                
                stats_data.append({
                    'filtered_threshold': filtered_th,
                    'enrichment_threshold': enr_th,
                    'filtered_mean': np.nan,
                    'filtered_sem': np.nan,
                    'enrichment_mean': np.nan,
                    'enrichment_sem': np.nan,
                    'sample_size': 0
                })
                continue
            
            x_vals = [filtered_counts_th[gs] for gs in common_sets]
            y_vals = [enr_counts_th[gs] for gs in common_sets]
            
            filtered_mean = np.mean(x_vals) if x_vals else np.nan
            filtered_sem = np.std(x_vals, ddof=1) / np.sqrt(len(x_vals)) if len(x_vals) > 1 else np.nan
            enrichment_mean = np.mean(y_vals) if y_vals else np.nan
            enrichment_sem = np.std(y_vals, ddof=1) / np.sqrt(len(y_vals)) if len(y_vals) > 1 else np.nan
            
            stats_data.append({
                'filtered_threshold': filtered_th,
                'enrichment_threshold': enr_th,
                'filtered_mean': filtered_mean,
                'filtered_sem': filtered_sem,
                'enrichment_mean': enrichment_mean,
                'enrichment_sem': enrichment_sem,
                'sample_size': len(common_sets)
            })
            
            # Scatter
            ax.scatter(x_vals, y_vals, color='#2C5F93', s=15, alpha=0.7)
            
            # Correlation
            if len(x_vals) > 1:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                ax.text(0.05, 0.9, f"r={corr:.2f}", ha='left', va='top', transform=ax.transAxes, fontsize=7)
            
            # Trendline
            if len(common_sets) > 1:
                try:
                    fit = np.polyfit(x_vals, y_vals, 1)
                    fit_fn = np.poly1d(fit)
                    x_range = np.linspace(min(x_vals), max(x_vals), 100)
                    ax.plot(x_range, fit_fn(x_range), '--', color='#333333', linewidth=0.8)
                except Exception as e:
                    logging.error(f"Failed to add trendline: {str(e)}")
            
            ax.set_title(f"F<{filtered_th}, E<{enr_th}", fontsize=8)
            ax.set_xlabel("GeneInsight terms", fontsize=7)
            ax.set_ylabel("StringDB terms", fontsize=7)
            ax.grid(True, linestyle='-', alpha=0.3)
            
            # Make axes equal and ensure both start at zero
            max_val = max(max(x_vals or [0]), max(y_vals or [0]))
            padding = max_val * 0.1
            ax.set_xlim(0, max_val + padding)
            ax.set_ylim(0, max_val + padding)
            ax.set_aspect('equal')
    
    # Add title
    fig.suptitle("GeneInsight vs StringDB at different thresholds", fontsize=12)
    
    # Save the figure
    out_file = os.path.join(args.output_dir, "filtered_vs_enrichment_thresholds.pdf")
    plt.savefig(out_file, dpi=300, format='pdf')
    logging.info(f"Saved PDF figure: {out_file}")
    
    png_file = out_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300)
    logging.info(f"Saved PNG figure: {png_file}")
    
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = os.path.join(args.output_dir, "threshold_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    logging.info(f"Saved statistics to CSV: {stats_csv_path}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Threshold analysis completed in {elapsed_time:.2f} seconds")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "threshold_analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set up plot style
    setup_plot_style()
    
    logging.info("Starting threshold analysis script")
    
    # Generate threshold comparison plot
    plot_pairwise_counts_with_filtering(args)
    
    logging.info("Threshold analysis completed successfully")

if __name__ == "__main__":
    main()