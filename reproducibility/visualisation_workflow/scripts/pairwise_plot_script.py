import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import time
import datetime

# This script is for generate_pairwise_plots_Fig2A


def parse_args():
    parser = argparse.ArgumentParser(description='Generate pairwise comparison plots')
    parser.add_argument('--output_dir', default="results",
                        help='Directory for output files')
    return parser.parse_args()

def setup_plot_style():
    """Set up Nature Biotechnology style"""
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

def create_pairwise_plot(data_df, label_a, label_b, output_file, output_dir):
    """Create and save scatter plot for pairwise comparison"""
    if data_df is None or data_df.empty:
        logging.warning(f"No data available for {label_a} vs. {label_b} plot.")
        return
    
    x_column = f"{label_a}_rows"
    y_column = f"{label_b}_rows"
    
    if x_column not in data_df.columns or y_column not in data_df.columns:
        logging.warning(f"Required columns not found in data for {label_a} vs. {label_b}.")
        return
    
    x_vals = data_df[x_column].values
    y_vals = data_df[y_column].values
    
    # Get correlation value from the dataframe if available, or calculate it
    if 'correlation' in data_df.columns:
        corr = data_df['correlation'].iloc[0]
    else:
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
    
    fig, ax = plt.subplots()
    
    ax.scatter(x_vals, y_vals, 
               color='#2C5F93', 
               edgecolor='none',
               s=30,  # Increased point size
               alpha=0.7)
    
    # Larger font for correlation text
    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
            fontsize=10, va='top', ha='left', fontweight='bold')
    
    # Set exactly 5 ticks on each axis
    max_x = max(x_vals)
    max_y = max(y_vals)
    ax.set_xticks(np.linspace(0, max_x, 5))
    ax.set_yticks(np.linspace(0, max_y, 5))
    
    if max(x_vals) > 1000 or max(y_vals) > 1000:
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    
    # Increased font sizes
    ax.set_xlabel(f"{label_a}", fontsize=12)
    ax.set_ylabel(f"{label_b}", fontsize=12)
    ax.set_title(f"{label_a} vs. {label_b}", fontsize=14, fontweight='bold')
    
    # Tick label font size
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    x_padding = max(x_vals) * 0.1
    y_padding = max(y_vals) * 0.1
    ax.set_xlim(right=max(x_vals) + x_padding)
    ax.set_ylim(top=max(y_vals) + y_padding)
    
    if abs(max(x_vals) - max(y_vals)) / max(max(x_vals), max(y_vals)) < 0.5:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    full_path = os.path.join(output_dir, output_file)
    plt.savefig(full_path, dpi=300, format='pdf')
    plt.savefig(full_path.replace('.pdf', '.png'), dpi=300)
    plt.close()
    logging.info(f"Saved: {full_path}")

def generate_all_plots(output_dir):
    """Generate plots for all data files in the output directory"""
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 
                 f.startswith('data_') and f != "all_pairwise_comparisons.csv"]
    
    for data_file in data_files:
        try:
            comparison_name = os.path.splitext(data_file)[0].replace('data_', '')
            parts = comparison_name.split('_vs_')
            
            if len(parts) == 2:
                label_a, label_b = parts[0].capitalize(), parts[1].capitalize()
                data_path = os.path.join(output_dir, data_file)
                
                df = pd.read_csv(data_path)
                plot_file = f"scatter_{comparison_name}.pdf"
                
                create_pairwise_plot(df, label_a, label_b, plot_file, output_dir)
            
        except Exception as e:
            logging.error(f"Error generating plot for {data_file}: {str(e)}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "pairwise_plots.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    start_time = time.time()
    logging.info(f"Starting pairwise plotting at {datetime.datetime.now()}")
    
    # Set up plot style
    setup_plot_style()
    
    # Generate plots for all processed data
    generate_all_plots(args.output_dir)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Pairwise plotting completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()