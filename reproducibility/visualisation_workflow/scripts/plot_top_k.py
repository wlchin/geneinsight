import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import argparse

# this is plot_top_k.py

# Set the aesthetics for a publication-style plot
def set_publication_style():
    # Set font to Arial (or a similar sans-serif)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Figure settings
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['figure.dpi'] = 300
    
    # Text settings
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.titlesize'] = 14  # Title font size
    plt.rcParams['axes.labelsize'] = 13  # Axis label font size
    plt.rcParams['xtick.labelsize'] = 12  # x-tick label font size
    plt.rcParams['ytick.labelsize'] = 12  # y-tick label font size
    plt.rcParams['legend.fontsize'] = 12  # Legend font size
    
    # Line settings
    plt.rcParams['axes.linewidth'] = 0.8  # Frame line width
    plt.rcParams['lines.linewidth'] = 1.0  # Line width
    plt.rcParams['lines.markersize'] = 4  # Marker size
    
    # Grid settings
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    
    # Legend settings
    plt.rcParams['legend.frameon'] = False  # No frame on legend
    
    # Remove top and right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def main(input_file, output_pdf, output_png, title=None, top_k_values=None, metric_column='recall_cosine'):
    # Read the data
    results_df = pd.read_csv(input_file)
    
    # Filter for only the specified top_k values
    desired_topk = top_k_values if top_k_values else [2, 5, 10, 25, 50]
    filtered_df = results_df[results_df['top_k'].isin(desired_topk)]
    
    # Set publication-style aesthetics
    set_publication_style()
    
    # Create color palette (using muted colors)
    # Using a color-blind friendly palette
    palette = sns.color_palette("colorblind", n_colors=len(desired_topk))
    
    # Create figure with two subplots: main plot and legend
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    
    # Main plot
    ax1 = plt.subplot(gs[0])
    
    # Plot markers and lines
    for i, k in enumerate(desired_topk):
        df_k = filtered_df[filtered_df['top_k'] == k]
        
        # Sort by source length for connecting lines
        df_k = df_k.sort_values('source_length')
        
        # Plot points with solid circles without margins
        ax1.scatter(df_k['source_length'], df_k[metric_column], 
                   s=25, color=palette[i], marker='o',  # Using 'o' for all
                   label=f'top-{k}', edgecolor='none', alpha=1.0)
        
        # No connecting lines between individual points
    
    # No trend lines as requested
    
    # Configure y-axis: range 0.2 to 1.0 with exactly 4 tick marks (0.2, 0.4, 0.6, 0.8, 1.0)
    ax1.set_ylim(0.2, 1.0)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Remove the tick marks on both y and x axes but keep labels
    ax1.tick_params(axis='both', which='both', length=0)
    
    # Configure x-axis
    ax1.set_xlabel('Source Document Length (number of terms)')
    ax1.set_ylabel(f'Top-k {"Sentence Similarity" if metric_column == "recall_sentence" else "Cosine Similarity"}')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Create custom legend subplot
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')  # Hide the axes
    
    # Create a custom legend with solid circles (no margins)
    legend_elements = []
    for i, k in enumerate(desired_topk):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color=palette[i], 
                                         markerfacecolor=palette[i], markeredgecolor='none',
                                         markersize=8, linestyle='',
                                         label=f'top-{k}'))
    
    # Place the legend in the second subplot
    ax2.legend(handles=legend_elements, loc='center', ncol=5, frameon=False)
    
    # Add a title
    default_title = f'Top-k {"Sentence Similarity" if metric_column == "recall_sentence" else "Cosine Similarity"} vs. Source Document Length'
    plot_title = title if title else default_title
    plt.suptitle(plot_title, fontsize=10, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
    
    # Save with high resolution
    plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
    
    # Also save as PNG for easy viewing
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    
    print(f"Plots saved as {output_pdf} and {output_png}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate publication-style plots for similarity data.')
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True, 
                        help='Path to the input CSV file containing the results data')
    
    # Optional arguments
    parser.add_argument('--output-pdf', '-p', default='publication_topk_similarity.pdf',
                        help='Path to save the output PDF file (default: %(default)s)')
    parser.add_argument('--output-png', '-g', default='publication_topk_similarity.png',
                        help='Path to save the output PNG file (default: %(default)s)')
    parser.add_argument('--title', '-t', default=None,
                        help='Plot title (default is based on the metric column)')
    parser.add_argument('--top-k', '-k', nargs='+', type=int, default=[2, 5, 10, 25, 50],
                        help='List of top-k values to include in the plot (default: %(default)s)')
    parser.add_argument('--metric-column', '-m', default='recall_cosine',
                        choices=['recall_cosine', 'recall_sentence'],
                        help='Metric column to plot (default: %(default)s)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with parsed arguments
    main(
        input_file=args.input,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        title=args.title,
        top_k_values=args.top_k,
        metric_column=args.metric_column
    )