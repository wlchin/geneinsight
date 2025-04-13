#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse

# this is plot_violin.py

def create_publication_violin(csv_file, output_file, dpi=600):
    """
    Creates a publication-quality violin plot in modern scientific style.
    
    Parameters:
    -----------
    csv_file : str
        Path to the input CSV file containing the data
    output_file : str
        Path to save the output figure
    dpi : int
        Resolution of the output figure
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the required columns and drop any missing values
    sets_data = df['sets_avg_distance'].dropna().values
    enrichment_data = df['enrichment_avg_distance'].dropna().values
    
    # Set up the publication style
    plt.rcParams.update({
        # Font settings
        'font.family': 'Arial',
        'font.sans-serif': 'Arial',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        
        # Line settings
        'axes.linewidth': 0.8,
        'lines.linewidth': 0.8,
        'patch.linewidth': 0.8,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': dpi,
        
        # Layout settings
        'figure.constrained_layout.use': True,
    })
    
    # Define color palette
    colors = {
        'main_blue': '#0072B5',  # Blue
        'main_orange': '#BC3C29'  # Orange/red
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 3.5))  # Column width for publication
    
    # Create violin plot
    positions = [1, 2]
    violin_parts = ax.violinplot(
        [sets_data, enrichment_data], 
        positions=positions,
        widths=0.7, 
        showmeans=False, 
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin appearance
    for i, pc in enumerate(violin_parts['bodies']):
        if i == 0:
            pc.set_facecolor(colors['main_blue'])
        else:
            pc.set_facecolor(colors['main_orange'])
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
        pc.set_alpha(0.8)
    
    # Add box plots inside violins for statistical information
    boxprops = dict(linestyle='-', linewidth=0.8, color='black')
    whiskerprops = dict(linestyle='-', linewidth=0.8, color='black')
    medianprops = dict(linestyle='-', linewidth=1.5, color='white')
    
    bp = ax.boxplot(
        [sets_data, enrichment_data],
        positions=positions,
        widths=0.15,
        patch_artist=True,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        showcaps=False,
        showfliers=False
    )
    
    # Fill boxplots with the same color as violins
    for i, patch in enumerate(bp['boxes']):
        if i == 0:
            patch.set_facecolor(colors['main_blue'])
        else:
            patch.set_facecolor(colors['main_orange'])
    
    # Add scatter points for individual data (optional for small datasets)
    if len(sets_data) < 30 and len(enrichment_data) < 30:
        for i, data in enumerate([sets_data, enrichment_data]):
            # Add jitter to x position
            x = np.random.normal(positions[i], 0.05, size=len(data))
            ax.scatter(x, data, s=4, color='black', alpha=0.6, edgecolor=None, zorder=3)
    
    # Compute and display statistics
    for i, data in enumerate([sets_data, enrichment_data]):
        # Calculate mean
        mean = np.mean(data)
        
        # Add mean line (optional)
        ax.hlines(mean, positions[i]-0.3, positions[i]+0.3, colors='black', linestyles='dashed', linewidth=0.8)
    
    # Set axis labels
    ax.set_ylabel('Average Distance', labelpad=5)
    
    # Set x-tick labels with custom font
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['GeneInsight\nAvg Distance', 'StringDB API\nAvg Distance'])
    
    # Remove top and right spines (publication style)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # MODIFICATIONS: Set y-axis from 0 to 1 with exactly 5 ticks
    
    # Set fixed y-axis range from 0 to 1
    y_min, y_max = 0, 1
    
    # Set exactly 5 ticks on y-axis (0, 0.25, 0.5, 0.75, 1)
    y_ticks = np.linspace(y_min, y_max, 5)
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    
    # MODIFICATIONS: Remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    # MODIFICATIONS: Remove grid lines completely
    ax.yaxis.grid(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=False)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Create a publication-quality violin plot."
    )
    parser.add_argument(
        '--csv_file', type=str, required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        '--output_file', type=str, required=True, help="Path to save the output violin plot image."
    )
    parser.add_argument(
        '--dpi', type=int, default=600, help="Resolution of the output image (default: 600)."
    )
    args = parser.parse_args()

    create_publication_violin(args.csv_file, args.output_file, args.dpi)
    print(f"Figure saved to {args.output_file}")

if __name__ == '__main__':
    main()