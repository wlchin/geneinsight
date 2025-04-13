import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

def create_visualization():
    # Check if results directory exists, create if not
    os.makedirs("results", exist_ok=True)
    
    # Load the processed data
    try:
        final_df = pd.read_csv("results/topic_modelling_metrics.csv")
    except FileNotFoundError:
        print("Error: Could not find processed data file. Please run the data processing script first.")
        return
    
    # ----------------------------------------------------------------------
    # Set up visualization parameters for publication-quality figures
    # ----------------------------------------------------------------------
    
    # Set the aesthetic for a publication-quality figure with LARGER fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 14,            # Increased base font size
        'axes.labelsize': 16,       # Larger axis labels
        'axes.titlesize': 18,       # Larger titles
        'xtick.labelsize': 14,      # Larger tick labels
        'ytick.labelsize': 14,      # Larger tick labels
        'legend.fontsize': 14,      # Larger legend
        'axes.linewidth': 1.0,      # Slightly thicker axes lines
        'grid.linewidth': 0.5,      # Medium gridlines
        'lines.linewidth': 1.5,     # Thicker lines for better visibility
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,          # Enable grid by default
        'grid.alpha': 0.3,          # Subtle grid
        'figure.constrained_layout.use': True,  # Better layout handling
    })

    # Professional blue color for boxplots
    box_color = "#4C72B0"

    # ---------------------------------------------------------------------
    # Create a single vertical figure with all three plots
    # ---------------------------------------------------------------------
    
    # Create figure with specific size - larger to accommodate larger fonts
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), dpi=300)
    
    # Common x-axis values - sort numeric values to ensure proper ordering
    topic_runs = sorted(final_df['num_samples'].unique())
    
    # Plot 1: RBO (higher is better) - Similarity measure
    sns.boxplot(data=final_df, x="num_samples", y="rbo", ax=axes[0], 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    axes[0].set_title("Rank-Biased Overlap (RBO)", fontweight='bold')
    axes[0].set_ylabel("RBO Score")
    axes[0].set_ylim(0, 1.05)  # Slightly higher to give space at the top
    axes[0].grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    axes[0].set_xlabel("")  # Remove x-label, only keep on bottom plot
    
    # Add a reference line at RBO = 0.7 (commonly used threshold) without text
    axes[0].axhline(y=0.7, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot 2: Kendall Distance (lower is better) - Distance measure
    sns.boxplot(data=final_df, x="num_samples", y="kendall_distance", ax=axes[1], 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    axes[1].set_title("Kendall Distance", fontweight='bold')
    axes[1].set_ylabel("Distance")
    axes[1].grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    axes[1].set_xlabel("")  # Remove x-label, only keep on bottom plot
    
    # Plot 3: Kendall Tau (higher is better) - Correlation coefficient
    sns.boxplot(data=final_df, x="num_samples", y="kendall_tau", ax=axes[2], 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    axes[2].set_title("Kendall's τ (Tau)", fontweight='bold')
    axes[2].set_ylabel("Correlation")
    axes[2].set_ylim(-0.05, 1.05)  # Adjusted to show full range
    axes[2].grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    axes[2].set_xlabel("Number of Topic Modelling Runs", fontweight='bold')
    
    # Add a reference line at Tau = 0.7 (strong correlation) without text
    axes[2].axhline(y=0.7, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Improve all subplots
    for ax in axes:
        # Add light horizontal grid lines
        ax.grid(False, axis='x')  # No vertical gridlines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Add subtle connecting lines between medians to show trend
        if ax == axes[0]:
            metric = 'rbo'
        elif ax == axes[1]:
            metric = 'kendall_distance'
        else:
            metric = 'kendall_tau'
        
        # Calculate and plot median values with connecting lines
        medians = []
        for z in topic_runs:
            subset = final_df[final_df['num_samples'] == z]
            if not subset.empty:
                medians.append(subset[metric].median())
            else:
                medians.append(np.nan)  # Handle missing data
                
        # Plot connecting line with slightly thicker red line
        ax.plot(range(len(topic_runs)), medians, 'o-', color='#e74c3c', 
                linewidth=2.0, markersize=5, alpha=0.8, zorder=10)
        
        # Set tick parameters for larger ticks
        ax.tick_params(axis='both', which='major', length=6, width=1.2)
        ax.tick_params(axis='both', which='minor', length=3, width=0.8)
        
        # Make all spines visible but thicker
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            
        # Make sure x-axis ticks show all values
        ax.set_xticks(range(len(topic_runs)))
        ax.set_xticklabels(topic_runs)
        
        # Rotate x ticks if they're too crowded
        if len(topic_runs) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # More space between plots for larger fonts
    
    # Save with white background for publication
    plt.savefig("results/ranking_metrics.png", dpi=600, bbox_inches='tight')
    plt.savefig("results/ranking_metrics.pdf", bbox_inches='tight')
    
    print("Visualization complete. Files saved as 'ranking_metrics.png' and 'ranking_metrics.pdf'")

if __name__ == "__main__":
    create_visualization()