import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import os
import numpy as np
import logging

def create_combined_visualization():
    """
    Creates a combined visualization with soft cardinality diagrams in the first two rows
    and Kendall rank visualization in the bottom three rows.
    """
    # ----------------------------------------------------------------------
    # Configure logging
    # ----------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("combined_visualization.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Combined visualization script started.")

    # ----------------------------------------------------------------------
    # Check if results directory exists, create if not
    # ----------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    
    # ----------------------------------------------------------------------
    # Load the processed data for both visualizations
    # ----------------------------------------------------------------------
    try:
        # Load Kendall rank data
        kendall_df = pd.read_csv("results/topic_modelling_metrics.csv")
        logging.info(f"Loaded Kendall rank data with {len(kendall_df)} entries.")
        
        # Load soft cardinality data - with new column names
        soft_card_df = pd.read_csv("results/soft_cardinality_results.csv")
        logging.info(f"Loaded soft cardinality data with {len(soft_card_df)} entries.")
    except FileNotFoundError as e:
        logging.error(f"Could not find processed data file: {e}")
        print(f"Error: Could not find processed data file. Please run the data processing scripts first.")
        return
    
    # ----------------------------------------------------------------------
    # Set up visualization parameters for publication-quality figures
    # ----------------------------------------------------------------------
    
    # Use consistent font sizes
    SMALL_SIZE = 10    # Axis ticks
    MEDIUM_SIZE = 12   # Axis labels
    LARGE_SIZE = 14    # Titles
    
    # Professional color palette
    box_color = "#4C72B0"  # Professional blue
    trend_color = "#E74C3C"  # Clear red for trend lines
    ref_line_color = "#2C3E50"  # Dark slate for reference lines
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': MEDIUM_SIZE,     # Base font size
        'axes.labelsize': MEDIUM_SIZE, # Axis labels
        'axes.titlesize': LARGE_SIZE,  # Titles
        'xtick.labelsize': SMALL_SIZE, # Tick labels
        'ytick.labelsize': SMALL_SIZE, # Tick labels
        'legend.fontsize': SMALL_SIZE, # Legend
        'axes.linewidth': 0.8,         # Thinner axes lines
        'grid.linewidth': 0.5,         # Gridlines
        'lines.linewidth': 1.5,        # Lines for better visibility
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,             # Enable grid by default
        'grid.alpha': 0.3,             # Subtle grid
        'figure.constrained_layout.use': True,  # Better layout handling
        'axes.spines.top': False,      # Remove top spine
        'axes.spines.right': False,    # Remove right spine
    })

    # ----------------------------------------------------------------------
    # Create the combined figure with GridSpec
    # ----------------------------------------------------------------------
    # Modified figure dimensions to create more square soft cardinality plots
    # Increase width relative to height
    fig = plt.figure(figsize=(8, 18), dpi=300)
    
    # Use more customized GridSpec with different width_ratios and height_ratios
    # to make the top 4 subplots more square
    gs = gridspec.GridSpec(5, 2, figure=fig, 
                          height_ratios=[1.2, 1.2, 1, 1, 1],  # Taller first two rows
                          width_ratios=[1, 1])  # Equal width columns
    
    # ----------------------------------------------------------------------
    # FIRST SECTION: Soft Cardinality Visualization (2 rows, 2 columns)
    # ----------------------------------------------------------------------
    # Extract the unique thresholds for soft cardinality
    thresholds = sorted(soft_card_df["threshold"].unique())
    logging.info(f"Found {len(thresholds)} unique thresholds: {thresholds}")
    
    # Subplot titles for soft cardinality with threshold info only
    thresholds_info = [
        {"threshold": 0.6, "title": "Threshold = 0.6"},
        {"threshold": 0.7, "title": "Threshold = 0.7"},
        {"threshold": 0.8, "title": "Threshold = 0.8"},
        {"threshold": 0.9, "title": "Threshold = 0.9"}
    ]
    
    # Create the soft cardinality subplots (first 2 rows)
    for idx, info in enumerate(thresholds_info):
        # Calculate row and column for 2x2 grid
        row = idx // 2
        col = idx % 2
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get subset for this threshold
        subset = soft_card_df[soft_card_df["threshold"] == info["threshold"]]
        
        # Create boxplot - using new column names
        sns.boxplot(
            x="sample_size", 
            y="avg_soft_cardinality", 
            data=subset, 
            ax=ax,
            color=box_color,
            width=0.6,
            fliersize=3,
            showfliers=True
        )
        
        # Add trendline connecting median values - using new column names
        medians = [subset[subset["sample_size"]==x]["avg_soft_cardinality"].median() 
                for x in sorted(subset["sample_size"].unique())]
        x_values = range(len(sorted(subset["sample_size"].unique())))
        ax.plot(x_values, medians, 'o-', color=trend_color, linewidth=1.5, 
                markersize=4, alpha=0.8, zorder=10)
        
        # Set labels and threshold title
        ax.set_title(info["title"], fontweight='bold')
        ax.set_xlabel("Number of Topic Modelling Runs", fontweight='normal')
        ax.set_ylabel("Normalized Soft Cardinality", fontweight='normal')
        
        # Style the subplot with only left and bottom spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(0.8)
        
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.tick_params(axis='both', which='major', length=4, width=1.0)
        
        # Ensure x-axis ticks show all values - using new column name
        run_values = sorted(subset["sample_size"].unique())
        ax.set_xticks(range(len(run_values)))
        ax.set_xticklabels(run_values)
        
        # Make the plot more square by adjusting the aspect ratio
        # This makes the plot appear more square visually
        ax.set_box_aspect(0.8)  # Values close to 1 make the plot more square
    
    # ----------------------------------------------------------------------
    # SECOND SECTION: Kendall Rank Visualization (3 rows, full width)
    # ----------------------------------------------------------------------
    # Common x-axis values for Kendall - sort numeric values to ensure proper ordering
    topic_runs = sorted(kendall_df['num_samples'].unique())
    
    # Create the Kendall rank subplots (last 3 rows, spanning all columns)
    
    # Plot 1: RBO (higher is better) - Similarity measure
    ax_rbo = fig.add_subplot(gs[2, :])
    sns.boxplot(data=kendall_df, x="num_samples", y="rbo", ax=ax_rbo, 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    ax_rbo.set_title("Rank-Biased Overlap (RBO)", fontweight='bold')
    ax_rbo.set_ylabel("RBO Score")
    ax_rbo.set_ylim(0, 1.05)  # Slightly higher to give space at the top
    ax_rbo.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax_rbo.set_xlabel("")  # Remove x-label, only keep on bottom plot
    
    # Plot 2: Kendall Distance (lower is better) - Distance measure
    ax_dist = fig.add_subplot(gs[3, :])
    sns.boxplot(data=kendall_df, x="num_samples", y="kendall_distance", ax=ax_dist, 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    ax_dist.set_title("Kendall Distance", fontweight='bold')
    ax_dist.set_ylabel("Distance")
    ax_dist.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax_dist.set_xlabel("")  # Remove x-label, only keep on bottom plot
    
    # Plot 3: Kendall Tau (higher is better) - Correlation coefficient
    ax_tau = fig.add_subplot(gs[4, :])
    sns.boxplot(data=kendall_df, x="num_samples", y="kendall_tau", ax=ax_tau, 
                color=box_color, width=0.6, fliersize=4, showfliers=True)
    ax_tau.set_title("Kendall's τ (Tau)", fontweight='bold')
    ax_tau.set_ylabel("Correlation")
    ax_tau.set_ylim(-0.05, 1.05)  # Adjusted to show full range
    ax_tau.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax_tau.set_xlabel("Number of Topic Modelling Runs", fontweight='bold')
    
    # Improve the Kendall rank subplots
    for ax, metric in [(ax_rbo, 'rbo'), (ax_dist, 'kendall_distance'), (ax_tau, 'kendall_tau')]:
        # Add light horizontal grid lines
        ax.grid(False, axis='x')  # No vertical gridlines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Calculate and plot median values with connecting lines
        medians = []
        for z in topic_runs:
            subset = kendall_df[kendall_df['num_samples'] == z]
            if not subset.empty:
                medians.append(subset[metric].median())
            else:
                medians.append(np.nan)  # Handle missing data
                
        # Plot connecting line with slightly thicker trend line
        ax.plot(range(len(topic_runs)), medians, 'o-', color=trend_color, 
                linewidth=1.5, markersize=4, alpha=0.8, zorder=10)
        
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
    
    # ----------------------------------------------------------------------
    # Adjust layout and save figure
    # ----------------------------------------------------------------------
    # Add a single unified legend at the bottom of the figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=box_color, lw=0, marker='s', markersize=10, 
               label='Boxplot Distribution', markerfacecolor=box_color),
        Line2D([0], [0], color=trend_color, lw=1.5, marker='o', markersize=4, 
               label='Median Trend'),
    ]
    
    # Add the legend at the bottom of the figure
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=SMALL_SIZE)
    
    # Improve layout with more precise control and significantly more space between plots
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend at bottom
    plt.subplots_adjust(hspace=0.6, wspace=0.3)  # Adjusted spacing between plots
    
    # Save with white background for publication in both vector and raster formats
    plt.savefig("results/combined_metrics.png", dpi=600, bbox_inches='tight', 
                transparent=False, facecolor='white')
    plt.savefig("results/combined_metrics.pdf", bbox_inches='tight', 
                transparent=False)
    plt.savefig("results/combined_metrics.svg", bbox_inches='tight', 
                transparent=False, format='svg')  # SVG for easy editing in Illustrator
    
    logging.info("Combined visualization complete. Files saved as 'combined_metrics.png' and 'combined_metrics.pdf'")
    print("Combined visualization complete. Files saved as 'combined_metrics.png' and 'combined_metrics.pdf'")

if __name__ == "__main__":
    create_combined_visualization()