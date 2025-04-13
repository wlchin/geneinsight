import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging

# ----------------------------------------------------------------------
# 1. Configure logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("soft_cardinality_viz.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)

# Log start of the script
logging.info("Soft cardinality visualization script started.")

def create_visualization():
    # ----------------------------------------------------------------------
    # 2. Load Processed Data
    # ----------------------------------------------------------------------
    
    results_csv_path = os.path.join("results/cardinality", "all_gene_sets_average.csv")
    
    try:
        results_df = pd.read_csv(results_csv_path)
        logging.info(f"Loaded results data with {len(results_df)} entries.")
    except FileNotFoundError:
        logging.error(f"Could not find the results CSV file at {results_csv_path}")
        print(f"Error: Could not find processed data file at {results_csv_path}. Please run the data processing script first.")
        return
    
    # Extract the unique thresholds
    thresholds = sorted(results_df["threshold"].unique())
    logging.info(f"Found {len(thresholds)} unique thresholds: {thresholds}")

    # ----------------------------------------------------------------------
    # 3. Create Publication-Ready Visualization
    # ----------------------------------------------------------------------

    # Set publication-quality aesthetics with moderate font sizes
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 11,            # Moderate base font size
        'axes.labelsize': 12,       # Moderate axis labels
        'axes.titlesize': 13,       # Moderate titles
        'xtick.labelsize': 10,      # Moderate tick labels
        'ytick.labelsize': 10,      # Moderate tick labels
        'legend.fontsize': 10,      # Moderate legend
        'axes.linewidth': 1.0,      # Slightly thicker axes lines
        'grid.linewidth': 0.5,      # Medium gridlines
        'lines.linewidth': 1.5,     # Thicker lines for better visibility
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,          # Enable grid by default
        'grid.alpha': 0.3,          # Subtle grid
    })

    # Define a professional color for boxplots
    box_color = "#4C72B0"  # Professional blue

    # Create publication-quality figure - larger size to accommodate wider spacing
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)

    # Plot titles with improved typography and labeling format
    subplot_titles = [
        "A. Low Similarity (Threshold = 0.4)",
        "B. Moderate Similarity (Threshold = 0.6)", 
        "C. High Similarity (Threshold = 0.8)", 
        "D. Very High Similarity (Threshold = 0.9)"
    ]

    # Flatten the 2x2 axes grid for easier iteration
    axes = axes.flatten()

    # Create subplots
    for idx, th in enumerate(thresholds):
        ax = axes[idx]
        
        # Get subset for this threshold
        subset = results_df[results_df["threshold"] == th]
        
        # Create boxplot
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
        
#sample_size,threshold,avg_soft_cardinality

        # Add trendline connecting median values
        medians = [subset[subset["sample_size"]==x]["avg_soft_cardinality"].median() 
                for x in sorted(subset["sample_size"].unique())]
        x_values = range(len(sorted(subset["sample_size"].unique())))
        ax.plot(x_values, medians, 'o-', color='#e74c3c', linewidth=1.5, 
                markersize=4, alpha=0.8, zorder=10)
        
        # Set titles and labels
        ax.set_title(subplot_titles[idx], fontweight='bold', loc='left')
        ax.set_xlabel("Number of Topic Modelling Runs", fontweight='normal')
        ax.set_ylabel("Normalized Soft Cardinality", fontweight='normal')
        
        # Ensure all plot frames are closed and visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        
        # Add light horizontal grid lines but remove vertical ones
        ax.grid(False, axis='x')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Set tick parameters for larger ticks
        ax.tick_params(axis='both', which='major', length=4, width=1.0)
        
        # Ensure x-axis ticks show all values
        run_values = sorted(subset["sample_size"].unique())
        ax.set_xticks(range(len(run_values)))
        ax.set_xticklabels(run_values)

    # First create a tight layout to get automatic sizing
    plt.tight_layout()
    
    # Then adjust with MUCH more space between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.35)  # Significantly increased spacing

    # Ensure results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.info(f"Created directory: {results_dir}")

    # Save in publication-quality formats to results folder with white background
    result_png = os.path.join(results_dir, "soft_cardinality.png")
    result_pdf = os.path.join(results_dir, "soft_cardinality.pdf")
    plt.savefig(result_png, dpi=600, bbox_inches='tight')
    plt.savefig(result_pdf, bbox_inches='tight')

    logging.info(f"Publication-quality figures saved to {result_png} and {result_pdf}")

if __name__ == "__main__":
    create_visualization()
    logging.info("Visualization completed successfully.")
    print("Soft cardinality visualization complete. Files saved as 'soft_cardinality.png' and 'soft_cardinality.pdf' in the results folder.")