import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# this is cosine_plot_script.py

def plot_cosine_scores(cosine_csv_path, output_prefix="cosine_score_comparison"):
    """
    Create bar plots from CSV files for cosine scores.
    
    Parameters:
    - cosine_csv_path: Path to the cosine scores CSV file
    - output_prefix: Prefix for output filenames
    """
    # Set the style to match publication manuscript format
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['font.size'] = 13  # Increased base font size
    mpl.rcParams['axes.linewidth'] = 0.8
    mpl.rcParams['xtick.major.width'] = 0  # Remove tick marks
    mpl.rcParams['ytick.major.width'] = 0  # Remove tick marks
    mpl.rcParams['xtick.major.size'] = 0   # Remove tick marks
    mpl.rcParams['ytick.major.size'] = 0   # Remove tick marks

    # Load data from CSV files
    cosine_df = pd.read_csv(cosine_csv_path)
    
    # Extract top_k values, which are in the first column
    topk_values = cosine_df['top_k'].tolist()
    
    # Extract timepoints (file labels) from column names
    # Column format is expected to be like "25_mean", "50_mean", etc.
    mean_cols = [col for col in cosine_df.columns if col.endswith('_mean')]
    timepoints = sorted([int(col.split('_')[0]) for col in mean_cols])
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    # Bar positioning parameters
    bar_width = 0.13  # Narrower bars
    bar_spacing = 0.03  # Space between bars within a group
    group_spacing = 0.3  # Space between groups (increased)
    
    # Calculate the total width of a group including internal spacing
    group_width = (len(topk_values) * bar_width) + ((len(topk_values) - 1) * bar_spacing)
    
    # Calculate the x positions for each group
    # Using wider spacing between groups
    x = np.arange(len(timepoints)) * (1 + group_spacing)
    
    # Use seaborn's colorblind palette (same as in the first script)
    colors = sns.color_palette("colorblind", n_colors=len(topk_values))

    # Cosine Score Plot
    for i, k in enumerate(topk_values):
        values = []
        errors = []
        
        for tp in timepoints:
            mean_col = f"{tp}_mean"
            sem_col = f"{tp}_sem"
            
            # Find the row for this top_k value
            row = cosine_df[cosine_df['top_k'] == k]
            
            if not row.empty:
                values.append(row[mean_col].values[0])
                errors.append(row[sem_col].values[0])
            else:
                values.append(0)
                errors.append(0)
        
        # Calculate position for this bar within the group
        bar_positions = x + (i * (bar_width + bar_spacing)) - (group_width / 2) + (bar_width / 2)
        
        ax.bar(
            bar_positions, 
            values, 
            bar_width, 
            yerr=errors,
            color=colors[i],
            capsize=3,
            edgecolor='none',  # Remove edges for solid bars
            linewidth=0
        )
        
    ax.set_ylabel('Cosine Score', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=14)
    ax.set_xlabel('Final Number of Themes', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis to have 5 ticks from 0 to 1
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 5))  # 5 ticks: 0, 0.25, 0.5, 0.75, 1.0
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size

    # Create custom handles for the legend
    custom_handles = [Rectangle((0,0), 1, 1, color=colors[i]) 
                     for i in range(len(topk_values))]
    custom_labels = [f'TopK = {k}' for k in topk_values]

    # Add a legend to the figure with a box around it and much larger font
    legend = fig.legend(
        custom_handles, 
        custom_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=min(5, len(topk_values)), 
        frameon=True,  # Add frame around legend
        fontsize=15,    # Increased font size for legend
        edgecolor='black',
        borderaxespad=0.8
    )

    # Make the legend box border black and thicker
    legend.get_frame().set_linewidth(1.0)

    # Add extra buffer space at the bottom for the legend
    plt.subplots_adjust(bottom=0.28)

    # Save figure at high quality
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_prefix}.svg', bbox_inches='tight')
    
    print(f"Cosine score plot saved with prefix: {output_prefix}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plot from cosine score CSV file")
    parser.add_argument("--cosine", required=True, help="Path to cosine score CSV file")
    parser.add_argument("--output", default="cosine_score_comparison", help="Output file prefix")
    
    args = parser.parse_args()
    
    plot_cosine_scores(args.cosine, args.output)