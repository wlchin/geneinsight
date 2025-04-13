import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# this is pearson_barchart.py

def create_pearson_bar_chart(input_csv, output_figure, dpi=300, figure_width=6, figure_height=5):
    """
    Create a bar chart of Pearson correlation scores from the correlation analysis CSV.
    Uses a publication-styled theme.
    
    Parameters:
    -----------
    input_csv : str
        Path to the CSV file containing correlation analysis results
    output_figure : str
        Path where the output figure will be saved
    dpi : int
        Resolution of the output figure
    figure_width : float
        Width of the figure in inches
    figure_height : float
        Height of the figure in inches
    """
    # Load the data
    data = pd.read_csv(input_csv)
    
    # Ensure Label is treated as categorical/string for proper ordering
    data['Label'] = data['Label'].astype(str)
    
    # Set up the publication-inspired theme
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    
    # Blue palette colors
    palette = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]
    
    # Create the bar chart
    bars = sns.barplot(
        x='Label', 
        y='Pearson', 
        data=data,
        palette=palette[:len(data)],
        width=0.7,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.set_ylim(0.85, 1.0)  # Adjust as needed based on your data
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add value labels on top of the bars
    for i, bar in enumerate(bars.patches):
        value = data.iloc[i]['Pearson']
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.005,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_figure, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Bar chart saved to: {output_figure}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a bar chart of Pearson correlation scores.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with correlation analysis results.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output figure file path (e.g., pearson_barchart.png).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI of the output figure. Default: 300")
    parser.add_argument("--width", type=float, default=6,
                        help="Width of the figure in inches. Default: 6")
    parser.add_argument("--height", type=float, default=5,
                        help="Height of the figure in inches. Default: 5")
    
    args = parser.parse_args()
    
    create_pearson_bar_chart(
        args.input, 
        args.output, 
        dpi=args.dpi, 
        figure_width=args.width, 
        figure_height=args.height
    )