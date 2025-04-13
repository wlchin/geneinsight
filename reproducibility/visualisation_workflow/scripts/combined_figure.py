import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import json
import seaborn as sns
from matplotlib import rcParams
import os

def setup_publication_style():
    """Set up plotting style to match publication guidelines with improved font sizes"""
    plt.style.use('default')  # Reset to default style
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    
    # INCREASED FONT SIZES
    rcParams['font.size'] = 11         # Increased from 10
    rcParams['axes.labelsize'] = 12    # Increased from 11
    rcParams['axes.titlesize'] = 13    # Increased from 12
    rcParams['xtick.labelsize'] = 10   # Increased from 9
    rcParams['ytick.labelsize'] = 10   # Increased from 9
    rcParams['legend.fontsize'] = 10   # Increased from 9
    
    # IMPROVED LINE WIDTHS
    rcParams['axes.linewidth'] = 1.0   # Increased from 0.8
    rcParams['xtick.major.width'] = 0.7  # Increased from 0.5
    rcParams['ytick.major.width'] = 0.7  # Increased from 0.5
    rcParams['xtick.major.size'] = 3.5   # Increased from 3
    rcParams['ytick.major.size'] = 3.5   # Increased from 3
    
    # Define a color palette - slightly more vibrant
    colors = ['#1F77B4', '#66A61E', '#A6761D', '#E7298A', '#D95F02', '#7570B3', '#E6AB02']
    return colors

def setup_panel_style(ax, remove_ticks=False, add_grid=False):
    """Apply consistent styling to each panel with improved aesthetics"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Improved spine and tick appearance
    ax.spines['bottom'].set_linewidth(1.0)  # Increased from default
    ax.spines['left'].set_linewidth(1.0)    # Increased from default
    
    if remove_ticks:
        ax.tick_params(axis='both', which='both', length=0)
    else:
        ax.tick_params(direction='out', length=4.5, width=0.7)  # Improved tick appearance
    
    # Optional grid for better readability
    if add_grid:
        ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.5)
        ax.set_axisbelow(True)  # Ensure grid is behind the data
    
    return ax

def create_panel_2a(ax, colors, data_path="results/data_filtered_vs_enrichment.csv"):
    """Create panel 2A: Pairwise comparison plot with updated labels"""
    # Load data from CSV
    data_df = pd.read_csv(data_path)
    
    x_column = "Filtered_rows"  # Will be "Terms from GeneInsight"
    y_column = "Enrichment_rows"  # Will be "Terms from StringDB"
    
    x_vals = data_df[x_column].values
    y_vals = data_df[y_column].values
    
    # Calculate correlation
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    
    # Create scatter plot with improved aesthetics
    ax.scatter(x_vals, y_vals, 
              color='#2C5F93', 
              edgecolor='none',
              s=35,  # Slightly larger points
              alpha=0.8)  # Slightly more opaque
    
    # Add correlation text with improved position and styling
    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
           fontsize=11, va='top', ha='left', fontweight='bold')
    
    # Set limits
    max_x = max(x_vals)
    max_y = max(y_vals)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Add padding
    x_padding = max_x * 0.1
    y_padding = max_y * 0.1
    ax.set_xlim(right=max_x + x_padding)
    ax.set_ylim(top=max_y + y_padding)
    
    # UPDATED LABELS as requested
    ax.set_xlabel("Terms from GeneInsight", fontsize=12)
    ax.set_ylabel("Terms from StringDB", fontsize=12)
    
    return ax

def create_panel_2b(ax, colors, data_path="results/processed_data/stringdb_overlap_data.json"):
    """Create panel 2B: StringDB Term Overlap bar plot with improved styling"""
    # Load data from JSON
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    means = data['means']
    errors = data['errors']
    
    # Create bar chart with error bars - improved styling
    x = np.arange(len(categories))
    width = 0.65  # Slightly wider
    color = '#005c7a'  # Blue color
    edge_color = '#004357'  # Darker shade for edge
    
    bars = ax.bar(x, means, width, color=color, edgecolor=edge_color, linewidth=1.0, 
          capsize=3.5, yerr=errors, error_kw={'elinewidth': 1.0, 'capthick': 1.0})
    
    # Set axis labels with improved font sizes
    ax.set_xlabel('Cosine Similarity Threshold', fontsize=12)
    ax.set_ylabel('Overlap with StringDB Terms', fontsize=12)
    
    # Set x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    
    # Set exactly 5 ticks on y-axis
    y_max = max(means) + max(errors) + 0.01
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max, 5))
    
    # Format y-ticks
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Display values on top of bars (improved formatting)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.005,
               f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9,
               fontweight='bold')  # Made bold for better visibility
    
    return ax

def create_panel_2c(ax, colors, data_path="results/avg_distances_100.csv"):
    """Create panel 2C: Violin plot with updated labels"""
    # Load data from CSV
    df = pd.read_csv(data_path)
    
    # Extract the required columns and drop any missing values
    sets_data = df['sets_avg_distance'].dropna().values
    enrichment_data = df['enrichment_avg_distance'].dropna().values
    
    # Define improved color palette
    main_blue = '#0072B5'    # Blue
    main_orange = '#BC3C29'  # Orange/red
    
    # Create violin plot with improved aesthetics
    positions = [1, 2]
    violin_parts = ax.violinplot(
        [sets_data, enrichment_data], 
        positions=positions,
        widths=0.7, 
        showmeans=False, 
        showmedians=False,
        showextrema=False
    )
    
    # Customize violin appearance - improved
    for i, pc in enumerate(violin_parts['bodies']):
        if i == 0:
            pc.set_facecolor(main_blue)
        else:
            pc.set_facecolor(main_orange)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)  # Increased from 0.8
        pc.set_alpha(0.8)
    
    # Add box plots inside violins with improved styling
    boxprops = dict(linestyle='-', linewidth=1.0, color='black')
    whiskerprops = dict(linestyle='-', linewidth=1.0, color='black')
    medianprops = dict(linestyle='-', linewidth=1.8, color='white')
    
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
    
    # Fill boxplots with improved colors
    for i, patch in enumerate(bp['boxes']):
        if i == 0:
            patch.set_facecolor(main_blue)
        else:
            patch.set_facecolor(main_orange)
    
    # UPDATED LABEL as requested
    ax.set_ylabel('Average Pairwise Distance', fontsize=12, labelpad=5)
    
    # UPDATED x-tick labels as requested
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['GeneInsight', 'StringDB'], fontsize=10)  # Matching the fontsize of other panels
    
    # Set fixed y-axis range from 0 to 1
    ax.set_ylim(0.2, 1)
    y_ticks = np.linspace(0.2, 1, 5)
    ax.set_yticks(y_ticks)
    
    return ax

def create_panel_2d(ax, colors, data_path="results/cosine_score.csv"):
    """Create panel 2D: Cosine Score bar plot with improved styling"""
    # Load data from CSV
    cosine_df = pd.read_csv(data_path)
    
    # Extract top_k values
    topk_values = [2, 5, 10, 25, 50]  # Fixed to these specific values
    
    # Extract timepoints (file labels) from column names
    mean_cols = [col for col in cosine_df.columns if col.endswith('_mean')]
    timepoints = sorted([int(col.split('_')[0]) for col in mean_cols])
    
    # Bar positioning parameters
    bar_width = 0.13
    bar_spacing = 0.03
    group_spacing = 0.3
    
    # Calculate the total width of a group including internal spacing
    group_width = (len(topk_values) * bar_width) + ((len(topk_values) - 1) * bar_spacing)
    
    # Calculate the x positions for each group
    x = np.arange(len(timepoints)) * (1 + group_spacing)
    
    # Use a more distinct colorblind palette
    cb_colors = sns.color_palette("colorblind", n_colors=len(topk_values))
    
    # Cosine Score Plot with improved styling
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
            color=cb_colors[i],
            capsize=2.5,  # Increased from 2
            edgecolor='black',  # Added edge color
            linewidth=0.5,  # Added edge linewidth
            error_kw={'elinewidth': 0.8, 'capthick': 0.8}  # Improved error bars
        )
    
    # Set labels and ticks with improved font sizes
    ax.set_ylabel('Cosine Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=10)
    ax.set_xlabel('Final Number of Themes', fontsize=12)
    
    # Set y-axis to have 5 ticks from 0 to 1
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 5))
    
    return ax

def create_panel_2e(ax, colors, data_path="results/mover_score.csv"):
    """Create panel 2E: Mover Score bar plot with improved styling"""
    # Load data from CSV
    mover_df = pd.read_csv(data_path)
    
    # Extract top_k values
    topk_values = [2, 5, 10, 25, 50]  # Fixed to these specific values
    
    # Extract timepoints (file labels) from column names
    mean_cols = [col for col in mover_df.columns if col.endswith('_mean')]
    timepoints = sorted([int(col.split('_')[0]) for col in mean_cols])
    
    # Bar positioning parameters
    bar_width = 0.13
    bar_spacing = 0.03
    group_spacing = 0.3
    
    # Calculate the total width of a group including internal spacing
    group_width = (len(topk_values) * bar_width) + ((len(topk_values) - 1) * bar_spacing)
    
    # Calculate the x positions for each group
    x = np.arange(len(timepoints)) * (1 + group_spacing)
    
    # Use a more distinct colorblind palette
    cb_colors = sns.color_palette("colorblind", n_colors=len(topk_values))
    
    # Mover Score Plot with improved styling
    for i, k in enumerate(topk_values):
        values = []
        errors = []
        
        for tp in timepoints:
            mean_col = f"{tp}_mean"
            sem_col = f"{tp}_sem"
            
            # Find the row for this top_k value
            row = mover_df[mover_df['top_k'] == k]
            
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
            color=cb_colors[i],
            capsize=2.5,  # Increased from 2
            edgecolor='black',  # Added edge color
            linewidth=0.5,  # Added edge linewidth
            error_kw={'elinewidth': 0.8, 'capthick': 0.8}  # Improved error bars
        )
    
    # Set labels and ticks with improved font sizes
    ax.set_ylabel('Movers Score', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=10)
    ax.set_xlabel('Final Number of Themes', fontsize=12)
    
    # Set y-axis to have 5 ticks from 0 to 1
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 5))
    
    return ax

def create_panel_2f(ax, colors, data_path="results/correlation_analysis.csv"):
    """Create panel 2F: Pearson correlation bar chart with updated labels"""
    # Load data from CSV
    data = pd.read_csv(data_path)
    
    # Ensure Label is treated as categorical/string for proper ordering
    data['Label'] = data['Label'].astype(str)
    
    # Blue palette
    palette = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]
    
    # Create the bar chart with improved styling
    x = np.arange(len(data))
    bars = ax.bar(
        x, 
        data['Pearson'], 
        width=0.7,
        color=palette[:len(data)],
        edgecolor='black',
        linewidth=0.7  # Increased from 0.5
    )
    
    # UPDATED AXIS LABELS as requested
    ax.set_xlabel('Final Number of Themes', fontsize=12, labelpad=5)
    ax.set_ylabel('Pearson correlation between\ncosine and mover scores', fontsize=12, labelpad=5)
    
    # Set y-axis range
    ax.set_ylim(0.85, 1.0)  # Adjust as needed based on data
    
    # Add x-tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(data['Label'], fontsize=10)
    
    # Add value labels on top of the bars - improved
    for i, bar in enumerate(bars):
        value = data.iloc[i]['Pearson']
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.005,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'  # Made bold for better visibility
        )
    
    return ax

def create_panel_2g(ax, colors, data_path="data/topk_data/cosine_score_results_100.csv"):
    """Create panel 2G: TopK vs Source Document Length plot with updated labels only"""
    # Read the data
    results_df = pd.read_csv(data_path)
    
    # Define desired top_k values
    desired_topk = [2, 5, 10, 25, 50]  # Fixed to these specific values
    filtered_df = results_df[results_df['top_k'].isin(desired_topk)]
    
    # Using a color-blind friendly palette - same as panels D and E
    palette = sns.color_palette("colorblind", n_colors=len(desired_topk))
    
    # Plot markers without connecting lines (as in original)
    for i, k in enumerate(desired_topk):
        df_k = filtered_df[filtered_df['top_k'] == k]
        
        # Sort by source length for consistency
        df_k = df_k.sort_values('source_length')
        
        # Plot points with solid circles - no connecting lines
        ax.scatter(df_k['source_length'], df_k['recall_cosine'], 
                  s=20, color=palette[i], marker='o',
                  label=f'top-{k}', edgecolor='none', alpha=1.0)
    
    # Configure y-axis: range 0.3 to 1.0 with exactly 5 tick marks
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.3, 0.45, 0.6, 0.75, 0.9])
    
    # UPDATED LABEL as requested
    ax.set_xlabel('Number of enriched terms before summarisation', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    
    # Remove the tick marks but keep labels
    ax.tick_params(axis='both', which='both', length=0)
    
    # No grid lines (as requested)
    
    return ax

def main():
    """Main function to create the combined figure with improved spacing"""
    # Create figure and gridspec layout - improved dimensions for better spacing
    fig = plt.figure(figsize=(14, 14))  # Increased height for better spacing
    
    # First create the outer GridSpec with 2 rows and improved spacing
    outer_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1.2], hspace=0.35)  # Increased from 0.3
    
    # Create the top GridSpec for the first two rows of plots with improved spacing
    top_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_gs[0], hspace=0.35, wspace=0.6)  # Increased from 0.3, 0.5
    
    # Create the bottom GridSpec for the spanning plot
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.6)  # Increased from 0.5
    
    # Set up improved style
    colors = setup_publication_style()
    
    # Top row panels
    # Panel 1: (Fig2A) - Pairwise comparison plot
    ax1 = fig.add_subplot(top_gs[0, 0])
    ax1 = setup_panel_style(ax1)
    ax1 = create_panel_2a(ax1, colors)
    
    # Panel 2: (Fig2B) - StringDB Term Overlap
    ax2 = fig.add_subplot(top_gs[0, 1])
    ax2 = setup_panel_style(ax2, remove_ticks=True)
    ax2 = create_panel_2b(ax2, colors)
    
    # Panel 3: (Fig2C) - Violin Plot
    ax3 = fig.add_subplot(top_gs[0, 2])
    ax3 = setup_panel_style(ax3, remove_ticks=True)
    ax3 = create_panel_2c(ax3, colors)
    
    # Middle row panels
    # Panel 4: (Fig2D) - Cosine Score Comparison
    ax4 = fig.add_subplot(top_gs[1, 0])
    ax4 = setup_panel_style(ax4, remove_ticks=True)
    ax4 = create_panel_2d(ax4, colors)
    
    # Panel 5: (Fig2E) - Movers Score Comparison
    ax5 = fig.add_subplot(top_gs[1, 1])
    ax5 = setup_panel_style(ax5, remove_ticks=True)
    ax5 = create_panel_2e(ax5, colors)
    
    # Panel 6: (Fig2F) - Pearson Correlation
    ax6 = fig.add_subplot(top_gs[1, 2])
    ax6 = setup_panel_style(ax6)
    ax6 = create_panel_2f(ax6, colors)
    
    # Bottom row panel (spanning all three columns)
    # Panel 7: (Fig2G) - TopK vs Source Document Length
    ax7 = fig.add_subplot(bottom_gs[0, :])
    ax7 = setup_panel_style(ax7, remove_ticks=True)  # Removed grid
    ax7 = create_panel_2g(ax7, colors)
    
    # Create common legend for panels D, E and G with more space and moved upward
    # Use the same topK values across all panels
    topk_values = [2, 5, 10, 25, 50]
    cb_colors = sns.color_palette("colorblind", n_colors=len(topk_values))
    
    # Create a separate axis for the legend - adjusted position for the new layout
    legend_ax = fig.add_axes([0.35, 0.37, 0.3, 0.05])  # Slightly adjusted position
    legend_ax.set_axis_off()
    
    # Create custom handles for the legend
    from matplotlib.patches import Rectangle
    custom_handles = [Rectangle((0,0), 1, 1, color=cb_colors[i]) 
                     for i in range(len(topk_values))]
    custom_labels = [f'TopK = {k}' for k in topk_values]
    
    # Add a legend with improved styling
    legend = legend_ax.legend(
        custom_handles, 
        custom_labels, 
        loc='center',
        ncol=min(5, len(topk_values)), 
        frameon=True,
        fontsize=11,  # Increased from 10
        edgecolor='black',
        borderaxespad=1.5,
        title="Panels D, E & G"
    )
    legend.get_title().set_fontsize(12)  # Increased from 11
    legend.get_title().set_fontweight('bold')
    
    # Add more padding around the legend box
    legend.get_frame().set_linewidth(1.0)  # Increased from 0.8
    
    # Adjust the figure layout with improved spacing
    # Don't use tight_layout() as it will affect the legend position
    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.07)
    
    # Ensure the output directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save the figure in high resolution
    plt.savefig("results/combined_figure.png", dpi=600, bbox_inches='tight')
    plt.savefig("results/combined_figure.pdf", bbox_inches='tight')
    
    print("Combined figure saved to results directory.")
    
if __name__ == "__main__":
    main()