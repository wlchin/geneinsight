#!/usr/bin/env python3
# plot_overlap.py

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import logging
from matplotlib import rcParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("Starting overlap plotting script")

# Load processed data
data_file = 'results/processed_data/stringdb_overlap_data.json'
logger.info(f"Loading data from {data_file}")

try:
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    means = data['means']
    errors = data['errors']
    
    logger.info(f"Loaded data with {len(categories)} categories")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Set up the figure with publication style
plt.style.use('default')
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.8
rcParams['axes.labelsize'] = 8
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7
rcParams['xtick.major.size'] = 0  # No tick marks
rcParams['ytick.major.size'] = 0  # No tick marks

# Create figure with specific size (in inches) for publication
fig, ax = plt.subplots(figsize=(3.5, 2.625))  # Typical publication ratio (4:3)

# Define color theme
color = '#005c7a'  # Blue
edge_color = '#004357'  # Darker shade for edge

# Create bar chart with error bars
x = np.arange(len(categories))
width = 0.6
bars = ax.bar(x, means, width, color=color, edgecolor=edge_color, linewidth=0.8, 
       capsize=3, yerr=errors, error_kw={'elinewidth': 0.8, 'capthick': 0.8})

# Remove top and right spines for publication style
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Remove grid lines
ax.grid(False)

# Set axis labels
ax.set_xlabel('Cosine Similarity Threshold')
ax.set_ylabel('Overlap with StringDB Terms')

# Set x-tick labels
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Set exactly 5 ticks on y-axis
y_max = max(means) + max(errors) + 0.01
ax.set_ylim(0, y_max)
ax.set_yticks(np.linspace(0, y_max, 5))

# Format y-ticks to have consistent decimal places
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Add a title with same font as other labels but slightly larger
ax.set_title('StringDB Term Overlap by Similarity Threshold', fontsize=9, pad=10)

# Display values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.005,
            f'{means[i]:.3f}', ha='center', va='bottom', fontsize=6)

# Adjust layout
fig.tight_layout()

# Save as high-resolution PNG and SVG to results folder
os.makedirs('results', exist_ok=True)  # Create results folder if it doesn't exist
output_png = 'results/stringdb_overlap_plot.png'
output_svg = 'results/stringdb_overlap_plot.svg'

plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_svg, bbox_inches='tight', format='svg')

logger.info(f"Plot saved as {output_png} and {output_svg}")

# Print summary statistics
logger.info("Summary Statistics:")
for i, category in enumerate(categories):
    logger.info(f"Threshold {category}: Mean = {means[i]:.4f}, SEM = {errors[i]:.4f}")