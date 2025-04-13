#!/usr/bin/env python3
# process_overlap_data.py

import numpy as np
import pandas as pd
import os
import logging
import json
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("Starting overlap data processing script")

# Define file paths
base_dir = 'results/compute_soft_cardinality'
files = [
    'soft_cardinality_100_6.csv',
    'soft_cardinality_100_7.csv',
    'soft_cardinality_100_8.csv',
    'soft_cardinality_100_9.csv'
]

logger.info(f"Processing {len(files)} CSV files from directory: {base_dir}")

# Function to read CSV and extract the sets_enrichment_sc column
def read_csv_data(filename):
    file_path = os.path.join(base_dir, filename)
    logger.info(f"Reading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded file {filename} with shape: {df.shape}")
        # Check if the column exists
        if 'sets_enrichment_sc' in df.columns:
            data = df['sets_enrichment_sc'].values
            logger.info(f"Extracted {len(data)} values from 'sets_enrichment_sc' column")
            logger.info(f"Data summary for {filename}: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
            return data
        else:
            logger.warning(f"Warning: 'sets_enrichment_sc' column not found in {filename}")
            return np.array([])
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return np.array([])

# Read data from each file
data_list = []
for file in files:
    logger.info(f"Processing file: {file}")
    data = read_csv_data(file)
    data_list.append(data)
    logger.info(f"Added data from {file}, current data list length: {len(data_list)}")

# Calculate mean and SEM for each file
logger.info("Calculating statistics for each dataset")
means = []
errors = []
for i, data in enumerate(data_list):
    if len(data) > 0:
        # Drop NA values
        data_clean = data[~np.isnan(data)]
        na_count = len(data) - len(data_clean)
        if na_count > 0:
            logger.warning(f"Dropped {na_count} NA values from dataset {i} (file: {files[i]})")
        
        if len(data_clean) > 0:
            mean = np.mean(data_clean)
            sem = stats.sem(data_clean)
            means.append(mean)
            errors.append(sem)
            logger.info(f"Calculated mean={mean:.4f}, SEM={sem:.4f} for dataset using {len(data_clean)} values")
        else:
            logger.warning(f"Dataset {i} contains only NA values, setting mean and SEM to 0")
            means.append(0)
            errors.append(0)
    else:
        means.append(0)
        errors.append(0)
        logger.warning("Empty dataset encountered, setting mean and SEM to 0")

# Create categories labels (thresholds from filenames)
categories = ['0.6', '0.7', '0.8', '0.9']

# Print summary statistics
logger.info("Summary Statistics:")
for i, category in enumerate(categories):
    logger.info(f"Threshold {category}: Mean = {means[i]:.4f}, SEM = {errors[i]:.4f}")

# Save results to JSON for the plotting script
results = {
    'categories': categories,
    'means': means,
    'errors': errors
}

# Create results directory if it doesn't exist
os.makedirs('results/processed_data', exist_ok=True)

# Save to JSON file
output_file = 'results/processed_data/stringdb_overlap_data.json'
with open(output_file, 'w') as f:
    json.dump(results, f)

logger.info(f"Results saved to {output_file}")