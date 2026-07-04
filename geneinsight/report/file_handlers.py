"""
File handlers module for the GeneInsight report package.

This module contains utilities for file operations such as:
- Creating the folder structure
- Copying input files
- Copying the logo
- Handling script files
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import pandas as pd

def create_folder_structure(output_folder, gene_set):
    """Create the required folder structure for the pipeline."""
    folders = {
        "results": {
            "summary": {},
            "enrichment_df": {},
            "topics_for_genelists": {},
            "minor_topics": {},
            "clustered_topics": {},
            "filtered_sets": {},
            "context": {},
            "ontology_dict": {},
            "merged_context_ontology": {},
            "heatmaps": {
                gene_set: {}
            },
            "circle_plots": {},
            "summary_json": {},
            "rst_outputs": {
                f"rst_{gene_set}": {}
            },
            "csv_outputs": {
                f"rst_{gene_set}": {}
            },
            "sphinx_builds": {
                f"html_build_{gene_set}": {}
            }
        },
        "logs": {}
    }
    
    def create_nested_folders(parent, structure):
        for folder, subfolders in structure.items():
            folder_path = os.path.join(parent, folder)
            os.makedirs(folder_path, exist_ok=True)
            if subfolders:  # If there are subfolders
                create_nested_folders(folder_path, subfolders)
    
    base_path = Path(output_folder)
    create_nested_folders(base_path, folders)
    
    return base_path

def copy_input_files(input_folder, output_folder, gene_set):
    """Copy input CSV files to their appropriate locations."""
    # Updated file mapping to include the proper paths in the results directory
    file_mapping = {
        "clustered.csv": f"results/clustered_topics/{gene_set}_clustered_topics.csv",
        "summary.csv": f"results/summary/{gene_set}.csv",
        "api_results.csv": f"results/minor_topics/{gene_set}_minor_topics.csv",
        "topics.csv": f"results/topics_for_genelists/{gene_set}_topic_model.csv",
        "enrichment.csv": f"results/enrichment_df/{gene_set}__enrichment.csv",
        "ontology_dict.csv": f"results/ontology_dict/{gene_set}_ontology_dict.csv",
        "enriched.csv": f"results/filtered_sets/{gene_set}_filtered_gene_sets.csv"
    }
    
    missing_files = []
    for src_name, dest_path in file_mapping.items():
        src_path = os.path.join(input_folder, src_name)
        if not os.path.exists(src_path):
            missing_files.append(src_name)
            continue
        
        dest_full_path = os.path.join(output_folder, dest_path)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)
        shutil.copy2(src_path, dest_full_path)
        logging.info(f"Copied {src_path} to {dest_full_path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing required input files: {', '.join(missing_files)}")

def copy_logo(output_folder):
    """Copy logo file to data directory or create a placeholder."""
    import os
    import shutil
    import logging

    data_dir = os.path.join(output_folder, "data")
    os.makedirs(data_dir, exist_ok=True)
    logo_path = os.path.join(data_dir, "GeneInsight.png")
    
    try:
        # Attempt to import the new location of the logo
        import geneinsight.report.assets
        package_dir = os.path.dirname(os.path.abspath(geneinsight.report.assets.__file__))
        package_logo_path = os.path.join(package_dir, "logo.png")
    except ImportError:
        # Fallback to the original package location if the new one is not available
        import geneinsight
        package_dir = os.path.dirname(os.path.abspath(geneinsight.__file__))
        package_logo_path = os.path.join(package_dir, "data", "GeneInsight.png")
    
    if os.path.exists(package_logo_path):
        shutil.copy2(package_logo_path, logo_path)
        logging.info(f"Copied logo from package to {logo_path}")
    else:
        # Create a simple placeholder using PIL if available
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (200, 80), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((20, 30), "GeneInsight", fill=(0, 0, 0))
            img.save(logo_path)
            logging.info(f"Created placeholder logo at {logo_path}")
        except (ImportError, Exception) as e:
            logging.warning(f"Could not create logo: {e}. Using a dummy file instead.")
            # Just create an empty file as placeholder
            with open(logo_path, 'wb') as f:
                f.write(b'PNG placeholder')
    
    return logo_path

def copy_scripts(output_folder, scripts_to_include):
    """Copy the required scripts to the output folder structure."""
    # In the package version, we don't need to copy scripts as they're part of the package
    logging.info("Scripts are part of the package - no copying needed")
    pass