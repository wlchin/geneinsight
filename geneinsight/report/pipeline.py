"""
GeneInsight Pipeline Module

This module contains the main pipeline logic for the GeneInsight report generation process.
It orchestrates the entire workflow from input CSV files to the final Sphinx documentation.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

from .file_handlers import create_folder_structure, copy_input_files, copy_logo, copy_scripts
from .collect_context import generate_context
from .context_merge import merge_context_ontology
# Update imports to match your actual module structure
from .geneplotter import generate_heatmaps
from .circleplot import generate_circle_plot
from .summary import generate_json_summary
from .rst_generator import generate_rst_files, generate_summary_page, generate_download_rst, copy_data_files
from .sphinx_builder import build_sphinx_docs

def setup_logging(log_folder):
    """Set up logging with timestamped file and console output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("GeneInsight pipeline started")
    logging.info(f"Log file: {log_file}")
    
    return log_file

def run_command(cmd, description, cwd=None):
    """Run a command and log its output."""
    import subprocess
    
    logging.info(f"Running {description}...")
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True, cwd=cwd)
        logging.info(f"{description} completed successfully")
        return True, output
    except subprocess.CalledProcessError as e:
        logging.error(f"{description} failed with exit code {e.returncode}")
        logging.error(f"Command output: {e.output}")
        return False, e.output

def run_pipeline(
    input_folder, 
    output_folder, 
    gene_set,
    # New parameters to support multiple API services for context generation:
    context_service="openai", 
    context_api_key=None, 
    context_model="gpt-4o-mini", 
    context_base_url=None
):
    """
    Run the complete GeneInsight pipeline.
    
    Args:
        input_folder (str): Folder with the input files.
        output_folder (str): Folder where outputs will be saved.
        gene_set (str): Identifier for the gene set.
        context_service (str): Service to use for context generation ("openai" or "ollama").
        context_api_key (str): API key for the chosen service.
        context_model (str): Model to use for generation.
        context_base_url (str): Base URL for the API (if needed, e.g. for ollama).
    """
    start_time = time.time()
    logging.info(f"Starting GeneInsight pipeline for gene set: {gene_set}")
    logging.info(f"Using context generation service: {context_service}")
    
    # Define file paths
    paths = {
        "summary": os.path.join(output_folder, f"results/summary/{gene_set}.csv"),
        "enrichment": os.path.join(output_folder, f"results/enrichment_df/{gene_set}__enrichment.csv"),
        "topic_model": os.path.join(output_folder, f"results/topics_for_genelists/{gene_set}_topic_model.csv"),
        "minor_topics": os.path.join(output_folder, f"results/minor_topics/{gene_set}_minor_topics.csv"),
        "clustered_topics": os.path.join(output_folder, f"results/clustered_topics/{gene_set}_clustered_topics.csv"),
        "filtered_sets": os.path.join(output_folder, f"results/filtered_sets/{gene_set}_filtered_gene_sets.csv"),
        "headings": os.path.join(output_folder, f"results/context/{gene_set}_headings.csv"),
        "subheadings": os.path.join(output_folder, f"results/context/{gene_set}_subheadings.csv"),
        "ontology_dict": os.path.join(output_folder, f"results/ontology_dict/{gene_set}_ontology_dict.csv"),
        "merged": os.path.join(output_folder, f"results/merged_context_ontology/{gene_set}_merged.csv"),
        "heatmap_log": os.path.join(output_folder, f"results/heatmaps/{gene_set}.log"),
        "circle_plot": os.path.join(output_folder, f"results/circle_plots/{gene_set}_circle_plot.html"),
        "summary_json": os.path.join(output_folder, f"results/summary_json/{gene_set}.json"),
        "rst_folder": os.path.join(output_folder, f"results/rst_outputs/rst_{gene_set}"),
        "csv_folder": os.path.join(output_folder, f"results/csv_outputs/rst_{gene_set}"),
        "sphinx_log": os.path.join(output_folder, f"results/sphinx_builds/{gene_set}.log"),
        "sphinx_build": os.path.join(output_folder, f"results/sphinx_builds/html_build_{gene_set}")
    }
    
    try:
        # Step 0: Create folder structure and copy input files
        base_path = create_folder_structure(output_folder, gene_set)
        logging.info(f"Created folder structure at {base_path}")
        
        copy_input_files(input_folder, output_folder, gene_set)
        logo_path = copy_logo(output_folder)
        
        # Step 1: Generate context with additional API service parameters
        generate_context(
            summary_path=paths["summary"],
            clustered_topics_path=paths["clustered_topics"],
            output_headings_path=paths["headings"],
            output_subheadings_path=paths["subheadings"],
            service=context_service,
            api_key=context_api_key,
            model=context_model,
            base_url=context_base_url
        )
        
        # Step 2: Merge context and ontology
        merge_context_ontology(
            subheadings_path=paths["subheadings"],
            ontology_dict_path=paths["ontology_dict"],
            output_path=paths["merged"]
        )
        
        # Step 3: Generate visualizations
        generate_heatmaps(
            df_path=paths["merged"],
            save_folder=os.path.join(output_folder, f"results/heatmaps/{gene_set}"),
            log_file=paths["heatmap_log"]
        )
        
        generate_circle_plot(
            input_csv=paths["clustered_topics"],
            headings_csv=paths["headings"],
            output_html=paths["circle_plot"],
            extra_vectors_csv=paths["filtered_sets"]
        )
        
        # Step 4: Generate JSON summary
        generate_json_summary(
            enrichment_path=paths["enrichment"],
            topic_model_path=paths["topic_model"],
            minor_topics_path=paths["minor_topics"],
            clustered_topics_path=paths["clustered_topics"],
            output_path=paths["summary_json"]
        )
        
        # Step 5: Generate RST files
        generate_rst_files(
            headings_path=paths["headings"],
            merged_path=paths["merged"],
            filtered_sets_path=paths["filtered_sets"],
            output_dir=paths["rst_folder"],
            csv_folder=paths["csv_folder"]
        )
        
        generate_summary_page(
            output_folder=paths["rst_folder"],
            json_path=paths["summary_json"],
            html_path=paths["circle_plot"]
        )
        
        generate_download_rst(
            csv_folder=paths["csv_folder"],
            output_path=os.path.join(paths["rst_folder"], "download.rst")
        )
        
        copy_data_files(
            filtered_sets_path=paths["filtered_sets"],
            merged_path=paths["merged"],
            dest_folder=paths["rst_folder"]
        )
        
        # Step 6: Build Sphinx documentation
        build_sphinx_docs(
            rst_folder=paths["rst_folder"],
            image_folder=os.path.join(output_folder, f"results/heatmaps/{gene_set}"),
            output_folder=paths["sphinx_build"],
            log_path=paths["sphinx_log"],
            html_embedding_path=paths["circle_plot"],
            logo_path=logo_path
        )
        
        # Calculate total time
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Pipeline completed in {execution_time:.2f} seconds")
        
        # Output final location
        html_index = os.path.join(paths["sphinx_build"], "build", "html", "index.html")
        if os.path.exists(html_index):
            logging.info(f"Success! The Sphinx documentation is available at: {html_index}")
            return True, html_index
        else:
            logging.warning(f"Sphinx build may have had issues. Check {paths['sphinx_log']} for details.")
            return False, None
        
    except Exception as e:
        logging.error(f"Pipeline execution error: {str(e)}", exc_info=True)
        return False, str(e)
