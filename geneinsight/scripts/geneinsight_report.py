#!/usr/bin/env python3
"""
Standalone script for generating an interactive HTML report from TopicGenes results.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
import json
from datetime import datetime

from geneinsight.report.geneplotter import generate_heatmaps
from geneinsight.report.circleplot import generate_circle_plot
from geneinsight.report.summary import generate_json_summary
from geneinsight.report.rst_generator import generate_rst_files, generate_download_rst
from geneinsight.report.sphinx_builder import setup_sphinx_project

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_file(directory, pattern):
    """Find files matching a pattern in a directory."""
    matches = list(Path(directory).glob(pattern))
    if matches:
        return str(matches[0])
    return None

def ensure_directory(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory

def validate_results_dir(results_dir):
    """
    Validate that the results directory contains the necessary files.
    
    Returns:
        dict: Dictionary of paths to required files
    """
    required_files = {
        'enrichment': find_file(results_dir, '*__enrichment.csv'),
        'documents': find_file(results_dir, '*__documents.csv'),
        'topics': find_file(results_dir, '*_topics.csv'),
        'filtered_sets': find_file(results_dir, '*_filtered_*.csv'),
        'clustered_topics': find_file(results_dir, '*_clustered_topics.csv'),
        'api_results': find_file(results_dir, '*_api_results.csv')
    }
    
    missing_files = [key for key, value in required_files.items() if not value]
    
    if missing_files:
        logger.error(f"Missing required files in results directory: {', '.join(missing_files)}")
        logger.error("Please make sure you have run the full TopicGenes pipeline")
        return None
    
    return required_files

def copy_results_to_temp(results_files, temp_dir):
    """
    Copy result files to temporary directory with standardized names.
    
    Returns:
        dict: Dictionary of standardized paths in the temp directory
    """
    standard_paths = {}
    
    # Define standardized names
    name_mapping = {
        'enrichment': 'enrichment.csv',
        'documents': 'documents.csv',
        'topics': 'topics.csv',
        'filtered_sets': 'filtered_sets.csv',
        'clustered_topics': 'clustered_topics.csv',
        'api_results': 'api_results.csv'
    }
    
    # Copy files with standardized names
    for key, original_path in results_files.items():
        if original_path:
            standard_name = name_mapping.get(key, os.path.basename(original_path))
            standard_path = os.path.join(temp_dir, standard_name)
            shutil.copy2(original_path, standard_path)
            standard_paths[key] = standard_path
    
    return standard_paths

def generate_report(results_dir, output_dir, title=None, cleanup=True):
    """
    Generate an interactive HTML report from TopicGenes results.
    
    Args:
        results_dir: Directory containing TopicGenes results
        output_dir: Directory to store the generated report
        title: Title for the report (default: derived from gene set name)
        cleanup: Whether to clean up temporary files after building the report
        
    Returns:
        str: Path to the output directory or None if report generation failed
    """
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    ensure_directory(output_dir)
    
    # Create a temporary working directory
    temp_dir = os.path.join(output_dir, "temp")
    ensure_directory(temp_dir)
    
    try:
        # Get gene set name from results directory
        gene_set_name = os.path.basename(results_dir)
        if not gene_set_name:
            gene_set_name = "gene_set"
        
        # Use derived title if not provided
        if not title:
            title = f"TopicGenes Analysis: {gene_set_name}"
        
        logger.info(f"Generating report for gene set: {gene_set_name}")
        
        # Validate results directory
        results_files = validate_results_dir(results_dir)
        if not results_files:
            return None
        
        # Copy results to temp directory
        standard_paths = copy_results_to_temp(results_files, temp_dir)
        
        # Directory structure for report generation
        report_dirs = {
            'heatmaps': ensure_directory(os.path.join(temp_dir, "heatmaps")),
            'circle_plots': ensure_directory(os.path.join(temp_dir, "circle_plots")),
            'rst_outputs': ensure_directory(os.path.join(temp_dir, "rst_outputs")),
            'summary_json': ensure_directory(os.path.join(temp_dir, "summary_json")),
            'sphinx_build': ensure_directory(os.path.join(temp_dir, "sphinx_build")),
        }
        
        # Step 1: Generate JSON summary
        logger.info("Generating JSON summary...")
        json_file = os.path.join(report_dirs['summary_json'], f"{gene_set_name}.json")
        generate_json_summary(
            enrichment_file=standard_paths['enrichment'],
            topic_model_file=standard_paths['topics'],
            api_file=standard_paths['api_results'],
            clustered_topics_file=standard_paths['clustered_topics'],
            output_file=json_file
        )
        
        # Step 2: Generate circle plot
        logger.info("Generating circle plot...")
        circle_plot_file = os.path.join(report_dirs['circle_plots'], f"{gene_set_name}_circle_plot.html")
        generate_circle_plot(
            input_csv=standard_paths['clustered_topics'],
            headings_csv=find_file(results_dir, '*_headings.csv') or standard_paths['clustered_topics'],
            extra_vectors_csv=standard_paths['filtered_sets'],
            output_html=circle_plot_file
        )
        
        # Step 3: Generate heatmaps
        logger.info("Generating heatmaps...")
        heatmaps_log = os.path.join(report_dirs['heatmaps'], f"{gene_set_name}.log")
        merged_context_file = find_file(results_dir, '*_merged*.csv')
        if merged_context_file:
            generate_heatmaps(
                df_path=merged_context_file,
                save_folder=report_dirs['heatmaps'],
                log_file=heatmaps_log
            )
        
        # Step 4: Generate RST files
        logger.info("Generating RST documentation...")
        rst_dir = os.path.join(report_dirs['rst_outputs'], f"rst_{gene_set_name}")
        ensure_directory(rst_dir)
        
        # Generate main RST files
        headings_csv = find_file(results_dir, '*_headings.csv')
        merged_csv = find_file(results_dir, '*_merged*.csv')
        
        if headings_csv and merged_csv:
            generate_rst_files(
                headings_csv=headings_csv,
                merged_csv=merged_csv,
                filtered_genesets_csv=standard_paths['filtered_sets'],
                output_dir=rst_dir,
                log_file=os.path.join(rst_dir, "generation.log")
            )
            
            # Generate download RST
            generate_download_rst(
                csv_folder=os.path.join(rst_dir, "csv"),
                output_file=os.path.join(rst_dir, "download.rst")
            )
        
        # Step 5: Build Sphinx documentation
        logger.info("Building Sphinx documentation...")
        sphinx_log = os.path.join(report_dirs['sphinx_build'], f"{gene_set_name}.log")
        
        # Use package's default logo
        from pkg_resources import resource_filename
        logo_path = resource_filename('topicgenes.report', 'assets/logo.png')
        
        if not os.path.exists(logo_path):
            # Create a simple placeholder logo if the package logo is missing
            from PIL import Image, ImageDraw
            placeholder_logo = os.path.join(temp_dir, "logo.png")
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "TG", fill=(255, 255, 0))
            img.save(placeholder_logo)
            logo_path = placeholder_logo
        
        setup_sphinx_project(
            project_dir=os.path.join(output_dir, "html"),
            external_docs_dir=rst_dir,
            image_dir=report_dirs['heatmaps'],
            html_embedding_file=circle_plot_file,
            logo_path=logo_path,
            log_file=sphinx_log,
            project_title=title
        )
        
        logger.info(f"Report generation complete. Open {os.path.join(output_dir, 'html', 'build', 'html', 'index.html')} in a web browser to view.")
        
        return output_dir
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up temporary files if requested
        if cleanup:
            logger.info("Cleaning up temporary files...")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report from TopicGenes results"
    )
    
    parser.add_argument(
        "results_dir",
        help="Directory containing TopicGenes results"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./report",
        help="Directory to store the generated report (default: ./report)"
    )
    
    parser.add_argument(
        "--title",
        default="TopicGenes Report",
        help="Title for the report (default: TopicGenes Report)"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up temporary files after building the report"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    output_dir = generate_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        title=args.title,
        cleanup=not args.no_cleanup
    )
    
    if not output_dir:
        sys.exit(1)

if __name__ == "__main__":
    main()