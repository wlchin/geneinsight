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
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def find_file(directory, pattern):
    """Find files matching a pattern in a directory."""
    logger.debug(f"Looking for files matching pattern '{pattern}' in directory: {directory}")
    
    matches = list(Path(directory).glob(pattern))
    
    if matches:
        logger.debug(f"Found matching files: {[str(m) for m in matches]}")
        return str(matches[0])
    else:
        logger.debug(f"No files matching pattern '{pattern}' found in directory")
        return None

def list_directory_contents(directory, max_depth=2, current_depth=0):
    """
    List all files and directories within a directory up to a maximum depth.
    
    Args:
        directory: Directory to list contents of
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth (used internally)
    """
    if current_depth > max_depth:
        return
    
    try:
        items = os.listdir(directory)
        
        for item in items:
            item_path = os.path.join(directory, item)
            
            if os.path.isdir(item_path):
                logger.debug(f"{'  ' * current_depth}[DIR] {item}")
                list_directory_contents(item_path, max_depth, current_depth + 1)
            else:
                file_size = os.path.getsize(item_path)
                logger.debug(f"{'  ' * current_depth}[FILE] {item} ({file_size} bytes)")
    except Exception as e:
        logger.debug(f"Error listing directory {directory}: {e}")

def ensure_directory(directory):
    """Ensure a directory exists."""
    logger.debug(f"Ensuring directory exists: {directory}")
    os.makedirs(directory, exist_ok=True)
    return directory

def validate_results_dir(results_dir):
    """
    Validate that the results directory contains the necessary files.
    
    Returns:
        dict: Dictionary of paths to required files
    """
    logger.info(f"Validating results directory: {results_dir}")
    logger.debug(f"Absolute path of results directory: {os.path.abspath(results_dir)}")
    
    # List all files in the results directory to help with debugging
    logger.debug("Contents of results directory:")
    list_directory_contents(results_dir)
    
    required_files = {
        'enrichment': find_file(results_dir, '*enrichment.csv'),
        'documents': find_file(results_dir, '*documents.csv'),
        'topics': find_file(results_dir, '*topics.csv'),
        'filtered_sets': find_file(results_dir, '*enriched.csv'),
        'clustered_topics': find_file(results_dir, '*clustered.csv'),
        'api_results': find_file(results_dir, '*api_results.csv')
    }
    
    # Log what we found
    logger.debug("Required files found:")
    for key, value in required_files.items():
        logger.debug(f"  {key}: {value}")
    
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
    logger.info(f"Copying results to temporary directory: {temp_dir}")
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
            logger.debug(f"Copying {original_path} -> {standard_path}")
            
            try:
                shutil.copy2(original_path, standard_path)
                standard_paths[key] = standard_path
                
                # Verify the file was copied successfully
                if os.path.exists(standard_path):
                    file_size = os.path.getsize(standard_path)
                    logger.debug(f"Successfully copied file, size: {file_size} bytes")
                else:
                    logger.warning(f"File copy failed, destination file does not exist")
            except Exception as e:
                logger.error(f"Error copying file: {e}")
    
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
    # Log system information
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    logger.info(f"Output directory (absolute path): {output_dir}")
    ensure_directory(output_dir)
    
    # Create a temporary working directory
    temp_dir = os.path.join(output_dir, "temp")
    ensure_directory(temp_dir)
    
    try:
        # Get gene set name from results directory
        gene_set_name = os.path.basename(results_dir)
        if not gene_set_name:
            gene_set_name = "gene_set"
        logger.info(f"Using gene set name: {gene_set_name}")
        
        # Use derived title if not provided
        if not title:
            title = f"TopicGenes Analysis: {gene_set_name}"
        logger.info(f"Report title: {title}")
        
        # Validate results directory
        results_files = validate_results_dir(results_dir)
        if not results_files:
            logger.error("Results directory validation failed")
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
        
        # Log directory structure
        logger.debug("Report directory structure:")
        for key, path in report_dirs.items():
            logger.debug(f"  {key}: {path}")
        
        # Step 1: Generate JSON summary
        logger.info("Generating JSON summary...")
        json_file = os.path.join(report_dirs['summary_json'], f"{gene_set_name}.json")
        logger.debug(f"JSON summary output file: {json_file}")
        
        try:
            logger.debug(f"Using enrichment file: {standard_paths.get('enrichment', 'NOT FOUND')}")
            logger.debug(f"Using topic model file: {standard_paths.get('topics', 'NOT FOUND')}")
            logger.debug(f"Using API file: {standard_paths.get('api_results', 'NOT FOUND')}")
            logger.debug(f"Using clustered topics file: {standard_paths.get('clustered_topics', 'NOT FOUND')}")
            
            generate_json_summary(
                enrichment_file=standard_paths['enrichment'],
                topic_model_file=standard_paths['topics'],
                api_file=standard_paths['api_results'],
                clustered_topics_file=standard_paths['clustered_topics'],
                output_file=json_file
            )
            
            if os.path.exists(json_file):
                file_size = os.path.getsize(json_file)
                logger.debug(f"JSON summary generated successfully, size: {file_size} bytes")
            else:
                logger.warning("JSON summary generation failed, output file does not exist")
        except Exception as e:
            logger.error(f"Error generating JSON summary: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Step 2: Generate circle plot
        logger.info("Generating circle plot...")
        circle_plot_file = os.path.join(report_dirs['circle_plots'], f"{gene_set_name}_circle_plot.html")
        logger.debug(f"Circle plot output file: {circle_plot_file}")
        
        # Look for headings file
        headings_file = find_file(results_dir, '*_headings.csv')
        logger.debug(f"Headings file found: {headings_file}")
        
        try:
            generate_circle_plot(
                input_csv=standard_paths['clustered_topics'],
                headings_csv=headings_file or standard_paths['clustered_topics'],
                extra_vectors_csv=standard_paths['filtered_sets'],
                output_html=circle_plot_file
            )
            
            if os.path.exists(circle_plot_file):
                file_size = os.path.getsize(circle_plot_file)
                logger.debug(f"Circle plot generated successfully, size: {file_size} bytes")
            else:
                logger.warning("Circle plot generation failed, output file does not exist")
        except Exception as e:
            logger.error(f"Error generating circle plot: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Step 3: Generate heatmaps
        logger.info("Generating heatmaps...")
        heatmaps_log = os.path.join(report_dirs['heatmaps'], f"{gene_set_name}.log")
        
        # Look for merged context file
        merged_context_file = find_file(results_dir, '*_merged*.csv')
        logger.debug(f"Merged context file found: {merged_context_file}")
        
        if merged_context_file:
            try:
                generate_heatmaps(
                    df_path=merged_context_file,
                    save_folder=report_dirs['heatmaps'],
                    log_file=heatmaps_log
                )
                
                # Check for generated heatmaps
                heatmap_files = list(Path(report_dirs['heatmaps']).glob('*.png'))
                logger.debug(f"Generated {len(heatmap_files)} heatmap files")
            except Exception as e:
                logger.error(f"Error generating heatmaps: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("Skipping heatmap generation - merged context file not found")
        
        # Step 4: Generate RST files
        logger.info("Generating RST documentation...")
        rst_dir = os.path.join(report_dirs['rst_outputs'], f"rst_{gene_set_name}")
        ensure_directory(rst_dir)
        
        # Generate main RST files
        headings_csv = find_file(results_dir, '*_headings.csv')
        merged_csv = find_file(results_dir, '*_merged*.csv')
        
        logger.debug(f"Headings CSV found: {headings_csv}")
        logger.debug(f"Merged CSV found: {merged_csv}")
        
        if headings_csv and merged_csv:
            try:
                generate_rst_files(
                    headings_csv=headings_csv,
                    merged_csv=merged_csv,
                    filtered_genesets_csv=standard_paths['filtered_sets'],
                    output_dir=rst_dir,
                    log_file=os.path.join(rst_dir, "generation.log")
                )
                
                # Check for generated RST files
                rst_files = list(Path(rst_dir).glob('*.rst'))
                logger.debug(f"Generated {len(rst_files)} RST files")
                
                # Generate download RST
                generate_download_rst(
                    csv_folder=os.path.join(rst_dir, "csv"),
                    output_file=os.path.join(rst_dir, "download.rst")
                )
                
                if os.path.exists(os.path.join(rst_dir, "download.rst")):
                    logger.debug("Download RST file generated successfully")
                else:
                    logger.warning("Download RST file generation failed")
            except Exception as e:
                logger.error(f"Error generating RST files: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("Skipping RST file generation - required files not found")
        
        # Step 5: Build Sphinx documentation
        logger.info("Building Sphinx documentation...")
        sphinx_log = os.path.join(report_dirs['sphinx_build'], f"{gene_set_name}.log")
        
        # Use package's default logo
        try:
            import importlib.resources as pkg_resources
            logo_path = str(pkg_resources.files('geneinsight.report').joinpath('assets/logo.png'))
            logger.debug(f"Using logo from package resources: {logo_path}")
        except Exception as e:
            logger.warning(f"Error loading logo from package resources: {e}")
            logo_path = None
        
        if not logo_path or not os.path.exists(logo_path):
            # Create a simple placeholder logo if the package logo is missing
            try:
                from PIL import Image, ImageDraw
                placeholder_logo = os.path.join(temp_dir, "logo.png")
                logger.debug(f"Creating placeholder logo at: {placeholder_logo}")
                
                img = Image.new('RGB', (100, 100), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.text((10, 10), "TG", fill=(255, 255, 0))
                img.save(placeholder_logo)
                logo_path = placeholder_logo
                
                if os.path.exists(logo_path):
                    logger.debug("Placeholder logo created successfully")
                else:
                    logger.warning("Placeholder logo creation failed")
            except Exception as e:
                logger.error(f"Error creating placeholder logo: {e}")
                logo_path = None
        
        try:
            html_output_dir = os.path.join(output_dir, "html")
            logger.debug(f"Setting up Sphinx project in: {html_output_dir}")
            
            setup_sphinx_project(
                project_dir=html_output_dir,
                external_docs_dir=rst_dir,
                image_dir=report_dirs['heatmaps'],
                html_embedding_file=circle_plot_file,
                logo_path=logo_path,
                log_file=sphinx_log,
                project_title=title
            )
            
            # Check if index.html was generated
            index_html = os.path.join(html_output_dir, 'build', 'html', 'index.html')
            if os.path.exists(index_html):
                logger.info(f"Sphinx documentation built successfully: {index_html}")
            else:
                logger.warning("Sphinx documentation build may have failed - index.html not found")
                
                # Check for sphinx log file to help with debugging
                if os.path.exists(sphinx_log):
                    with open(sphinx_log, 'r') as f:
                        log_content = f.read()
                    logger.debug(f"Sphinx build log:\n{log_content}")
        except Exception as e:
            logger.error(f"Error building Sphinx documentation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Report generation complete. Open {os.path.join(output_dir, 'html', 'build', 'html', 'index.html')} in a web browser to view.")
        
        return output_dir
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        # Clean up temporary files if requested
        if cleanup:
            logger.info(f"Cleaning up temporary files in: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report from TopicGenes results"
    )
    
    parser.add_argument(
        "--results-dir",
        required=True,
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
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    output_dir = generate_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        title=args.title,
        cleanup=not args.no_cleanup
    )
    
    if not output_dir:
        logger.error("Report generation failed")
        sys.exit(1)
    else:
        logger.info("Report generation succeeded")

if __name__ == "__main__":
    main()