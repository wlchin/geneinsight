#!/usr/bin/env python3
"""
Command-line interface for the TopicGenes package.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from typing import Dict, Any, Optional

from .pipeline import Pipeline
from .__init__ import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "n_samples": 5,
    "num_topics": 10,
    "pvalue_threshold": 0.01,
    "api_service": "openai",
    "api_model": "gpt-4o-mini",
    "api_parallel_jobs": 4,
    "api_base_url": None,
    "target_filtered_topics": 25,
}

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported or invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TopicGenes: Topic modeling pipeline for gene sets with enrichment analysis"
    )
    
    parser.add_argument(
        "query_gene_set",
        help="Path to file containing query gene set"
    )
    
    parser.add_argument(
        "background_gene_list",
        help="Path to file containing background gene list"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory to store outputs (default: ./output)"
    )
    
    parser.add_argument(
        "-t", "--temp-dir",
        help="Directory for temporary files (default: system temp directory)"
    )
    
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        help=f"Number of topic models to run with different seeds (default: {DEFAULT_CONFIG['n_samples']})"
    )
    
    parser.add_argument(
        "-k", "--num-topics",
        type=int,
        help=f"Number of topics to extract in topic modeling (default: {DEFAULT_CONFIG['num_topics']})"
    )
    
    parser.add_argument(
        "-p", "--pvalue-threshold",
        type=float,
        help=f"Adjusted P-value threshold for filtering results (default: {DEFAULT_CONFIG['pvalue_threshold']})"
    )
    
    parser.add_argument(
        "--api-service",
        help=f"API service for topic refinement (default: {DEFAULT_CONFIG['api_service']})"
    )
    
    parser.add_argument(
        "--api-model",
        help=f"Model name for the API service (default: {DEFAULT_CONFIG['api_model']})"
    )
    
    parser.add_argument(
        "--api-parallel-jobs",
        type=int,
        help=f"Number of parallel API jobs (default: {DEFAULT_CONFIG['api_parallel_jobs']})"
    )
    
    parser.add_argument(
        "--api-base-url",
        help="Base URL for the API service"
    )
    
    parser.add_argument(
        "--target-filtered-topics",
        type=int,
        help=f"Target number of topics after filtering (default: {DEFAULT_CONFIG['target_filtered_topics']})"
    )
    
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not zip the output directory"
    )
    
    parser.add_argument(
        "-c", "--config",
        help="Path to configuration file (JSON or YAML)"
    )
    
    # Add report generation arguments
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate an HTML report after pipeline completion"
    )
    
    parser.add_argument(
        "--report-dir",
        default="./report",
        help="Directory to store the generated report (default: ./report)"
    )
    
    parser.add_argument(
        "--report-title",
        default=None,
        help="Title for the generated report (default: derived from gene set name)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"TopicGenes v{__version__}"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Load configuration from file if provided
    config = DEFAULT_CONFIG.copy()
    if args.config:
        try:
            loaded_config = load_config(args.config)
            config.update(loaded_config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    # Override config with command line arguments
    if args.n_samples is not None:
        config["n_samples"] = args.n_samples
    if args.num_topics is not None:
        config["num_topics"] = args.num_topics
    if args.pvalue_threshold is not None:
        config["pvalue_threshold"] = args.pvalue_threshold
    if args.api_service is not None:
        config["api_service"] = args.api_service
    if args.api_model is not None:
        config["api_model"] = args.api_model
    if args.api_parallel_jobs is not None:
        config["api_parallel_jobs"] = args.api_parallel_jobs
    if args.api_base_url is not None:
        config["api_base_url"] = args.api_base_url
    if args.target_filtered_topics is not None:
        config["target_filtered_topics"] = args.target_filtered_topics
    
    # Check if input files exist
    if not os.path.exists(args.query_gene_set):
        logger.error(f"Query gene set file not found: {args.query_gene_set}")
        sys.exit(1)
    if not os.path.exists(args.background_gene_list):
        logger.error(f"Background gene list file not found: {args.background_gene_list}")
        sys.exit(1)
    
    # Create and run the pipeline
    try:
        logger.info("Initializing pipeline...")
        pipeline = Pipeline(
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            n_samples=config["n_samples"],
            num_topics=config["num_topics"],
            pvalue_threshold=config["pvalue_threshold"],
            api_service=config["api_service"],
            api_model=config["api_model"],
            api_parallel_jobs=config["api_parallel_jobs"],
            api_base_url=config["api_base_url"],
            target_filtered_topics=config["target_filtered_topics"],
        )
        
        logger.info("Running pipeline...")
        output_path = pipeline.run(
            query_gene_set=args.query_gene_set,
            background_gene_list=args.background_gene_list,
            zip_output=not args.no_zip,
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results available at: {output_path}")
        
        # Generate report if requested
        if args.generate_report:
            logger.info("Generating HTML report...")
            
            # Import here to avoid importing unless needed
            try:
                from .scripts.geneinsight_report import generate_report
                
                # Derive report title from gene set name if not provided
                report_title = args.report_title
                if not report_title:
                    gene_set_name = os.path.splitext(os.path.basename(args.query_gene_set))[0]
                    report_title = f"TopicGenes Analysis: {gene_set_name}"
                
                # Generate the report
                results_dir = os.path.dirname(output_path) if os.path.isfile(output_path) else output_path
                report_path = generate_report(
                    results_dir=results_dir,
                    output_dir=args.report_dir,
                    title=report_title
                )
                
                if report_path:
                    logger.info(f"Report generated successfully at {report_path}")
                    logger.info(f"Open {os.path.join(report_path, 'html/build/html/index.html')} in a web browser to view.")
                else:
                    logger.error("Report generation failed.")
            except ImportError as e:
                logger.error(f"Could not generate report: {e}")
                logger.error("Make sure you have installed the report generation dependencies:")
                logger.error("pip install umap-learn plotly colorcet sphinx sphinx-rtd-theme pillow")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()