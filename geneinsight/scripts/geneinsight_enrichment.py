#!/usr/bin/env python3
"""
Standalone script for running StringDB gene enrichment.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

from geneinsight.enrichment.stringdb import process_gene_enrichment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run StringDB gene enrichment analysis"
    )
    
    parser.add_argument(
        "gene_set",
        help="Path to file containing gene set"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory to store outputs (default: ./output)"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["single", "list"],
        default="single",
        help="Query mode: 'list' for batch query, 'single' for individual gene queries (default: single)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.gene_set):
        logger.error(f"Gene set file not found: {args.gene_set}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        logger.info(f"Running StringDB enrichment for {args.gene_set} in {args.mode} mode")
        
        # Run StringDB enrichment
        enrichment_df, documents = process_gene_enrichment(
            input_file=args.gene_set,
            output_dir=args.output_dir,
            mode=args.mode
        )
        
        # Get output file paths
        input_basename = os.path.splitext(os.path.basename(args.gene_set))[0]
        enrichment_output = os.path.join(args.output_dir, f"{input_basename}__enrichment.csv")
        documents_output = os.path.join(args.output_dir, f"{input_basename}__documents.csv")
        
        logger.info(f"Enrichment results saved to {enrichment_output}")
        logger.info(f"Documents saved to {documents_output}")
        
    except Exception as e:
        logger.error(f"Error running StringDB enrichment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
