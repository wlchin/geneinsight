#!/usr/bin/env python3
"""
Standalone script for performing hypergeometric enrichment analysis.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from topicgenes.enrichment.hypergeometric import hypergeometric_enrichment

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
        description="Perform hypergeometric enrichment analysis"
    )
    
    parser.add_argument(
        "summary_csv",
        help="Path to CSV file containing summary data"
    )
    
    parser.add_argument(
        "gene_origin",
        help="Path to file containing gene origin list"
    )
    
    parser.add_argument(
        "background_genes",
        help="Path to file containing background gene list"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        default="./output/enriched.csv",
        help="Path to output CSV file (default: ./output/enriched.csv)"
    )
    
    parser.add_argument(
        "-p", "--pvalue",
        type=float,
        default=0.01,
        help="P-value threshold for filtering results (default: 0.01)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input files exist
    for file_path, file_name in [
        (args.summary_csv, "Summary CSV"),
        (args.gene_origin, "Gene origin"),
        (args.background_genes, "Background genes")
    ]:
        if not os.path.exists(file_path):
            logger.error(f"{file_name} file not found: {file_path}")
            sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    try:
        logger.info(f"Performing hypergeometric enrichment analysis")
        logger.info(f"Using gene origin: {args.gene_origin}")
        logger.info(f"Using background genes: {args.background_genes}")
        logger.info(f"P-value threshold: {args.pvalue}")
        
        # Run hypergeometric enrichment
        result_df = hypergeometric_enrichment(
            df_path=args.summary_csv,
            gene_origin_path=args.gene_origin,
            background_genes_path=args.background_genes,
            output_csv=args.output_file,
            pvalue_threshold=args.pvalue
        )
        
        if result_df.empty:
            logger.warning("No significant results found")
        else:
            logger.info(f"Found {len(result_df)} significant results")
        
        logger.info(f"Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error performing hypergeometric enrichment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
