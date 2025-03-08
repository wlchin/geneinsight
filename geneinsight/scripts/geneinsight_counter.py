#!/usr/bin/env python3
"""
Standalone script for counting top terms in topic modeling results.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from geneinsight.analysis.counter import count_top_terms

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
        description="Count top terms in topic modeling results"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file"
    )
    
    parser.add_argument(
        "output_file",
        help="Path to the output CSV file"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top terms to output (default: all)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        logger.info(f"Counting top terms in {args.input_file}")
        
        result_df = count_top_terms(
            input_file=args.input_file,
            output_file=args.output_file,
            top_n=args.top_n
        )
        
        if result_df.empty:
            logger.warning("No top terms found")
        else:
            logger.info(f"Found {len(result_df)} top terms")
        
    except Exception as e:
        logger.error(f"Error counting top terms: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()