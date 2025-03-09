#!/usr/bin/env python3
"""
Standalone script for running topic modeling on filtered gene sets (meta-analysis).
"""

import os
import sys
import argparse
import logging
from typing import Optional

from geneinsight.models.meta import run_multiple_seed_topic_modeling

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
        description="Run topic modeling on filtered gene sets"
    )
    
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the input CSV file with filtered gene sets"
    )
    
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output CSV file"
    )
    
    parser.add_argument(
        "--method",
        choices=["bertopic", "kmeans"],
        default="bertopic",
        help="Topic modeling method (default: bertopic)"
    )
    
    parser.add_argument(
        "--num-topics",
        type=int,
        default=None,
        help="Number of topics to extract (default: auto-determined)"
    )
    
    parser.add_argument(
        "--ncomp",
        type=int,
        default=2,
        help="Number of components for dimensionality reduction (default: 2)"
    )
    
    parser.add_argument(
        "--seed-value",
        type=int,
        default=0,
        help="Initial random seed value (default: 0)"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of models to run with different seeds (default: 5)"
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
        logger.info(f"Running topic modeling on filtered gene sets")
        
        result_df = run_multiple_seed_topic_modeling(
            input_file=args.input_file,
            output_file=args.output_file,
            method=args.method,
            num_topics=args.num_topics,
            ncomp=args.ncomp,
            seed_value=args.seed_value,
            n_samples=args.n_samples
        )
        
        logger.info(f"Topic modeling complete with {len(result_df)} entries")
        
    except Exception as e:
        logger.error(f"Error running topic modeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()