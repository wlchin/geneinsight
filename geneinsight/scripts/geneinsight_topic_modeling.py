#!/usr/bin/env python3
"""
Standalone script for running topic modeling on documents using the geneinsight pipeline.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from geneinsight.models.bertopic import run_multiple_seed_topic_modeling

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
        description="Run topic modeling on documents"
    )
    
    parser.add_argument(
        "documents_csv",
        help="Path to CSV file containing documents (requires 'description' column)"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        default="./output/topics.csv",
        help="Path to output CSV file (default: ./output/topics.csv)"
    )
    
    parser.add_argument(
        "-m", "--method",
        choices=["bertopic", "kmeans"],
        default="bertopic",
        help="Topic modeling method (default: bertopic)"
    )
    
    parser.add_argument(
        "-k", "--num-topics",
        type=int,
        default=10,
        help="Number of topics to extract (default: 10)"
    )
    
    parser.add_argument(
        "-c", "--components",
        type=int,
        default=2,
        help="Number of PCA components for KMeans (default: 2)"
    )
    
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Initial random seed (default: 42)"
    )
    
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=5,
        help="Number of models to run with different seeds (default: 5)"
    )
    
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Don't use sentence embeddings (not recommended)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.documents_csv):
        logger.error(f"Documents CSV file not found: {args.documents_csv}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    try:
        logger.info(f"Running topic modeling on {args.documents_csv}")
        logger.info(f"Method: {args.method}, Topics: {args.num_topics}, Samples: {args.n_samples}")
        
        # Run topic modeling
        topics_df = run_multiple_seed_topic_modeling(
            input_file=args.documents_csv,
            output_file=args.output_file,
            method=args.method,
            num_topics=args.num_topics,
            ncomp=args.components,
            seed_value=args.seed,
            n_samples=args.n_samples,
            use_sentence_embeddings=not args.no_embeddings
        )
        
        logger.info(f"Topic modeling results saved to {args.output_file}")
        logger.info(f"Generated {len(topics_df)} document-topic assignments")
        
    except Exception as e:
        logger.error(f"Error running topic modeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
