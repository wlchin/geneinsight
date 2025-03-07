#!/usr/bin/env python3
"""
Standalone script for generating prompts from topic modeling results using the geneinsight pipeline.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from geneinsight.workflows.prompt_generation import generate_prompts

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
        description="Generate prompts from topic modeling results"
    )
    
    parser.add_argument(
        "topics_csv",
        help="Path to CSV file containing topic modeling results"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        default="./output/prompts.csv",
        help="Path to output CSV file (default: ./output/prompts.csv)"
    )
    
    parser.add_argument(
        "-n", "--num-subtopics",
        type=int,
        default=5,
        help="Number of subtopics to generate per topic (default: 5)"
    )
    
    parser.add_argument(
        "-w", "--max-words",
        type=int,
        default=10,
        help="Maximum number of words per subtopic title (default: 10)"
    )
    
    parser.add_argument(
        "-r", "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per subtopic (default: 3)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.topics_csv):
        logger.error(f"Topics CSV file not found: {args.topics_csv}")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    try:
        logger.info(f"Generating prompts from {args.topics_csv}")
        
        # Generate prompts
        prompts_df = generate_prompts(
            input_file=args.topics_csv,
            num_subtopics=args.num_subtopics,
            max_words=args.max_words,
            output_file=args.output_file,
            max_retries=args.max_retries
        )
        
        logger.info(f"Generated {len(prompts_df)} prompts")
        logger.info(f"Prompts saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
