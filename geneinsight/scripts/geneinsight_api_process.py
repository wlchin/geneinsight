#!/usr/bin/env python3
"""
Standalone script for processing prompts through API.
"""

import os
import sys
import argparse
import logging
from typing import Optional

from geneinsight.api.client import batch_process_api_calls

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
        description="Process prompts through API"
    )
    
    parser.add_argument(
        "prompts_csv",
        help="Path to CSV file containing prompts"
    )
    
    parser.add_argument(
        "-o", "--output-file",
        default="./output/api_results.csv",
        help="Path to output CSV file (default: ./output/api_results.csv)"
    )
    
    parser.add_argument(
        "-s", "--service",
        choices=["openai", "together", "ollama"],
        default="openai",
        help="API service to use (default: openai)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "-u", "--base-url",
        help="Base URL for API service (required for Together and Ollama)"
    )
    
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.prompts_csv):
        logger.error(f"Prompts CSV file not found: {args.prompts_csv}")
        sys.exit(1)
    
    # Check if API key is set for OpenAI or Together
    if args.service.lower() in ["openai", "together"]:
        env_var = f"{args.service.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            logger.error(f"Environment variable {env_var} not set")
            logger.error(f"Please set your {args.service} API key")
            sys.exit(1)
    
    # Check if base URL is set for Together or Ollama
    if args.service.lower() in ["together", "ollama"] and not args.base_url:
        logger.error(f"Base URL is required for {args.service}")
        logger.error("Use --base-url to set the API base URL")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    try:
        logger.info(f"Processing prompts from {args.prompts_csv}")
        logger.info(f"Service: {args.service}, Model: {args.model}, Jobs: {args.jobs}")
        
        # Process prompts
        result_df = batch_process_api_calls(
            prompts_csv=args.prompts_csv,
            output_api=args.output_file,
            service=args.service,
            model=args.model,
            base_url=args.base_url,
            n_jobs=args.jobs
        )
        
        logger.info(f"API processing complete")
        logger.info(f"Results saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error processing prompts: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
