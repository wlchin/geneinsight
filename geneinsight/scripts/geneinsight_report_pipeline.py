#!/usr/bin/env python3
"""
GeneInsight Report Pipeline CLI Script

This script provides a command-line interface for the GeneInsight report pipeline.
When installed with the package, it's available as the 'geneinsight-report' command.
"""

import os
import sys
import argparse
import logging

from geneinsight.report.pipeline import run_pipeline, setup_logging

def main():
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the GeneInsight report pipeline")
    parser.add_argument("--input_folder", required=True, help="Folder containing the input CSV files")
    parser.add_argument("--output_folder", required=True, help="Folder where the output will be generated")
    parser.add_argument("--gene_set_name", required=True, help="Name of the gene set")
    args = parser.parse_args()
    
    # Create the logs folder
    log_folder = os.path.join(args.output_folder, "logs")
    log_file = setup_logging(log_folder)
    
    # Print welcome message
    print(f"""
=====================================
  GeneInsight Report Pipeline
=====================================
Input folder: {args.input_folder}
Output folder: {args.output_folder}
Gene set name: {args.gene_set_name}
Log file: {log_file}
=====================================
    """)
    
    # Run the pipeline
    success, result = run_pipeline(args.input_folder, args.output_folder, args.gene_set_name)
    
    if success:
        print(f"""
=====================================
  Pipeline Completed Successfully!
=====================================
The Sphinx documentation is available at:
{result}
=====================================
        """)
        sys.exit(0)
    else:
        print(f"""
=====================================
  Pipeline Failed
=====================================
Error: {result}
Check the log file for details: {log_file}
=====================================
        """)
        sys.exit(1)

if __name__ == "__main__":
    main()