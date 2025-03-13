import argparse
import sys
import logging

from .pipeline import Pipeline

# Configure logging at the CLI level if desired
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        prog="geneinsight",
        description="Gene Insight CLI - run the TopicGenes pipeline with a given query set and background."
    )
    parser.add_argument(
        "query_gene_set",
        help="Path to file containing the query gene set."
    )
    parser.add_argument(
        "background_gene_list",
        help="Path to file containing the background gene list."
    )
    parser.add_argument(
        "-o", "--output_dir", default="./output",
        help="Directory to store final outputs. Default: './output'"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip generating an HTML report."
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of topic models to run with different seeds (default 5)."
    )
    parser.add_argument(
        "--num_topics", type=int, default=None,
        help="Number of topics to extract in topic modeling (default None)."
    )
    parser.add_argument(
        "--pvalue_threshold", type=float, default=0.05,
        help="Adjusted P-value threshold for filtering results (default 0.05)."
    )
    parser.add_argument(
        "--api_service", type=str, default="openai",
        help="API service for topic refinement (default 'openai')."
    )
    parser.add_argument(
        "--api_model", type=str, default="gpt-4o-mini",
        help="Model name for the API service (default 'gpt-4o-mini')."
    )
    parser.add_argument(
        "--api_parallel_jobs", type=int, default=50,
        help="Number of parallel API jobs (default 50)."
    )
    parser.add_argument(
        "--api_base_url", type=str, default=None,
        help="Base URL for the API service, if needed."
    )
    parser.add_argument(
        "--target_filtered_topics", type=int, default=25,
        help="Target number of topics after filtering (default 25)."
    )
    parser.add_argument(
        "--temp_dir", type=str, default=None,
        help="Temporary directory for intermediate files."
    )
    parser.add_argument(
        "--report_title", type=str, default=None,
        help="Title for the generated report."
    )
    # Add species argument
    parser.add_argument(
        "--species", type=int, default=9606,
        help="Species identifier (default: 9606 for human)."
    )
    
    args = parser.parse_args()

    pipeline = Pipeline(
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        n_samples=args.n_samples,
        num_topics=args.num_topics,
        pvalue_threshold=args.pvalue_threshold,
        api_service=args.api_service,
        api_model=args.api_model,
        api_parallel_jobs=args.api_parallel_jobs,
        api_base_url=args.api_base_url,
        target_filtered_topics=args.target_filtered_topics,
        species=args.species  # Pass species parameter to Pipeline
    )

    logger.info("Running GeneInsight pipeline from CLI...")

    pipeline.run(
        query_gene_set=args.query_gene_set,
        background_gene_list=args.background_gene_list,
        generate_report=not args.no_report,
        report_title=args.report_title
    )

if __name__ == "__main__":
    main()