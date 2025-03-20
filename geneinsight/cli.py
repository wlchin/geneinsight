import argparse
import sys
import logging

from rich.logging import RichHandler
from rich.console import Console

from .pipeline import Pipeline

console = Console()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        prog="geneinsight",
        description="Gene Insight CLI - run the TopicGenes pipeline with a given query set and background."
    )
    parser.add_argument("query_gene_set", help="Path to file containing the query gene set.")
    parser.add_argument("background_gene_list", help="Path to file containing the background gene list.")
    parser.add_argument("-o", "--output_dir", default="./output",
                        help="Directory to store final outputs. Default: './output'")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip generating an HTML report.")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of topic models to run with different seeds (default 5).")
    parser.add_argument("--num_topics", type=int, default=None,
                        help="Number of topics to extract in topic modeling (default None).")
    parser.add_argument("--pvalue_threshold", type=float, default=0.05,
                        help="Adjusted P-value threshold for filtering results (default 0.05).")
    parser.add_argument("--api_service", type=str, default="openai",
                        help="API service for topic refinement (default 'openai').")
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini",
                        help="Model name for the API service (default 'gpt-4o-mini').")
    parser.add_argument("--api_parallel_jobs", type=int, default=1,
                        help="Number of parallel API jobs (default 1).")
    parser.add_argument("--api_base_url", type=str, default=None,
                        help="Base URL for the API service, if needed.")
    parser.add_argument("--target_filtered_topics", type=int, default=25,
                        help="Target number of topics after filtering (default 25).")
    parser.add_argument("--temp_dir", type=str, default=None,
                        help="Temporary directory for intermediate files.")
    parser.add_argument("--report_title", type=str, default=None,
                        help="Title for the generated report.")
    parser.add_argument("--species", type=int, default=9606,
                        help="Species identifier (default: 9606 for human).")
    # New argument for controlling number of topic models on filtered sets
    parser.add_argument("--filtered_n_samples", type=int, default=10,
                        help="Number of topic models to run on filtered sets (default 10).")
    # New argument for controlling API temperature
    parser.add_argument("--api_temperature", type=float, default=0.2,
                        help="Sampling temperature for API calls (default: 0.2).")
    # Changed from --no-ncbi-api to --enable-ncbi-api to make disabled the default
    parser.add_argument("--enable-ncbi-api", action="store_true",
                        help="Enable NCBI API calls for gene summaries (disabled by default).")
    # Argument for controlling verbosity
    parser.add_argument("-v", "--verbosity",
                        default="none",
                        choices=["none", "debug", "info", "warning", "error", "critical"],
                        help="Set logging verbosity. Use 'none' to disable logging. Default is 'none'.")

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Configure logging using RichHandler
    # ---------------------------------------------------------------------
    if args.verbosity == "none":
        # Disable all logging
        logging.disable(logging.CRITICAL)
    else:
        # Convert string (e.g. "info", "error") to logging constant (e.g. logging.INFO)
        level = getattr(logging, args.verbosity.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )

    logger.debug("Debug mode is on.")
    logger.info("Initializing GeneInsight pipeline from CLI...")
    
    # Log species and NCBI API settings
    species_names = {
        9606: "Human",
        10090: "Mouse",
        10116: "Rat",
        7955: "Zebrafish",
        7227: "Fruit fly",
        6239: "C. elegans",
        4932: "Yeast"
    }
    species_name = species_names.get(args.species, f"Species ID {args.species}")
    logger.info(f"Using species: {species_name} (taxonomy ID: {args.species})")
    logger.info(f"NCBI API calls: {'enabled' if args.enable_ncbi_api else 'disabled'}")

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
        species=args.species,
        filtered_n_samples=args.filtered_n_samples,
        api_temperature=args.api_temperature,  # pass temperature parameter
        call_ncbi_api=args.enable_ncbi_api     # NCBI API calls disabled by default, enabled with flag
    )

    try:
        pipeline.run(
            query_gene_set=args.query_gene_set,
            background_gene_list=args.background_gene_list,
            generate_report=not args.no_report,
            report_title=args.report_title
        )
        logger.info("GeneInsight pipeline completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error running GeneInsight pipeline: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())