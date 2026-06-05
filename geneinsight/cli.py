import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .pipeline import Pipeline

console = Console()
logger = logging.getLogger(__name__)

SUBCOMMANDS = {"run", "build-null", "validate-null", "cache-info"}


def _build_run_parser() -> argparse.ArgumentParser:
    """Argparse parser for the `run` subcommand (the original CLI surface)."""
    parser = argparse.ArgumentParser(
        prog="geneinsight",
        description="Gene Insight CLI - run the TopicGenes pipeline with a given query set and background."
    )
    parser.add_argument("query_gene_set", help="Path to file containing the query gene set.")
    parser.add_argument("background_gene_list", help="Path to file containing the background gene list.")
    parser.add_argument("-o", "--output-dir", default="./output",
                        help="Directory to store final outputs. Default: './output'")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip generating an HTML report.")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of topic models to run with different seeds (default 5).")
    parser.add_argument("--num-topics", type=int, default=None,
                        help="Number of topics to extract in topic modeling (default None).")
    parser.add_argument("--pvalue-threshold", type=float, default=0.05,
                        help="Adjusted P-value threshold for filtering results (default 0.05).")
    parser.add_argument("--api-service", type=str, default="openai",
                        help="API service for topic refinement (default 'openai').")
    parser.add_argument("--api-model", type=str, default="gpt-4o-mini",
                        help="Model name for the API service (default 'gpt-4o-mini').")
    parser.add_argument("--api-parallel-jobs", type=int, default=1,
                        help="Number of parallel API jobs (default 1).")
    parser.add_argument("--api-base-url", type=str, default=None,
                        help="Base URL for the API service, if needed.")
    parser.add_argument("--target-filtered-topics", type=int, default=25,
                        help="Target number of topics after filtering (default 25).")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Temporary directory for intermediate files.")
    parser.add_argument("--report-title", type=str, default=None,
                        help="Title for the generated report.")
    parser.add_argument("--species", type=int, default=9606,
                        help="Species identifier (default: 9606 for human).")
    parser.add_argument("--filtered-n-samples", type=int, default=10,
                        help="Number of topic models to run on filtered sets (default 10).")
    parser.add_argument("--api-temperature", type=float, default=0.2,
                        help="Sampling temperature for API calls (default: 0.2).")
    parser.add_argument("--enable-ncbi-api", action="store_true",
                        help="Enable NCBI API calls for gene summaries (disabled by default).")
    parser.add_argument("--use-local-stringdb", action="store_true",
                        help="Use local StringDB module instead of API (API is default).")
    parser.add_argument("--overlap-ratio-threshold", type=float, default=0.25,
                        help="Minimum overlap ratio threshold for filtering terms (default: 0.25).")
    parser.add_argument("-v", "--verbosity",
                        default="none",
                        choices=["none", "debug", "info", "warning", "error", "critical"],
                        help="Set logging verbosity. Use 'none' to disable logging. Default is 'none'.")

    # Empirical p-value arguments
    parser.add_argument("--empirical-p", action="store_true",
                        help="Annotate enriched.csv with empirical p_empirical / q_empirical "
                             "from a size-stratified null pool.")
    parser.add_argument("--empirical-p-statistic", choices=["raw", "overlap"], default="raw",
                        help="Statistic for empirical recalibration: 'raw' hypergeometric P "
                             "(default; better calibration) or 'overlap' ratio (Reviewer 3's framing).")
    parser.add_argument("--null-cache", type=str, default=None,
                        help="Path to a custom null-pool directory built with `geneinsight build-null`. "
                             "If omitted, the bundled default null pool is used.")

    # Instrumentation/metrics arguments
    parser.add_argument("--no-metrics", action="store_true",
                        help="Disable pipeline metrics collection (timing and token tracking).")
    parser.add_argument("--quiet-metrics", action="store_true",
                        help="Suppress console display of metrics summary (still saves to JSON).")
    parser.add_argument("--metrics-output", type=str, default=None,
                        help="Custom path for pipeline_metrics.json output file.")

    return parser


def _configure_logging(verbosity: str) -> None:
    if verbosity == "none":
        logging.disable(logging.CRITICAL)
        return
    level = getattr(logging, verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def _run(argv: list[str]) -> int:
    parser = _build_run_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbosity)
    logger.debug("Debug mode is on.")
    logger.info("Initializing GeneInsight pipeline from CLI...")

    species_names = {
        9606: "Human", 10090: "Mouse", 10116: "Rat", 7955: "Zebrafish",
        7227: "Fruit fly", 6239: "C. elegans", 4932: "Yeast",
    }
    species_name = species_names.get(args.species, f"Species ID {args.species}")
    logger.info(f"Using species: {species_name} (taxonomy ID: {args.species})")
    logger.info(f"NCBI API calls: {'enabled' if args.enable_ncbi_api else 'disabled'}")
    logger.info(f"StringDB mode: {'local' if args.use_local_stringdb else 'API'}")
    if args.empirical_p:
        cache_label = args.null_cache or "<bundled default>"
        logger.info(
            f"Empirical p-values: ENABLED (statistic={args.empirical_p_statistic}, "
            f"null cache={cache_label})"
        )

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
        api_temperature=args.api_temperature,
        call_ncbi_api=args.enable_ncbi_api,
        use_local_stringdb=args.use_local_stringdb,
        overlap_ratio_threshold=args.overlap_ratio_threshold,
        empirical_p=args.empirical_p,
        empirical_p_statistic=args.empirical_p_statistic,
        null_cache=args.null_cache,
        enable_metrics=not args.no_metrics,
        quiet_metrics=args.quiet_metrics,
        metrics_output_path=args.metrics_output,
    )

    try:
        pipeline.run(
            query_gene_set=args.query_gene_set,
            background_gene_list=args.background_gene_list,
            generate_report=not args.no_report,
            report_title=args.report_title,
        )
        logger.info("GeneInsight pipeline completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error running GeneInsight pipeline: {e}", exc_info=True)
        return 1


def _build_null(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="geneinsight build-null",
        description="Build a custom null pool by running random gene sets through the pipeline.",
    )
    parser.add_argument("--query-size", type=int, required=True,
                        help="Gene-set size to sample (1-500). Defines which size bin the null covers.")
    parser.add_argument("--replicates", type=int, required=True,
                        help="Number of random gene sets to run. More = tighter empirical resolution.")
    parser.add_argument("--background", type=str, required=True,
                        help="Path to background gene list (one gene per line).")
    parser.add_argument("--output-cache", type=str, required=True,
                        help="Directory to write the null pool (NullPool-compatible).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for deterministic gene-set sampling (default 42).")
    parser.add_argument("--species", type=int, default=9606,
                        help="Species taxonomy ID (default 9606 / human).")
    parser.add_argument("--api-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--use-local-stringdb", action="store_true",
                        help="Use local StringDB cache instead of the API.")
    parser.add_argument("-v", "--verbosity", default="info",
                        choices=["none", "debug", "info", "warning", "error", "critical"])

    args = parser.parse_args(argv)
    _configure_logging(args.verbosity)

    console.print(
        f"[bold]build-null[/bold]: size={args.query_size}, replicates={args.replicates}, "
        f"output={args.output_cache}"
    )
    console.print(
        "[yellow]Note:[/yellow] each replicate runs the full pipeline serially "
        "(api_parallel_jobs=1, enforced). Plan for ~1-2.5 h per replicate."
    )

    from .calibration.builder import build_null

    try:
        out = build_null(
            query_size=args.query_size,
            replicates=args.replicates,
            background=args.background,
            output_cache=args.output_cache,
            seed=args.seed,
            species=args.species,
            api_model=args.api_model,
            use_local_stringdb=args.use_local_stringdb,
        )
    except Exception as exc:
        logger.error(f"build-null failed: {exc}", exc_info=True)
        return 1
    console.print(f"[bold green]Null pool written to {out}[/bold green]")
    return 0


def _cache_info(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="geneinsight cache-info",
        description="Print the manifest and bin counts for a null pool.",
    )
    parser.add_argument("--cache", type=str, default=None,
                        help="Path to a null pool directory. Omit to inspect the bundled default.")
    parser.add_argument("--statistic", choices=["raw", "overlap"], default="raw")
    args = parser.parse_args(argv)

    from .calibration import NullPool

    try:
        pool = (NullPool.from_path(args.cache, statistic=args.statistic)
                if args.cache else
                NullPool.from_bundled(statistic=args.statistic))
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(f"[bold]NullPool[/bold] {pool!r}")
    console.print(f"[bold]Source:[/bold] {pool.source}")
    if pool.manifest:
        console.print("[bold]Manifest:[/bold]")
        console.print_json(json.dumps(pool.manifest))
    counts = pool.bin_counts()
    console.print("[bold]Bin counts:[/bold]")
    for label, n in sorted(counts.items(), key=lambda kv: int(kv[0].split("-")[0])):
        console.print(f"  {label:>8s}  {n:>7d}")
    return 0


def _validate_null(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="geneinsight validate-null",
        description="Leave-one-out KS validation of a null pool (smoke check).",
    )
    parser.add_argument("--cache", type=str, default=None,
                        help="Null pool directory (default: bundled).")
    parser.add_argument("--statistic", choices=["raw", "overlap"], default="raw")
    args = parser.parse_args(argv)

    import numpy as np
    from scipy.stats import kstest

    from .calibration import NullPool
    from .calibration.empirical_p import compute_empirical_p

    pool = (NullPool.from_path(args.cache, statistic=args.statistic)
            if args.cache else
            NullPool.from_bundled(statistic=args.statistic))

    console.print(f"[bold]Validating[/bold] {pool!r}")
    rows = []
    for bin_label in sorted(pool.bin_counts().keys(),
                            key=lambda b: int(b.split("-")[0])):
        sorted_null = pool.sorted_bin(bin_label)
        if len(sorted_null) < 2:
            continue
        # Leave-one-out: empirical p for each element vs the other n-1 (we approximate
        # using rank within the full pool; for typical n>>1 this is within 1/n of LOO).
        p_emp = compute_empirical_p(sorted_null, sorted_null, args.statistic)
        ks_stat = kstest(p_emp, "uniform").statistic
        rows.append((bin_label, len(sorted_null), ks_stat))

    console.print(f"[bold]Per-bin KS (vs Uniform(0,1)):[/bold]")
    for label, n, ks in rows:
        flag = "✓" if ks <= 0.10 else "!"
        console.print(f"  {flag} {label:>8s}  n={n:>6d}  KS={ks:.3f}")
    return 0


def main() -> int:
    """Dispatch entry point.

    - `geneinsight build-null|validate-null|cache-info ...` -> subcommand handler
    - `geneinsight run ...`                                 -> explicit run
    - `geneinsight QUERY BG [...]`                          -> back-compat fallthrough to run
    """
    argv = sys.argv[1:]
    if argv and argv[0] in SUBCOMMANDS:
        cmd = argv[0]
        rest = argv[1:]
        if cmd == "run":
            return _run(rest)
        if cmd == "build-null":
            return _build_null(rest)
        if cmd == "validate-null":
            return _validate_null(rest)
        if cmd == "cache-info":
            return _cache_info(rest)
    # Back-compat: treat the call as `geneinsight run ARGV...`
    return _run(argv)


if __name__ == "__main__":
    sys.exit(main())
