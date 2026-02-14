# Geneinsight

![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen)
![Python 3.10-3.13](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)

A Python package for topic modeling of gene sets with enrichment analysis.

Geneinsight uses AI language models to make sense of complex gene data, helping scientists uncover hidden patterns and biological themes in their genomic research without requiring advanced computational expertise.

## Overview

Geneinsight provides a comprehensive pipeline for analyzing gene sets through topic modeling and enrichment analysis:

![Geneinsight Framework](framework.png)

1. **Gene Enrichment**: Query StringDB for gene enrichment data
2. **Topic Modeling**: Apply BERTopic to identify themes in gene sets
3. **Prompt Generation**: Generate prompts for language models based on topic data
4. **API Integration**: Use language models to refine topics and create meaningful annotations
5. **Enrichment Analysis**: Perform hypergeometric enrichment analysis on identified gene sets
6. **Results Packaging**: Export all results in a zipped format for easy sharing and analysis
7. **Interactive Report**: Generate an HTML report to visualize the results

## Documentation

Comprehensive documentation is available at [the Geneinsight documentation site](https://wlchin.github.io/geneinsight/index.html).

The documentation includes detailed API references, examples, and advanced usage guides.

## Table of Contents
- [Installation](#installation)
  - [Quick Installation](#quick-installation)
  - [Installation with Virtual Environment](#installation-with-virtual-environment-recommended)
- [Command Line Interface](#command-line-interface)
  - [Gene Set Size Recommendations](#gene-set-size-recommendations)
  - [Basic Usage](#basic-usage)
  - [Available Options](#available-options)
  - [Multi-Species Support](#multi-species-support)
  - [Gene Summary Options](#gene-summary-options)
  - [StringDB Options](#stringdb-options)
- [API Support](#api-support)
- [Environment Variables](#environment-variables)
- [Output Format](#output-format)
- [Pipeline Metrics](#pipeline-metrics)
- [Interactive Report](#interactive-report)
- [Examples](#examples)
- [Reproducibility](#reproducibility)
- [External Resources](#external-resources)
- [License](#license)

## Installation

Requires Python 3.10+.

### Quick Installation

```bash
# Install directly from GitHub with pip
pip install git+https://github.com/wlchin/geneinsight.git

# OR use UV for faster installation (recommended)
# Install UV if you don't have it yet: https://github.com/astral-sh/uv
uv pip install git+https://github.com/wlchin/geneinsight.git
```

### Installation with Virtual Environment (Recommended)

Using a virtual environment is recommended to avoid package conflicts.

#### Using standard pip and venv

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Geneinsight
pip install git+https://github.com/wlchin/geneinsight.git
```

#### Using UV (Faster Alternative)

UV offers faster package resolution and installation:

```bash
# Install UV if you don't have it yet: https://github.com/astral-sh/uv

# Create a Python 3.10 virtual environment with UV
uv venv --python=3.10

# Activate the virtual environment
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Geneinsight with UV
uv pip install git+https://github.com/wlchin/geneinsight.git
```

## Command Line Interface

### Gene Set Size Recommendations

For optimal results, we recommend keeping query gene sets to **500 genes or fewer**, consistent with standard gene set enrichment analysis guidelines. Larger gene sets may:
- Reduce the specificity of identified biological themes
- Increase processing time and API costs
- Yield less interpretable results

There is no hard-coded limit, but results are most meaningful when analyzing focused gene sets derived from differential expression, GWAS hits, or other targeted analyses.

### Input File Format

Both query and background gene files should be plain text files with **one gene per line** and **no header row**:

```
TP53
BRCA1
EGFR
MYC
```

Gene identifiers can be gene symbols (e.g., `TP53`) or Ensembl IDs (e.g., `ENSG00000141510`). The pipeline will automatically convert Ensembl IDs to gene symbols via StringDB.

### Basic Usage

```bash
# Basic usage
geneinsight query_genes.txt background_genes.txt -o ./output

# Advanced usage with OpenAI API
geneinsight query_genes.txt background_genes.txt \
  --n-samples 5 \
  --num-topics 10 \
  --pvalue-threshold 0.05 \
  --api-service openai \
  --api-model gpt-4o-mini \
  --api-temperature 0.2

# Using Ollama with local LLM
geneinsight query_genes.txt background_genes.txt \
  --api-service ollama \
  --api-model llama3.1:8b \
  --api-base-url "http://localhost:11434/v1"

# Using a different species (mouse)
geneinsight query_genes.txt background_genes.txt --species 10090

# Enable NCBI API calls for gene summaries
geneinsight query_genes.txt background_genes.txt --enable-ncbi-api

# Using local StringDB module instead of API
geneinsight query_genes.txt background_genes.txt --use-local-stringdb
```

### Available Options

```
usage: geneinsight [-h] [-o OUTPUT_DIR] [--no-report] [--n-samples N_SAMPLES]
                   [--num-topics NUM_TOPICS] [--pvalue-threshold PVALUE_THRESHOLD]
                   [--api-service API_SERVICE] [--api-model API_MODEL] [--api-temperature API_TEMPERATURE]
                   [--api-parallel-jobs API_PARALLEL_JOBS] [--api-base-url API_BASE_URL]
                   [--target-filtered-topics TARGET_FILTERED_TOPICS] [--temp-dir TEMP_DIR]
                   [--report-title REPORT_TITLE] [--species SPECIES]
                   [--filtered-n-samples FILTERED_N_SAMPLES] [--enable-ncbi-api]
                   [--use-local-stringdb] [--overlap-ratio-threshold OVERLAP_RATIO_THRESHOLD]
                   [--no-metrics] [--quiet-metrics] [--metrics-output METRICS_OUTPUT]
                   [-v {none,debug,info,warning,error,critical}]
                   query_gene_set background_gene_list
```

| Argument                | Description                                            | Default  |
|-------------------------|--------------------------------------------------------|----------|
| `query_gene_set`        | Path to file containing the query gene set           | Required |
| `background_gene_list`  | Path to file containing the background gene list     | Required |
| `-o`, `--output-dir`     | Directory to store final outputs                      | ./output |
| `--no-report`           | Skip generating an HTML report                         | False    |
| `--n-samples`           | Number of topic models to run with different seeds     | 5        |
| `--filtered-n-samples`  | Number of topic models to run from filtered gene sets  | 10       |
| `--num-topics`          | Number of topics to extract in topic modeling          | None (auto) |
| `--pvalue-threshold`    | Adjusted P-value threshold for filtering results       | 0.05     |
| `--api-service`         | API service for topic refinement                       | openai   |
| `--api-model`           | Model name for the API service                         | gpt-4o-mini |
| `--api-parallel-jobs`   | Number of parallel API jobs                            | 1        |
| `--api-base-url`        | Base URL for the API service                           | None     |
| `--target-filtered-topics` | Target number of topics after filtering           | 25       |
| `--temp-dir`            | Temporary directory for intermediate files             | None     |
| `--report-title`        | Title for the generated report                         | None     |
| `--species`             | Species identifier (NCBI taxonomy ID)                  | 9606     |
| `--enable-ncbi-api`     | Enable NCBI API calls for gene summaries               | False    |
| `--use-local-stringdb`  | Use local StringDB module instead of API               | False    |
| `-v`, `--verbosity`      | Set logging verbosity                                  | none     |
| `--api-temperature`     | Sampling temperature for API calls                     | 0.2      |
| `--overlap-ratio-threshold` | Minimum overlap ratio threshold for filtering terms | 0.25     |
| `--no-metrics`          | Disable pipeline metrics collection                    | False    |
| `--quiet-metrics`       | Suppress console display of metrics summary            | False    |
| `--metrics-output`      | Custom path for pipeline_metrics.json output           | None     |

### Multi-Species Support

Geneinsight supports different species through NCBI taxonomy IDs. Common species IDs:

| Species               | Taxonomy ID |
|-----------------------|-------------|
| Human                 | 9606        |
| Mouse                 | 10090       |
| Rat                   | 10116       |
| Zebrafish             | 7955        |
| Fruit Fly             | 7227        |
| C. elegans            | 6239        |

The `--species` parameter accepts any valid NCBI taxonomy ID, allowing analysis of gene sets from virtually any organism.

### Gene Summary Options

By default, NCBI API calls for gene summaries are disabled to improve performance and avoid rate limiting. To enable detailed gene summaries with tooltips in the generated report, use the `--enable-ncbi-api` flag:

```bash
geneinsight query_genes.txt background_genes.txt --enable-ncbi-api
```

This will fetch gene descriptions from NCBI and include them in the report, but may slow down the processing.

### StringDB Options

Geneinsight offers two methods for retrieving gene enrichment data from StringDB:

1. **StringDB API** (default): Queries the StringDB web API for gene enrichment data
   - Provides real-time access to the latest StringDB data
   - Requires an internet connection
   - May be slower due to network requests and rate limiting

2. **Local StringDB Module**: Uses a local cache of StringDB data
   - Significantly faster for large gene sets
   - Works offline after initial cache download
   - Currently supports human (9606) and mouse (10090) species
   - Cache files are automatically downloaded on first use

To use the local StringDB module instead of the API:

```bash
geneinsight query_genes.txt background_genes.txt --use-local-stringdb
```

The local mode is particularly useful for batch processing of multiple gene sets or when working with limited internet connectivity.

## API Support

Geneinsight supports two API services for topic refinement:

1. **OpenAI API**: Default option that works with models like gpt-4o-mini
   - Requires an API key to be set in environment variables, or alternatively a .env file.
   - Works out of the box with default settings

2. **Ollama API**: Local option for running models on your own hardware
   - Requires a model that supports tool use (see [supported models](https://ollama.com/search?c=tools))
   - Requires setting the `--api-base-url` parameter, usually to `"http://localhost:11434/v1"`
   - Example usage: `--api-service ollama --api-model llama3:8.1b --api-base-url "http://localhost:11434/v1"`

## Environment Variables

Set the following environment variable for API access:

```
OPENAI_API_KEY=your_openai_key_here
```

## Output Format

The pipeline produces a directory (which can be optionally zipped) containing:

- `enrichment.csv`: Gene enrichment data from StringDB
- `documents.csv`: Document descriptions for topic modeling
- `topics.csv`: Topic modeling results
- `prompts.csv`: Generated prompts for API
- `api_results.csv`: Results from API calls
- `summary.csv`: Summary of topic modeling and enrichment
- `enriched.csv`: Hypergeometric enrichment results
- `filtered.csv`: Final filtered topics
- `metadata.csv`: Run information and parameters
- `pipeline_metrics.json`: Timing and token usage metrics (unless disabled with `--no-metrics`)

## Pipeline Metrics

Geneinsight tracks detailed metrics for each pipeline run, including:

- **Stage timing**: Duration of each pipeline step
- **Token usage**: Prompt and completion tokens consumed by API calls
- **API statistics**: Number of calls and latency information

By default, a summary is displayed after each run and saved to `pipeline_metrics.json` in the output directory. Control this behavior with:

```bash
# Disable metrics collection entirely
geneinsight query_genes.txt background_genes.txt --no-metrics

# Collect metrics but suppress console output
geneinsight query_genes.txt background_genes.txt --quiet-metrics

# Save metrics to a custom location
geneinsight query_genes.txt background_genes.txt --metrics-output /path/to/metrics.json
```

## Interactive Report

By default, Geneinsight generates an interactive HTML report visualizing the results. You can disable this with the `--no-report` flag. The report includes:

1. **Interactive Topic Map** - A 2D visualization showing relationships between topics
2. **Theme Pages** - Detailed pages for each identified theme
3. **Gene Set Visualizations** - Heatmaps showing gene presence across references
4. **Summary Statistics** - Key metrics about the analysis
5. **Download Interface** - Interactive interface to download specific themes

## Examples

Geneinsight provides example files to help you quickly get started with gene set analysis:

```bash
# Create an examples folder in your current directory
geneinsight-examples

# Or specify a custom destination
geneinsight-examples --path /path/to/destination
```

The examples folder contains:
- 2 sample text files with an example gene set and its background set.
- A README.md with documentation on how to use the examples

For a quick start, try running:

```bash
geneinsight examples/sample.txt examples/sample_background.txt -o ./output
```

## Reproducibility

The scripts used to reproduce the results presented in the manuscript are available as [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflows in the `reproducibility` folder. These workflows ensure complete reproducibility of our analyses and can be easily adapted for similar studies.

Key workflows include:
- Hyperparameter optimization
- Test set evaluation
- Visualization generation
- Cosine similarity and MoverScore calculation

## External Resources

Geneinsight integrates with several external resources:

- **[STRING-DB](https://string-db.org/)**: Used for protein-protein interaction network analysis and functional enrichment of gene sets
- **[NCBI](https://www.ncbi.nlm.nih.gov/)** (National Center for Biotechnology Information): Used for retrieving gene annotations and taxonomy information
- **[BERTopic](https://maartengr.github.io/BERTopic/index.html)**: The core topic modeling framework that powers Geneinsight's ability to discover themes in gene sets

## License

This project is licensed under the MIT License.