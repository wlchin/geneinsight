# Geneinsight

![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

A Python package for topic modeling of gene sets with enrichment analysis.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
  - [Quick Installation](#quick-installation)
  - [Installation with Virtual Environment](#installation-with-virtual-environment-recommended)
- [Command Line Interface](#command-line-interface)
  - [Available Options](#available-options)
  - [Multi-Species Support](#multi-species-support)
  - [Gene Summary Options](#gene-summary-options)
  - [StringDB Options](#stringdb-options)
- [API Support](#api-support)
- [Environment Variables](#environment-variables)
- [Output Format](#output-format)
- [Interactive Report](#interactive-report)
- [Examples](#examples)
- [Documentation](#documentation)
- [License](#license)

## Overview

Geneinsight provides a comprehensive pipeline for analyzing gene sets through topic modeling and enrichment analysis:

1. **Gene Enrichment**: Query StringDB for gene enrichment data
2. **Topic Modeling**: Apply BERTopic to identify themes in gene sets
3. **Prompt Generation**: Generate prompts for language models based on topic data
4. **API Integration**: Use language models to refine topics and create meaningful annotations
5. **Enrichment Analysis**: Perform hypergeometric enrichment analysis on identified gene sets
6. **Results Packaging**: Export all results in a zipped format for easy sharing and analysis
7. **Interactive Report**: Generate an HTML report to visualize the results

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

```bash
# Basic usage
geneinsight query_genes.txt background_genes.txt -o ./output

# Advanced usage with OpenAI API
geneinsight query_genes.txt background_genes.txt \
  --n_samples 5 \
  --num_topics 10 \
  --pvalue_threshold 0.05 \
  --api_service openai \
  --api_model gpt-4o-mini \
  --api_temperature 0.2

# Using Ollama with local LLM
geneinsight query_genes.txt background_genes.txt \
  --api_service ollama \
  --api_model llama3.1:8b \
  --api_base_url "http://localhost:11434/v1"

# Using a different species (mouse)
geneinsight query_genes.txt background_genes.txt --species 10090

# Enable NCBI API calls for gene summaries
geneinsight query_genes.txt background_genes.txt --enable-ncbi-api

# Using local StringDB module instead of API
geneinsight query_genes.txt background_genes.txt --use-local-stringdb
```

### Available Options

```
usage: geneinsight [-h] [-o OUTPUT_DIR] [--no-report] [--n_samples N_SAMPLES]
                   [--num_topics NUM_TOPICS] [--pvalue_threshold PVALUE_THRESHOLD]
                   [--api_service API_SERVICE] [--api_model API_MODEL] [--api_temperature API_TEMPERATURE]
                   [--api_parallel_jobs API_PARALLEL_JOBS] [--api_base_url API_BASE_URL]
                   [--target_filtered_topics TARGET_FILTERED_TOPICS] [--temp_dir TEMP_DIR]
                   [--report_title REPORT_TITLE] [--species SPECIES]
                   [--filtered_n_samples FILTERED_N_SAMPLES] [--enable-ncbi-api]
                   [--use-local-stringdb] [-v {none,debug,info,warning,error,critical}]
                   query_gene_set background_gene_list
```

| Argument                | Description                                            | Default  |
|-------------------------|--------------------------------------------------------|----------|
| `query_gene_set`        | Path to file containing the query gene set           | Required |
| `background_gene_list`  | Path to file containing the background gene list     | Required |
| `-o`, `--output_dir`     | Directory to store final outputs                      | ./output |
| `--no-report`           | Skip generating an HTML report                         | False    |
| `--n_samples`           | Number of topic models to run with different seeds     | 5        |
| `--filtered_n_samples`  | Number of topic models to run from filtered gene sets  | 10       |
| `--num_topics`          | Number of topics to extract in topic modeling          | None (auto) |
| `--pvalue_threshold`    | Adjusted P-value threshold for filtering results       | 0.05     |
| `--api_service`         | API service for topic refinement                       | openai   |
| `--api_model`           | Model name for the API service                         | gpt-4o-mini |
| `--api_parallel_jobs`   | Number of parallel API jobs                            | 1        |
| `--api_base_url`        | Base URL for the API service                           | None     |
| `--target_filtered_topics` | Target number of topics after filtering           | 25       |
| `--temp_dir`            | Temporary directory for intermediate files             | None     |
| `--report_title`        | Title for the generated report                         | None     |
| `--species`             | Species identifier (NCBI taxonomy ID)                  | 9606     |
| `--enable-ncbi-api`     | Enable NCBI API calls for gene summaries               | False    |
| `--use-local-stringdb`  | Use local StringDB module instead of API               | False    |
| `-v`, `--verbosity`      | Set logging verbosity                                  | none     |
| `--api_temperature`     | Sampling temperature for API calls                     | 0.2      |

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
   - Requires setting the `--api_base_url` parameter, usually to `"http://localhost:11434/v1"`
   - Example usage: `--api_service ollama --api_model llama3:8.1b --api_base_url "http://localhost:11434/v1"`

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

## Documentation

Comprehensive documentation is available at [the Geneinsight documentation site](https://wlchin.github.io/geneinsight/index.html).

The documentation includes detailed API references, examples, and advanced usage guides.

## License

This project is licensed under the MIT License.