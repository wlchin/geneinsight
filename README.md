# Geneinsight

![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)

A Python package for topic modeling of gene sets with enrichment analysis. 

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

Requires Python 3.9+.

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

## Quick Start

Run the pipeline on your gene sets:

```bash
# Basic usage
geneinsight query_genes.txt background_genes.txt -o ./output

# Advanced usage with OpenAI API
geneinsight query_genes.txt background_genes.txt \
  --n_samples 5 \
  --num_topics 10 \
  --pvalue_threshold 0.05 \
  --api_service openai \
  --api_model gpt-4o-mini

# Using Ollama with local LLM
geneinsight query_genes.txt background_genes.txt \
  --api_service ollama \
  --api_model llama3.1:8b \
  --api_base_url "http://localhost:11434/v1"
```

## Command Line Interface

```
usage: geneinsight [-h] [-o OUTPUT_DIR] [--no-report] [--n_samples N_SAMPLES]
                   [--num_topics NUM_TOPICS] [--pvalue_threshold PVALUE_THRESHOLD]
                   [--api_service API_SERVICE] [--api_model API_MODEL]
                   [--api_parallel_jobs API_PARALLEL_JOBS] [--api_base_url API_BASE_URL]
                   [--target_filtered_topics TARGET_FILTERED_TOPICS]
                   [--temp_dir TEMP_DIR] [--report_title REPORT_TITLE]
                   [--species SPECIES] [-v {none,debug,info,warning,error,critical}]
                   query_gene_set background_gene_list

Gene Insight CLI - run the GeneInsight pipeline with a given query set and background.

positional arguments:
  query_gene_set        Path to file containing the query gene set.
  background_gene_list  Path to file containing the background gene list.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to store final outputs. Default: './output'
  --no-report           Skip generating an HTML report.
  --n_samples N_SAMPLES
                        Number of topic models to run with different seeds (default 5).
  --num_topics NUM_TOPICS
                        Number of topics to extract in topic modeling (default None (automatic)).
  --pvalue_threshold PVALUE_THRESHOLD
                        Adjusted P-value threshold for filtering results (default 0.05).
  --api_service API_SERVICE
                        API service for topic refinement (default 'openai', also supports 'ollama').
  --api_model API_MODEL
                        Model name for the API service (default 'gpt-4o-mini').
  --api_parallel_jobs API_PARALLEL_JOBS
                        Number of parallel API jobs (default 1).
  --api_base_url API_BASE_URL
                        Base URL for the API service. Required for Ollama (typically "http://localhost:11434/v1").
  --target_filtered_topics TARGET_FILTERED_TOPICS
                        Target number of topics after filtering (default 25).
  --temp_dir TEMP_DIR   Temporary directory for intermediate files.
  --report_title REPORT_TITLE
                        Title for the generated report.
  --species SPECIES     Species identifier (default: 9606 for human).
  -v {none,debug,info,warning,error,critical}, --verbosity {none,debug,info,warning,error,critical}
                        Set logging verbosity. Use 'none' to disable logging. Default is 'none'.
```

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

To customize the report, use the `--report_title` parameter.

## Running Individual Steps

Geneinsight provides individual command-line tools for running each step of the pipeline separately:

### 1. StringDB Enrichment
```bash
geneinsight-enrichment gene_list.txt -o ./output -m single
```

### 2. Topic Modeling
```bash
geneinsight-topic ./output/gene_list__documents.csv -o ./output/topics.csv -n 5 -k 10
```

### 3. Prompt Generation
```bash
geneinsight-prompt ./output/topics.csv -o ./output/prompts.csv -n 5 -w 10
```

### 4. API Processing
```bash
geneinsight-api ./output/prompts.csv -o ./output/api_results.csv --service openai --model gpt-4o-mini
```

### 5. Hypergeometric Enrichment
```bash
geneinsight-hypergeometric ./output/api_results.csv gene_list.txt background_genes.txt -o ./output/enriched.csv -p 0.05
```

### 6. Topic Filtering
```bash
geneinsight-filter ./output/enriched.csv -o ./output/filtered_topics.csv -t 25
```

### 7. Report Generation
```bash
geneinsight-report ./output -o ./report --title "My Gene Analysis"
```

This modular approach gives you complete control over the pipeline and allows you to:
- Experiment with different parameters at each stage
- Resume processing from any point if a step fails
- Run only the specific steps you need
- Integrate individual steps into your own workflows

## License

This project is licensed under the MIT License.