# TopicGenes

A Python package for topic modeling of gene sets with enrichment analysis. 

## Overview

TopicGenes provides a comprehensive pipeline for analyzing gene sets through topic modeling and enrichment analysis:

1. **Gene Enrichment**: Query StringDB for gene enrichment data
2. **Topic Modeling**: Apply BERTopic to identify themes in gene sets
3. **Prompt Generation**: Generate prompts for language models based on topic data
4. **API Integration**: Use language models to refine topics and create meaningful annotations
5. **Enrichment Analysis**: Perform hypergeometric enrichment analysis on identified gene sets
6. **Results Packaging**: Export all results in a zipped format for easy sharing and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/topicgenes.git
cd topicgenes

# Install the package
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/yourusername/topicgenes.git
```

## Quick Start

Run the pipeline on your gene sets:

```bash
# Basic usage
topicgenes query_genes.txt background_genes.txt -o ./output

# Advanced usage with custom parameters
topicgenes query_genes.txt background_genes.txt \
  --n-samples 5 \
  --num-topics 10 \
  --pvalue-threshold 0.01 \
  --api-service openai \
  --api-model gpt-4o-mini
```

## Command Line Interface

```
usage: topicgenes [-h] [-o OUTPUT_DIR] [-t TEMP_DIR] [-n N_SAMPLES]
                  [-k NUM_TOPICS] [-p PVALUE_THRESHOLD] [--api-service API_SERVICE]
                  [--api-model API_MODEL] [--api-parallel-jobs API_PARALLEL_JOBS]
                  [--api-base-url API_BASE_URL]
                  [--target-filtered-topics TARGET_FILTERED_TOPICS] [--no-zip]
                  [-c CONFIG] [-v]
                  query_gene_set background_gene_list

TopicGenes: Topic modeling pipeline for gene sets with enrichment analysis

positional arguments:
  query_gene_set        Path to file containing query gene set
  background_gene_list  Path to file containing background gene list

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to store outputs (default: ./output)
  -t TEMP_DIR, --temp-dir TEMP_DIR
                        Directory for temporary files (default: system temp directory)
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Number of topic models to run with different seeds (default: 5)
  -k NUM_TOPICS, --num-topics NUM_TOPICS
                        Number of topics to extract in topic modeling (default: 10)
  -p PVALUE_THRESHOLD, --pvalue-threshold PVALUE_THRESHOLD
                        Adjusted P-value threshold for filtering results (default: 0.01)
  --api-service API_SERVICE
                        API service for topic refinement (default: openai)
  --api-model API_MODEL
                        Model name for the API service (default: gpt-4o-mini)
  --api-parallel-jobs API_PARALLEL_JOBS
                        Number of parallel API jobs (default: 4)
  --api-base-url API_BASE_URL
                        Base URL for the API service
  --target-filtered-topics TARGET_FILTERED_TOPICS
                        Target number of topics after filtering (default: 25)
  --no-zip              Do not zip the output directory
  -c CONFIG, --config CONFIG
                        Path to configuration file (JSON or YAML)
  -v, --version         show program's version number and exit
```

## Using Configuration Files

You can use configuration files (JSON or YAML) to specify parameters:

```yaml
# config.yaml
n_samples: 5
num_topics: 15
pvalue_threshold: 0.005
api_service: "openai"
api_model: "gpt-4o-mini"
api_parallel_jobs: 8
api_base_url: null
target_filtered_topics: 20
```

Then use it with:

```bash
topicgenes query_genes.txt background_genes.txt --config config.yaml
```

## Environment Variables

Set the following environment variables for API access:

```
OPENAI_API_KEY=your_openai_key_here
TOGETHER_API_KEY=your_together_key_here
```

## Output Format

The pipeline produces a zip file containing:

- `enrichment.csv`: Gene enrichment data from StringDB
- `documents.csv`: Document descriptions for topic modeling
- `topics.csv`: Topic modeling results
- `prompts.csv`: Generated prompts for API
- `api_results.csv`: Results from API calls
- `summary.csv`: Summary of topic modeling and enrichment
- `enriched.csv`: Hypergeometric enrichment results
- `filtered.csv`: Final filtered topics
- `metadata.csv`: Run information and parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Command Line Interface

```
usage: topicgenes [-h] [-o OUTPUT_DIR] [-t TEMP_DIR] [-n N_SAMPLES]
                  [-k NUM_TOPICS] [-p PVALUE_THRESHOLD] [--api-service API_SERVICE]
                  [--api-model API_MODEL] [--api-parallel-jobs API_PARALLEL_JOBS]
                  [--api-base-url API_BASE_URL]
                  [--target-filtered-topics TARGET_FILTERED_TOPICS] [--no-zip]
                  [-c CONFIG] [-v]
                  query_gene_set background_gene_list

TopicGenes: Topic modeling pipeline for gene sets with enrichment analysis

positional arguments:
  query_gene_set        Path to file containing query gene set
  background_gene_list  Path to file containing background gene list

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to store outputs (default: ./output)
  -t TEMP_DIR, --temp-dir TEMP_DIR
                        Directory for temporary files (default: system temp directory)
  -n N_SAMPLES, --n-samples N_SAMPLES
                        Number of topic models to run with different seeds (default: 5)
  -k NUM_TOPICS, --num-topics NUM_TOPICS
                        Number of topics to extract in topic modeling (default: 10)
  -p PVALUE_THRESHOLD, --pvalue-threshold PVALUE_THRESHOLD
                        Adjusted P-value threshold for filtering results (default: 0.01)
  --api-service API_SERVICE
                        API service for topic refinement (default: openai)
  --api-model API_MODEL
                        Model name for the API service (default: gpt-4o-mini)
  --api-parallel-jobs API_PARALLEL_JOBS
                        Number of parallel API jobs (default: 4)
  --api-base-url API_BASE_URL
                        Base URL for the API service
  --target-filtered-topics TARGET_FILTERED_TOPICS
                        Target number of topics after filtering (default: 25)
  --no-zip              Do not zip the output directory
  -c CONFIG, --config CONFIG
                        Path to configuration file (JSON or YAML)
  -v, --version         show program's version number and exit
```

## Running Individual Steps

TopicGenes also provides individual command-line tools for running each step of the pipeline separately:

### 1. StringDB Enrichment
```bash
topicgenes-enrichment gene_list.txt -o ./output -m single
```

### 2. Topic Modeling
```bash
topicgenes-topic ./output/gene_list__documents.csv -o ./output/topics.csv -n 5 -k 10
```

### 3. Prompt Generation
```bash
topicgenes-prompt ./output/topics.csv -o ./output/prompts.csv -n 5 -w 10
```

### 4. API Processing
```bash
topicgenes-api ./output/prompts.csv -o ./output/api_results.csv --service openai --model gpt-4o-mini
```

### 5. Hypergeometric Enrichment
```bash
topicgenes-hypergeometric ./output/api_results.csv gene_list.txt background_genes.txt -o ./output/enriched.csv -p 0.01
```

### 6. Topic Filtering
```bash
topicgenes-filter ./output/enriched.csv -o ./output/filtered_topics.csv -t 25
```

This modular approach gives you complete control over the pipeline and allows you to:
- Experiment with different parameters at each stage
- Resume processing from any point if a step fails
- Run only the specific steps you need
- Integrate individual steps into your own workflows