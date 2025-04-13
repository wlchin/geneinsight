## Overview

This workflow processes a collection of 100 validation gene sets through a multi-stage analysis pipeline. It utilizes topic modeling with multiple random seeds to ensure robust results, followed by API-based processing of minor topics, enrichment analysis, and result visualization.

## Workflow Description

The workflow consists of several key stages:

1. **Data Retrieval**: Obtains gene data from StringDB (using a local cache to avoid repeated API calls)
2. **Topic Modeling**: Generates multiple topic models with different random seeds (0-9)
3. **Minor Topic Processing**: 
   - Generates prompts for minor topics
   - Calls an external API to process these prompts
4. **Summary and Filtering**:
   - Creates summaries of topic model outputs
   - Filters gene sets using hypergeometric enrichment analysis
5. **Result Consolidation**:
   - Combines results from multiple seeds
   - Performs second-level topic modeling
6. **Analysis and Visualization**:
   - Calculates ranking metrics and soft cardinality
   - Generates visualization plots

## Input Requirements

- Gene set files in `data/test_set/*.txt` 
- Background gene list in background.txt
- StringDB cache in string_db_cache

## Output Files

The workflow generates various outputs in the results directory:
- Enrichment data (enrichment_df)
- Topic models (topics_for_genelists)
- Minor topics data (minor_topics)
- Summary files (summary)
- Filtered gene sets (filtered_sets)
- Cardinality metrics (`results/cardinality/`)
- Visualization plots:
  - `results/ranking_metrics.png`
  - `results/soft_cardinality.png`
  - `results/combined_metrics.png`

## Usage

This workflow uses Snakemake. To run the complete pipeline:

```bash
snakemake --cores N
```

Where `N` is the number of cores to use for parallel processing.

## Requirements

- Python 3.10
- Snakemake
- Required Python packages (specific requirements to be determined from scripts)
- GPU support for topic modeling

## Notes

- Multiple seeds (0-9) are used to ensure robust results
- The StringDB cache prevents redundant API calls