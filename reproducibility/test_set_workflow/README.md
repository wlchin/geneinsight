# Gene Set Analysis Workflow

This repository implements a Snakemake-based pipeline to process gene sets and extract meaningful topics and summaries from their enrichment data. The workflow is divided into several stages—from raw data processing to topic modeling, API-based minor topic extraction, and finally filtering and summarization. Each step leverages custom Python scripts to perform domain-specific tasks.

## Overview

The pipeline is designed for the following steps:

1. **Data Retrieval and Enrichment**  
   The workflow starts by processing gene set files and then fetching enrichment information from STRINGdb. Two modes of gene retrieval are supported: a default mode and a “list” mode.

2. **Topic Modeling**  
   A topic modeling algorithm is applied to the enrichment documents. This creates multiple topic model outputs using different seeds and sample sizes.

3. **Prompt Generation and Minor Topic Extraction**  
   Prompts are generated from the topic model outputs. These prompts are then used to call an API that extracts minor topics.

4. **Summary and Filtering**  
   The workflow generates a summary by combining the enrichment data with the extracted minor topics. It then applies hypergeometric enrichment analysis to filter gene sets.  
   Additional topic modeling is run on the filtered sets to identify key topics. Finally, these key topics are filtered according to several similarity thresholds.

5. **Final Outputs**  
   The final outputs include CSV files capturing the enrichment data, topic modeling results, minor topics, summaries, and various levels of filtered topics based on user-defined thresholds.

## Workflow Details

### 1. Data Retrieval and Enrichment

- **Input Data:**  
  Gene set files (in text format) are stored in `data/test_gene_sets/sampled_gene_sets/`.

- **Enrichment Retrieval:**  
  - **`get_stringdb_genes`:** Retrieves documents and enrichment information from STRINGdb.
  - **`get_stringdb_genes_in_list`:** Uses the “list” mode for enrichment retrieval.

- **Output:**  
  Files such as `results/enrichment_df_testset/{gene_set}__documents.csv` and `results/enrichment_df_testset/{gene_set}__enrichment.csv` are generated.

### 2. Topic Modeling

- **`get_topic_model`:**  
  Runs a topic model on the STRINGdb enrichment documents.  
  - **Input:** Enrichment documents.  
  - **Output:** A topic model CSV file stored in `results/topics_for_genelists/{gene_set}_topic_model.csv`.

### 3. Prompt Generation & API Minor Topic Extraction

- **`generate_prompts_for_minor_topics`:**  
  Generates a prompts CSV from the topic model output.  
  - **Output:** `results/prompts_for_minor_topics/{gene_set}_prompts.csv`.

- **`call_api_for_minor_topics`:**  
  Calls an external API to extract minor topics based on the generated prompts.  
  - **Output:** `results/minor_topics/{gene_set}_minor_topics.csv`.

### 4. Summary and Filtering

- **`get_summary`:**  
  Combines the enrichment data and minor topics to generate a summary.  
  - **Output:** `results/summary/{gene_set}.csv`.

- **`get_filtered_genesets`:**  
  Applies hypergeometric enrichment analysis to filter the gene sets based on the summary data.  
  - **Output:** `results/filtered_sets/{gene_set}_filtered_gene_sets.csv`.

### 5. Topic Modeling on Filtered Gene Sets & Key Topic Extraction

- **`topic_model_on_topic_model`:**  
  Runs an additional topic modeling step on the filtered gene sets to refine topic extraction.  
  - **Output:** `results/resampled_topics/{gene_set}_final_topic_modeling_results.csv`.

- **`get_key_topics`:**  
  Extracts the top terms (key topics) from the final topic modeling results.  
  - **Output:** `results/key_topics/{gene_set}_key_topics.csv`.

### 6. Topic Filtering

Multiple rules filter the key topics based on similarity thresholds to offer different levels of granularity:
- **`filter_by_sim`:** (threshold 50)  
  - **Output:** `results/filtered_topics/samp_50/{gene_set}_filtered_topics.csv`
- **`filter_by_sim_25`:** (threshold 25)  
  - **Output:** `results/filtered_topics/samp_25/{gene_set}_filtered_topics.csv`
- **`filter_by_sim_75`:** (threshold 75)  
  - **Output:** `results/filtered_topics/samp_75/{gene_set}_filtered_topics.csv`
- **`filter_by_sim_100`:** (threshold 100)  
  - **Output:** `results/filtered_topics/samp_100/{gene_set}_filtered_topics.csv`

## Required Scripts

The following Python scripts (located in the `scripts/` directory) are utilized in the workflow:

- `scripts/stringdb_retrieval.py`  
  *(Used for retrieving STRINGdb data in default and list modes.)*
- `scripts/topic_modelling_mutli_seed.py`  
  *(Performs topic modeling on the enrichment documents.)*
- `scripts/generate_prompts_new.py`  
  *(Generates prompts from topic modeling outputs.)*
- `scripts/call_api_combined_batch_multiseed.py`  
  *(Calls the API to extract minor topics based on generated prompts.)*
- `scripts/get_summary_multi_seed.py`  
  *(Generates a summary CSV from enrichment data and minor topics.)*
- `scripts/calculate_hypergeometric_enrichment_multiseed.py`  
  *(Performs hypergeometric enrichment analysis for filtering gene sets.)*
- `scripts/final_main_topics_for_model_on_model.py`  
  *(Applies a secondary topic modeling approach on filtered gene sets.)*
- `scripts/count_top_topics.py`  
  *(Identifies and counts key topics from the final topic modeling outputs.)*
- `scripts/filter_by_sim.py`  
  *(Filters key topics based on a given similarity threshold.)*

## Prerequisites

- **Python 3.10+**  
- **Snakemake**  
- Additional Python packages required by the scripts (e.g., `pandas`, `numpy`, etc.)

Make sure to set up your Python environment and install all the necessary dependencies. You may use a virtual environment and install the dependencies via `pip` as needed.

## Usage

1. **Configure Your Data:**  
   Place your gene set text files under `data/test_gene_sets/sampled_gene_sets/` and ensure the background genes file is at `data/background.txt`.

2. **Run the Workflow:**  
   Execute the following command in your terminal:
   ```bash
   snakemake --cores <num_cores>
