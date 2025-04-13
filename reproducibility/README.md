# GeneInsight Analysis Workflows

This repository contains a collection of bioinformatics workflows for analyzing gene insights. The workflows are organized into separate modules that work independently or in sequence to perform various analyses.

## Workflow Overview

Here's how the different workflows in this repository interact:

```
test_set_workflow        hyperparameter_workflow
       |                          |
       |                          |
       v                          v
cosine_and_moverscore_workflow    (independent)
       |
       v
visualisation_workflow
```

### Workflow Descriptions

#### 1. Test Set Workflow
Located in `/test_set_workflow/`

This workflow processes test sets of gene data and serves as the foundation for downstream analysis. It performs tasks such as:
- Generating prompts for analysis using `generate_prompts.py`
- Calling APIs in batches with `call_api_combined_batch.py`
- Performing hypergeometric enrichment analysis through `calculate_hypergeometric_enrichment.py`
- Conducting topic modeling via `topic_modelling.py`
- Retrieving StringDB data with `stringdb_retrieval.py`

To run this workflow, refer to the specific README in the test_set_workflow directory.

#### 2. Hyperparameter Workflow
Located in `/hyperparameter_worfklow/`

This workflow is independent and focuses on hyperparameter optimization for gene analysis models. It includes:
- Multi-seed topic modeling with GPU acceleration
- Calculating gene set specific CSVs
- Performing hypergeometric enrichment analysis across multiple seeds
- Various visualization scripts for ranking and evaluation metrics

Refer to the dedicated README in the hyperparameter_workflow directory for detailed usage instructions.

#### 3. Cosine and Moverscore Workflow
Located in `/cosine_and_moverscore_workflow/`

This workflow depends on the test_set_workflow and calculates similarity metrics between gene sets:
- Calculates cosine similarity between vectors
- Computes Mover similarity (based on Word Mover's Distance)
- Generates comparison tables in Markdown format
- Processes group-specific Mover scores

Check the workflow-specific README for execution details.

#### 4. Visualization Workflow
Located in `/visualisation_workflow/`

This workflow depends on the test_set_workflow and potentially the cosine_and_moverscore_workflow. It generates various visualizations for analysis results:
- Combined figures for comprehensive views
- Correlation computations and plots
- Various specialized plots (cosine, mover, pairwise, violin, etc.)
- Processing of overlap data and scores
- Vector embedding distance calculations

See the specific README in the visualization_workflow directory for usage instructions.

## Getting Started

Each workflow has its own requirements file and Snakefile for workflow management. To get started with a specific workflow:

1. Navigate to the workflow directory
2. Install the required dependencies (using pip or conda)
3. Review the workflow-specific README for detailed instructions
4. Execute the workflow using Snakemake

Example:
```bash
cd test_set_workflow
pip install -r requirements.txt
# Follow workflow-specific instructions in the README
snakemake --cores 4
```

## Dependencies

The workflows utilize Snakemake for workflow management. Each workflow directory contains a `requirements.txt` or `environment.yml` file that lists the specific packages required.

## Additional Resources

Please see the individual workflow READMEs for more detailed information on each workflow's functionality, parameters, and outputs.