# GeneInsight Examples

This folder contains example files for running GeneInsight analysis:

- `sample.txt`: An example query gene set containing gene symbols
- `sample_background.txt`: A background gene set for comparison

## Getting Started

To run a basic analysis with these examples:

```bash
# Run the basic pipeline
geneinsight sample.txt sample_background.txt -o ./my_results
```

## Try Different API Models

### Using OpenAI Models
```bash
# Using GPT-4o-mini (default)
geneinsight sample.txt sample_background.txt -o ./openai_results \
  --api_service openai \
  --api_model gpt-4o-mini

# Using GPT-4o
geneinsight sample.txt sample_background.txt -o ./openai_gpt4o_results \
  --api_service openai \
  --api_model gpt-4o
```

### Using Ollama with Local Models
```bash
# Using Llama 3.1 with Ollama
geneinsight sample.txt sample_background.txt -o ./ollama_results \
  --api_service ollama \
  --api_model llama3.1:8b \
  --api_base_url "http://localhost:11434/v1"
```

## Increasing Parallel API Calls

Speed up your analysis by processing multiple topics simultaneously:

```bash
# Process 4 topics in parallel
geneinsight sample.txt sample_background.txt -o ./parallel_results \
  --api_parallel_jobs 4
```

## Report Generation

Customize your analysis reports:

```bash
# Generate report with custom title
geneinsight sample.txt sample_background.txt -o ./report_results \
  --report_title "My Gene Analysis"

# Run analysis without generating a report
geneinsight sample.txt sample_background.txt -o ./no_report_results \
  --no-report
```

## File Format

### Query Gene Set (sample.txt)
The query gene set file contains the genes of interest, one gene symbol per line:

```
BRCA1
TP53
PTEN
...
```

### Background Gene Set (sample_background.txt)
The background gene set contains all genes that should be considered in the analysis. This typically includes all genes expressed in your dataset or all genes in the genome:

```
AARS
ABCA1
ABCA2
...
```

## Important Notes

- **Case Sensitivity**: Gene symbols are case-sensitive. Ensure that gene names in both files use consistent capitalization (typically uppercase for human genes).

- **Intersection Requirement**: There must be an intersection between the query gene set and the background gene set. All genes in your query set should be present in the background set for proper statistical analysis.

For more options, run:
```bash
geneinsight --help
```
