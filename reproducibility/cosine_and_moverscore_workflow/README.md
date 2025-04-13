# 📈 Similarity Score Subworkflow

This subworkflow calculates and compares similarity scores between enriched gene sets and their topic summaries at different summarisation levels and sample sizes. It is designed to assess alignment between gene set enrichment results and topic model summaries.

## 🔗 Upstream Dependency

This subworkflow **originates from the main workflow** and consumes the following outputs:

- `data/filtered_sets/*.csv`: filtered/enriched gene sets
- `data/filtered_topics/samp_{N}/*.csv`: topic summaries at various summarisation levels

The variable `N` represents the sample size and takes values from `{25, 50, 75, 100}`.

These outputs are also used in the **Top-K folder** for downstream visualisation.

---

## ⚙️ Rules Overview

### `rule all`

Defines the final targets:
- `results/mover_score_results_{N}.csv`
- `results/cosine_score_results_{N}.csv`

for all values of `N`.

---

### `rule moverscore`

Calculates similarity between each gene set and topic summary using the **MoverScore** method.

- **Inputs:**  
  `data/filtered_topics/samp_{N}/{gene_set}_filtered_topics.csv`

- **Outputs:**
  - `results/mover_score_results_{N}.csv`: table of MoverScore values
  - `results/mover_score_scatter_{N}.png`: scatterplot of scores

- **Script Used:**  
  `scripts/calculate_mover_similarity.py`

---

### `rule group_mover_score_results`

Aggregates all MoverScore CSVs into a markdown summary table.

- **Output:**  
  `results/table_mover.md`

- **Script Used:**  
  `scripts/group_mover.py`

---

### `rule cosinescore`

Computes cosine similarity between enriched gene sets and topic summaries.

- **Inputs:**  
  `data/filtered_topics/samp_{N}/{gene_set}_filtered_topics.csv`

- **Outputs:**
  - `results/cosine_score_results_{N}.csv`: cosine similarity results
  - `results/cosine_score_scatter_{N}.png`: scatterplot of scores

- **Script Used:**  
  `scripts/calculate_cosine_similarity.py`

---

### `rule generate_cosine_markdown`

Aggregates cosine similarity results into a markdown summary.

- **Output:**  
  `results/cosine_table.md`

- **Script Used:**  
  `scripts/generate_cosine_markdown_table.py`

---

### `rule generate_correlation_table`

Calculates Pearson correlation between MoverScore and CosineScore results.

- **Output:**  
  `results/correlation_table.md`

- **Script Used:**  
  `scripts/generate_pearson_markdown_table.py`

---

## 📁 Output Summary

| File                                   | Description                                      |
|----------------------------------------|--------------------------------------------------|
| `results/mover_score_results_{N}.csv`  | MoverScore results per sample size              |
| `results/cosine_score_results_{N}.csv` | Cosine similarity results per sample size       |
| `results/*_scatter_{N}.png`            | Scatterplots for each scoring method            |
| `results/table_mover.md`              | Combined markdown table of MoverScores          |
| `results/cosine_table.md`             | Combined markdown table of CosineScores         |
| `results/correlation_table.md`        | Correlation between Mover and Cosine scores     |

---

## 🧪 Sample Sizes Used

This subworkflow iterates over the following sample sizes:

- 25 terms
- 50 terms
- 75 terms
- 100 terms

---

## 📜 Script Definitions

### `calculate_mover_similarity.py`

A script that calculates the MoverScore between gene sets and their topic summaries.

**Key functions:**
- Uses BERT embeddings to compute word mover's distance
- Processes gene sets in batches for memory efficiency
- Calculates top-k recall metrics across different k values
- Generates scatter plots of scores vs. document size

**Dependencies:**
- Transformers (BERT)
- Optimal Transport (OT)
- PyTorch

---

### `calculate_cosine_similarity.py`

Computes cosine similarity between gene sets and topic summaries.

**Key functions:**
- Uses Sentence Transformers for text embedding
- Computes cosine similarity matrices between source and summary texts
- Calculates top-k recall metrics for different values of k
- Generates plots showing the relationship between document size and cosine similarity

**Dependencies:**
- Sentence Transformers
- PyTorch

---

### `group_mover.py`

Aggregates MoverScore results across different sample sizes.

**Key functions:**
- Groups results by top-k values
- Calculates mean and standard error (SEM) for each group
- Generates a markdown table summarizing the results

---

### `generate_cosine_markdown_table.py`

Creates a markdown table summarizing cosine similarity results.

**Key functions:**
- Processes cosine similarity CSV files from different sample sizes
- Calculates mean and standard error for each top-k value
- Formats results into a readable markdown table

---

### `generate_pearson_markdown_table.py`

Calculates correlations between MoverScore and cosine similarity metrics.

**Key functions:**
- Computes Pearson correlation coefficients between scoring methods
- Calculates t-statistics and p-values for statistical significance
- Generates a markdown table showing correlations for each sample size

**Statistics calculated:**
- Pearson correlation coefficient (r)
- t-statistic
- p-value

