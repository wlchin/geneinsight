"""Empirical p-value recalibration for Step 6 of the GeneInsight pipeline.

Provides a size-stratified, leave-one-out empirical p-value computed against a
pre-built null pool of random-query pipeline runs. Designed as an additive layer
to the existing parametric `enriched.csv` output: when enabled it appends
`p_empirical` and `q_empirical` columns without changing existing rows or columns.

Public API:
    NullPool                  Load a null pool (bundled default or user-supplied path).
    compute_empirical_p       Vectorized lookup of empirical p for one bin.
    bh_adjust                 Within-run Benjamini-Hochberg adjustment.
    assign_bin                Map a query gene count to one of the 10 size bins.
    annotate_enriched         End-to-end: take enriched/unfiltered df + gene_count,
                              return df with p_empirical and q_empirical columns.
"""
from .empirical_p import (
    BINS,
    BIN_LABELS,
    annotate_enriched,
    assign_bin,
    bh_adjust,
    compute_empirical_p,
)
from .null_pool import NullPool

__all__ = [
    "BINS",
    "BIN_LABELS",
    "NullPool",
    "annotate_enriched",
    "assign_bin",
    "bh_adjust",
    "compute_empirical_p",
]
