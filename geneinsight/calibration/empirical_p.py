"""Core empirical p-value math.

Statistic options:
    raw      : raw hypergeometric P from gseapy (smaller = more extreme).
               Better calibration (LOO KS=0.010 on the bundled pool).
    overlap  : overlap ratio k/m from the "Overlap" column (larger = more extreme).
               Directly answers Reviewer 3's framing but coarser (LOO KS=0.067).

The empirical p uses the Davison-Hinkley +1/+1 correction:
    p_emp = (1 + count_more_extreme) / (1 + n_null_in_bin)
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 10 fixed bins covering query gene-set sizes 1-500. Must match the bin labels
# used when the bundled null pool was built (see calibration/data/manifest.json).
BINS = [
    (1, 50), (51, 100), (101, 150), (151, 200), (201, 250),
    (251, 300), (301, 350), (351, 400), (401, 450), (451, 500),
]
BIN_LABELS = [f"{lo}-{hi}" for lo, hi in BINS]

Statistic = Literal["raw", "overlap"]


def assign_bin(gene_count: int) -> Optional[str]:
    """Return the bin label for a query gene count, or None if outside 1-500."""
    for lo, hi in BINS:
        if lo <= gene_count <= hi:
            return f"{lo}-{hi}"
    return None


def parse_overlap(overlap_str: object) -> float:
    """Parse gseapy 'n/m' overlap string to a float ratio; NaN on failure."""
    if pd.isna(overlap_str):
        return float("nan")
    try:
        a, b = str(overlap_str).split("/")
        a_i, b_i = int(a), int(b)
        return a_i / b_i if b_i > 0 else float("nan")
    except (ValueError, ZeroDivisionError):
        return float("nan")


def compute_empirical_p(
    observed: np.ndarray,
    sorted_null: np.ndarray,
    statistic: Statistic,
) -> np.ndarray:
    """Vectorized empirical p against a single pre-sorted null array.

    Args:
        observed: 1-D array of observed statistic values.
        sorted_null: 1-D array of null values, sorted ascending.
        statistic: "raw" (smaller = more extreme) or "overlap" (larger = more extreme).

    Returns:
        1-D array of empirical p-values in (0, 1].
    """
    observed = np.asarray(observed, dtype=float)
    n = len(sorted_null)
    if n == 0:
        return np.full_like(observed, np.nan)

    if statistic == "raw":
        # Count nulls with p <= observed (more extreme).
        count = np.searchsorted(sorted_null, observed, side="right")
    elif statistic == "overlap":
        # Count nulls with overlap >= observed (more extreme).
        count = n - np.searchsorted(sorted_null, observed, side="left")
    else:
        raise ValueError(f"Unknown statistic {statistic!r}; expected 'raw' or 'overlap'")

    return (1 + count) / (1 + n)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values, clipped to [0, 1].

    Matches the formula used in `bh_doing_real_work/compute_empirical_p_overlap.py`
    and `scipy.stats.false_discovery_control(method="bh")`.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def annotate_enriched(
    df: pd.DataFrame,
    gene_count: int,
    null_pool,  # NullPool (avoid circular import in signature)
    statistic: Statistic = "raw",
) -> pd.DataFrame:
    """Append p_empirical and q_empirical columns to an enriched DataFrame.

    Args:
        df: DataFrame from `enriched.csv` or `enriched_unfiltered.csv`.
            Must contain a 'P-value' column (raw) and/or 'Overlap' column (n/m).
        gene_count: Query gene-set size (deduplicated count).
        null_pool: a `NullPool` instance loaded for `statistic`.
        statistic: "raw" or "overlap".

    Returns:
        A new DataFrame with two appended columns. If the bin is out of range
        or no valid observations exist, the new columns are filled with NaN
        and the rest of the DataFrame is returned unchanged.
    """
    out = df.copy()
    out["p_empirical"] = np.nan
    out["q_empirical"] = np.nan

    if out.empty:
        return out

    bin_label = assign_bin(gene_count)
    if bin_label is None:
        logger.warning(
            "Query gene count %d is outside the supported empirical-p range "
            "(1-500). Skipping empirical-p annotation.", gene_count
        )
        return out

    try:
        sorted_null = null_pool.sorted_bin(bin_label)
    except KeyError:
        logger.warning(
            "Null pool has no entries for bin %s; skipping empirical-p.", bin_label
        )
        return out

    if statistic == "raw":
        if "P-value" not in out.columns:
            logger.warning(
                "Cannot compute raw-P empirical recalibration: 'P-value' column missing."
            )
            return out
        observed = out["P-value"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(observed)
    elif statistic == "overlap":
        if "Overlap" not in out.columns:
            logger.warning(
                "Cannot compute overlap empirical recalibration: 'Overlap' column missing."
            )
            return out
        observed = out["Overlap"].apply(parse_overlap).to_numpy(dtype=float)
        valid_mask = ~np.isnan(observed)
    else:
        raise ValueError(f"Unknown statistic {statistic!r}")

    if not valid_mask.any():
        return out

    p_emp = compute_empirical_p(observed[valid_mask], sorted_null, statistic)
    q_emp = bh_adjust(p_emp)

    out.loc[valid_mask, "p_empirical"] = p_emp
    out.loc[valid_mask, "q_empirical"] = q_emp
    return out
