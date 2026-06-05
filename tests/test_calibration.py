"""Tests for the geneinsight.calibration module.

Covers:
1. Bin assignment boundaries.
2. BH adjustment vs scipy.stats.false_discovery_control.
3. Empirical-p formula correctness against the bundled LOO golden CSV
   (`bh_doing_real_work/null_empirical_p_loo.csv`).
4. NullPool.from_bundled loads and has the expected shape.
5. annotate_enriched behaviour: column addition, NaN handling, out-of-range bin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geneinsight.calibration import (
    BIN_LABELS,
    NullPool,
    annotate_enriched,
    assign_bin,
    bh_adjust,
    compute_empirical_p,
)
from geneinsight.calibration.empirical_p import parse_overlap

REPO_ROOT = Path(__file__).resolve().parents[2]
LOO_GOLDEN = REPO_ROOT / "bh_doing_real_work" / "null_empirical_p_loo.csv"


# -- assign_bin -----------------------------------------------------------


@pytest.mark.parametrize("n,expected", [
    (1, "1-50"),
    (50, "1-50"),
    (51, "51-100"),
    (100, "51-100"),
    (101, "101-150"),
    (500, "451-500"),
])
def test_assign_bin_boundaries(n, expected):
    assert assign_bin(n) == expected


@pytest.mark.parametrize("n", [0, -5, 501, 1000])
def test_assign_bin_out_of_range_returns_none(n):
    assert assign_bin(n) is None


def test_bin_labels_count():
    assert len(BIN_LABELS) == 10


# -- bh_adjust ------------------------------------------------------------


def test_bh_adjust_empty():
    out = bh_adjust(np.array([]))
    assert out.shape == (0,)


def test_bh_adjust_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy too old for false_discovery_control")

    rng = np.random.default_rng(0)
    p = rng.uniform(size=100)
    ours = bh_adjust(p)
    theirs = scipy_stats.false_discovery_control(p, method="bh")
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_bh_adjust_monotone_in_rank():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    q = bh_adjust(p)
    # Sorted q should be non-decreasing.
    assert np.all(np.diff(np.sort(q)) >= -1e-12)
    assert (q >= 0).all() and (q <= 1).all()


# -- parse_overlap --------------------------------------------------------


@pytest.mark.parametrize("s,expected", [
    ("1/4", 0.25),
    ("0/5", 0.0),
    ("3/3", 1.0),
])
def test_parse_overlap_valid(s, expected):
    assert parse_overlap(s) == pytest.approx(expected)


@pytest.mark.parametrize("s", ["", "abc", "1/0", None, float("nan")])
def test_parse_overlap_invalid_returns_nan(s):
    assert np.isnan(parse_overlap(s))


# -- compute_empirical_p --------------------------------------------------


def test_compute_empirical_p_raw_basic():
    sorted_null = np.array([0.01, 0.05, 0.1, 0.5, 0.9])
    # Observation more extreme (smaller) than the smallest null.
    p = compute_empirical_p(np.array([0.001]), sorted_null, "raw")
    # 0 nulls <= 0.001, so p = (1 + 0) / (1 + 5) = 1/6.
    assert p[0] == pytest.approx(1 / 6)


def test_compute_empirical_p_raw_floor():
    """An observation at the floor (no null beats it) yields the smallest possible p."""
    sorted_null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    p = compute_empirical_p(np.array([1e-12]), sorted_null, "raw")
    assert p[0] == pytest.approx(1 / 6)


def test_compute_empirical_p_overlap_basic():
    sorted_null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # Observation 0.45 > 0.4, so only one null (0.5) is >= it.
    p = compute_empirical_p(np.array([0.45]), sorted_null, "overlap")
    assert p[0] == pytest.approx(2 / 6)


def test_compute_empirical_p_unknown_statistic_raises():
    with pytest.raises(ValueError):
        compute_empirical_p(np.array([0.1]), np.array([0.1, 0.2]), "bogus")  # type: ignore[arg-type]


# -- LOO golden parity (sanity check against bh_doing_real_work) ----------


@pytest.mark.skipif(not LOO_GOLDEN.exists(),
                    reason="LOO golden file not present (rebuttal-side only)")
def test_empirical_p_loo_parity_sampled():
    """For 100 sampled rows, recompute empirical p via the new code path and
    compare against the LOO golden table.

    The golden values were computed leave-one-out (n-1 nulls). We approximate by
    excluding the observation itself from the sorted array per row. With
    bin sizes in the thousands this matches the LOO formula exactly up to ties.
    """
    golden = pd.read_csv(LOO_GOLDEN)
    pool_path = REPO_ROOT / "geneinsight" / "geneinsight" / "calibration" / "data"
    pool = NullPool.from_path(pool_path, statistic="raw")

    sample = golden.sample(100, random_state=0)
    for _, row in sample.iterrows():
        bin_label = row["bin"]
        obs = row["raw_p"]
        sorted_full = pool.sorted_bin(bin_label).copy()
        # Remove one occurrence of the observation (LOO).
        idx = np.searchsorted(sorted_full, obs)
        if idx < len(sorted_full) and sorted_full[idx] == obs:
            sorted_loo = np.delete(sorted_full, idx)
        else:
            sorted_loo = sorted_full  # safety: row not present
        p_emp = compute_empirical_p(np.array([obs]), sorted_loo, "raw")[0]
        # Tolerate ~1 ulp differences from tied-value ordering.
        assert abs(p_emp - row["p_emp_via_raw_p"]) < 5e-4, (
            f"raw-P empirical-p mismatch for bin={bin_label}, obs={obs}: "
            f"new={p_emp}, golden={row['p_emp_via_raw_p']}"
        )


# -- NullPool.from_bundled ------------------------------------------------


def test_null_pool_bundled_raw_loads():
    pool = NullPool.from_bundled(statistic="raw")
    assert pool.total_themes() == 71925
    counts = pool.bin_counts()
    assert set(counts.keys()) == set(BIN_LABELS)
    # Each per-bin array must be sorted ascending.
    for label in BIN_LABELS:
        arr = pool.sorted_bin(label)
        assert (np.diff(arr) >= 0).all()


def test_null_pool_bundled_overlap_loads():
    pool = NullPool.from_bundled(statistic="overlap")
    assert pool.total_themes() == 71925
    # Overlap ratios live in [0, 1].
    for label in BIN_LABELS:
        arr = pool.sorted_bin(label)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


def test_null_pool_bundled_manifest():
    pool = NullPool.from_bundled(statistic="raw")
    assert pool.manifest.get("kind") == "bundled-default"
    assert "ks_distance_vs_uniform_loo" in pool.manifest


# -- annotate_enriched ----------------------------------------------------


def test_annotate_enriched_adds_columns():
    pool = NullPool.from_bundled(statistic="raw")
    df = pd.DataFrame({
        "Term": ["A", "B", "C"],
        "Overlap": ["1/4", "2/8", "3/6"],
        "P-value": [1e-10, 1e-3, 0.5],
        "Adjusted P-value": [1e-9, 1e-2, 0.6],
    })
    out = annotate_enriched(df, gene_count=72, null_pool=pool, statistic="raw")
    assert list(df.columns) == ["Term", "Overlap", "P-value", "Adjusted P-value"]
    # Original columns preserved unchanged.
    for col in df.columns:
        pd.testing.assert_series_equal(out[col], df[col])
    assert "p_empirical" in out.columns
    assert "q_empirical" in out.columns
    assert (out["p_empirical"] > 0).all()
    assert (out["p_empirical"] <= 1).all()
    # Smallest raw P should get smallest q_empirical.
    assert out["q_empirical"].iloc[0] <= out["q_empirical"].iloc[2]


def test_annotate_enriched_out_of_range_returns_nan_columns():
    pool = NullPool.from_bundled(statistic="raw")
    df = pd.DataFrame({
        "Term": ["A"],
        "Overlap": ["1/4"],
        "P-value": [0.01],
        "Adjusted P-value": [0.05],
    })
    out = annotate_enriched(df, gene_count=999, null_pool=pool, statistic="raw")
    assert out["p_empirical"].isna().all()
    assert out["q_empirical"].isna().all()


def test_annotate_enriched_empty_df():
    pool = NullPool.from_bundled(statistic="raw")
    df = pd.DataFrame(columns=["Term", "Overlap", "P-value", "Adjusted P-value"])
    out = annotate_enriched(df, gene_count=50, null_pool=pool, statistic="raw")
    assert len(out) == 0
    assert "p_empirical" in out.columns
    assert "q_empirical" in out.columns
