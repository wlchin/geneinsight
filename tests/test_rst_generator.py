"""Tests for geneinsight.report.rst_generator."""

import os
import pandas as pd
import pytest

from geneinsight.report.rst_generator import generate_rst_files


@pytest.fixture
def headings_csv(tmp_path):
    path = tmp_path / "headings.csv"
    pd.DataFrame({
        "cluster": [0],
        "transcript": ["query1\nquery2"],
        "heading": ["Theme 1 - Cell signaling"],
        "main_heading_text": ["Two-sentence summary here."],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def merged_csv(tmp_path):
    path = tmp_path / "merged.csv"
    pd.DataFrame({
        "query": ["query1"],
        "subheading_text": ["Subheading body."],
        "context": ["Some context."],
        "unique_genes": ["{'GENE1': 1}"],
        "ref_dict": ["{'Ref1 (GO:000001)': 'GENE1'}"],
        "Cluster": [0],
        "ontology_dict": ["{'GO:000001 biological process': 'GENE1'}"],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def filtered_sets_csv_missing_term(tmp_path):
    """A filtered_sets CSV that does NOT contain query1 — should force the lookup miss."""
    path = tmp_path / "filtered.csv"
    pd.DataFrame({
        "Term": ["unrelated_term"],
        "Odds Ratio": [1.5],
        "Adjusted P-value": [0.05],
        "Combined Score": [5.0],
        "P-value": [0.01],
    }).to_csv(path, index=False)
    return path


def test_generate_rst_files_raises_on_missing_term(
    headings_csv, merged_csv, filtered_sets_csv_missing_term, tmp_path
):
    """When a subtitle is not present in filtered_genesets_df, the function must
    raise (via the early subset assertion) rather than silently fabricate values."""
    output_dir = tmp_path / "rst_out"
    csv_folder = tmp_path / "csv_out"

    with pytest.raises(AssertionError, match="clustered.Term must be a subset of enriched.Term"):
        generate_rst_files(
            headings_path=str(headings_csv),
            merged_path=str(merged_csv),
            filtered_sets_path=str(filtered_sets_csv_missing_term),
            output_dir=str(output_dir),
            csv_folder=str(csv_folder),
            taxonomy_id="9606",
            call_ncbi_api=False,
        )


def test_generate_rst_files_raises_early_when_invariant_violated(
    headings_csv, merged_csv, filtered_sets_csv_missing_term, tmp_path
):
    """The function should raise immediately on invariant check, before processing
    any cluster — confirmed by checking that no output files are created."""
    output_dir = tmp_path / "rst_out"
    csv_folder = tmp_path / "csv_out"

    with pytest.raises(AssertionError, match="clustered.Term must be a subset of enriched.Term"):
        generate_rst_files(
            headings_path=str(headings_csv),
            merged_path=str(merged_csv),
            filtered_sets_path=str(filtered_sets_csv_missing_term),
            output_dir=str(output_dir),
            csv_folder=str(csv_folder),
            taxonomy_id="9606",
            call_ncbi_api=False,
        )

    # No partial output should have been created.
    assert not any(output_dir.glob("*.rst")) if output_dir.exists() else True
    assert not any(csv_folder.glob("*.csv")) if csv_folder.exists() else True


import csv

import numpy as np


@pytest.fixture
def headings_csv_with_empirical(tmp_path):
    path = tmp_path / "headings.csv"
    pd.DataFrame({
        "cluster": [0],
        "transcript": ["query1"],
        "heading": ["Theme 1 - Cell signaling"],
        "main_heading_text": ["Two-sentence summary."],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def merged_csv_with_empirical(tmp_path):
    path = tmp_path / "merged.csv"
    pd.DataFrame({
        "query": ["query1"],
        "subheading_text": ["Sub."],
        "context": ["ctx"],
        "unique_genes": ["{'GENE1': 1}"],
        "ref_dict": ["{'Ref1 (GO:1)': 'GENE1'}"],
        "Cluster": [0],
        "ontology_dict": ["{'GO:1': 'GENE1'}"],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def filtered_sets_csv_with_empirical(tmp_path):
    """Filtered gene sets CSV containing p_empirical + q_empirical columns."""
    path = tmp_path / "filtered.csv"
    pd.DataFrame({
        "Term": ["query1"],
        "Odds Ratio": [523.72],
        "Adjusted P-value": [0.005],
        "Combined Score": [2437.93],
        "P-value": [0.001],
        "p_empirical": [0.0031],
        "q_empirical": [0.012],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def filtered_sets_csv_without_empirical(tmp_path):
    """Filtered gene sets CSV WITHOUT empirical columns (models --empirical-p not used)."""
    path = tmp_path / "filtered.csv"
    pd.DataFrame({
        "Term": ["query1"],
        "Odds Ratio": [523.72],
        "Adjusted P-value": [0.005],
        "Combined Score": [2437.93],
        "P-value": [0.001],
    }).to_csv(path, index=False)
    return path


def test_theme_csv_includes_empirical_columns_when_present(
    headings_csv_with_empirical,
    merged_csv_with_empirical,
    filtered_sets_csv_with_empirical,
    tmp_path,
):
    """When filtered_genesets_df has p_empirical + q_empirical, the per-theme
    CSV must include empirical_p and empirical_q columns with the correct
    values, positioned between fdr and combined_score."""
    output_dir = tmp_path / "rst_out"
    csv_folder = tmp_path / "csv_out"

    generate_rst_files(
        headings_path=str(headings_csv_with_empirical),
        merged_path=str(merged_csv_with_empirical),
        filtered_sets_path=str(filtered_sets_csv_with_empirical),
        output_dir=str(output_dir),
        csv_folder=str(csv_folder),
        taxonomy_id="9606",
        call_ncbi_api=False,
    )

    theme_csvs = list(csv_folder.glob("Theme_*.csv"))
    assert len(theme_csvs) == 1
    with open(theme_csvs[0]) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        row = next(reader)

    assert "empirical_p" in headers
    assert "empirical_q" in headers
    assert headers.index("fdr") < headers.index("empirical_p")
    assert headers.index("empirical_p") < headers.index("empirical_q")
    assert headers.index("empirical_q") < headers.index("combined_score")
    assert row["empirical_p"] == "0.003"
    assert row["empirical_q"] == "0.012"


def test_theme_csv_omits_empirical_columns_when_absent(
    headings_csv_with_empirical,
    merged_csv_with_empirical,
    filtered_sets_csv_without_empirical,
    tmp_path,
):
    """When filtered_genesets_df lacks the empirical columns, the per-theme
    CSV must keep the original 6-column format (backwards-compat: today's
    downloads must not sprout new columns)."""
    output_dir = tmp_path / "rst_out"
    csv_folder = tmp_path / "csv_out"

    generate_rst_files(
        headings_path=str(headings_csv_with_empirical),
        merged_path=str(merged_csv_with_empirical),
        filtered_sets_path=str(filtered_sets_csv_without_empirical),
        output_dir=str(output_dir),
        csv_folder=str(csv_folder),
        taxonomy_id="9606",
        call_ncbi_api=False,
    )

    theme_csvs = list(csv_folder.glob("Theme_*.csv"))
    assert len(theme_csvs) == 1
    with open(theme_csvs[0]) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    assert headers == ["subtitle", "odds_ratio", "p-value", "fdr", "combined_score", "gene_set"]
