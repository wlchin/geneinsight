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
