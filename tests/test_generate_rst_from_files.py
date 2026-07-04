"""Tests for geneinsight.report.generate_rst_from_files."""

import pandas as pd
import pytest

from geneinsight.report.generate_rst_from_files import generate_rst_file


@pytest.fixture
def filtered_genesets_df():
    """A filtered_genesets_df that does NOT contain 'missing_subtitle'."""
    return pd.DataFrame({
        "Term": ["known_term"],
        "Odds Ratio": [1.5],
        "Adjusted P-value": [0.05],
        "Combined Score": [5.0],
        "P-value": [0.01],
    })


def _make_section(subtitle, with_subtitle_key=True):
    """Build a minimal section dict that generate_rst_file expects."""
    section = {
        "title": "Theme 1 - Test",
        "content": "Body content here.",
        "references": [],
        "thematic_geneset": [],
        "code_block": "{'GENE1': 'desc'}",
    }
    if with_subtitle_key:
        section["subtitle"] = subtitle
    return section


def test_generate_rst_file_raises_when_subtitle_not_in_filtered_df(
    tmp_path, filtered_genesets_df
):
    """When section has a subtitle not in filtered_genesets_df, must raise."""
    out_path = tmp_path / "cluster_0.rst"
    sections = [_make_section("missing_subtitle")]

    with pytest.raises(KeyError, match="No enriched.csv entry for subtitle"):
        generate_rst_file(
            filename=str(out_path),
            sections=sections,
            filtered_genesets_df=filtered_genesets_df,
            call_ncbi_api=False,
            taxonomy_id="9606",
        )


def test_generate_rst_file_raises_when_section_has_no_subtitle(
    tmp_path, filtered_genesets_df
):
    """When section dict lacks a 'subtitle' key entirely, must raise."""
    out_path = tmp_path / "cluster_0.rst"
    sections = [_make_section("ignored", with_subtitle_key=False)]

    with pytest.raises(KeyError, match="Section missing 'subtitle' key"):
        generate_rst_file(
            filename=str(out_path),
            sections=sections,
            filtered_genesets_df=filtered_genesets_df,
            call_ncbi_api=False,
            taxonomy_id="9606",
        )
