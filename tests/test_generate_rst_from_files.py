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


import numpy as np


def _filtered_df_with_empirical(term="known_term"):
    """Return a filtered_genesets_df that includes p_empirical / q_empirical."""
    return pd.DataFrame({
        "Term": [term],
        "Odds Ratio": [523.72],
        "Adjusted P-value": [0.005],
        "Combined Score": [2437.93],
        "P-value": [0.001],
        "p_empirical": [0.0031],
        "q_empirical": [0.012],
    })


def _filtered_df_without_empirical(term="known_term"):
    """Return a filtered_genesets_df WITHOUT the empirical columns
    (models the state when --empirical-p was not used)."""
    return pd.DataFrame({
        "Term": [term],
        "Odds Ratio": [523.72],
        "Adjusted P-value": [0.005],
        "Combined Score": [2437.93],
        "P-value": [0.001],
    })


def _minimal_section_with_subtitle():
    return {
        "title": "Theme 1 - Test",
        "subtitle": "known_term",
        "content": "Body.",
        "references": [],
        "thematic_geneset": [],
        "code_block": "{'GENE1': 'desc'}",
    }


def test_generate_rst_file_emits_empirical_lines_when_columns_present(tmp_path):
    """When filtered_genesets_df has p_empirical + q_empirical columns and
    the row's values are not NaN, the RST output must include two extra
    :abbr: lines (Empirical p, Empirical q) between FDR and Combined Score,
    and the FDR label must be renamed to 'Parametric FDR'."""
    out_path = tmp_path / "cluster_0.rst"
    generate_rst_file(
        filename=str(out_path),
        sections=[_minimal_section_with_subtitle()],
        filtered_genesets_df=_filtered_df_with_empirical(),
        call_ncbi_api=False,
        taxonomy_id="9606",
    )
    content = out_path.read_text()

    assert ":abbr:`Empirical p " in content, "Missing Empirical p line"
    assert ":abbr:`Empirical q " in content, "Missing Empirical q line"
    assert "0.003" in content, "Empirical p value 0.0031 not formatted"
    assert "0.012" in content, "Empirical q value 0.012 not formatted"
    assert ":abbr:`Parametric FDR " in content, "FDR label not renamed"
    assert ":abbr:`FDR (False Discovery Rate " not in content, (
        "Old FDR tooltip text still present; label should be renamed when "
        "empirical columns are present"
    )

    # Ordering check: empirical lines sit between the FDR line and the
    # Combined Score line, matching the layout spec.
    fdr_idx = content.index(":abbr:`Parametric FDR")
    emp_p_idx = content.index(":abbr:`Empirical p")
    emp_q_idx = content.index(":abbr:`Empirical q")
    cs_idx = content.index(":abbr:`Combined Score")
    assert fdr_idx < emp_p_idx < emp_q_idx < cs_idx


def test_generate_rst_file_omits_empirical_when_columns_absent(tmp_path):
    """When filtered_genesets_df does NOT have the empirical columns
    (i.e. --empirical-p was not used), the RST output must be identical
    to today's format: three abbr lines (Odds Ratio, FDR, Combined Score),
    no Empirical lines, and the FDR label unchanged."""
    out_path = tmp_path / "cluster_0.rst"
    generate_rst_file(
        filename=str(out_path),
        sections=[_minimal_section_with_subtitle()],
        filtered_genesets_df=_filtered_df_without_empirical(),
        call_ncbi_api=False,
        taxonomy_id="9606",
    )
    content = out_path.read_text()

    assert ":abbr:`Empirical" not in content, (
        "Empirical lines should not be emitted when p_empirical column is absent"
    )
    assert ":abbr:`FDR (False Discovery Rate " in content, (
        "FDR label should be unchanged when empirical columns are absent"
    )
    assert ":abbr:`Parametric FDR" not in content


def test_generate_rst_file_omits_empirical_when_row_value_is_nan(tmp_path):
    """When the column exists but the specific row has NaN empirical values
    (e.g. query gene count outside the 1-500 bin range), the row's rendering
    should skip the empirical lines rather than write 'nan'."""
    df = _filtered_df_with_empirical()
    df.loc[0, "p_empirical"] = np.nan
    df.loc[0, "q_empirical"] = np.nan

    out_path = tmp_path / "cluster_0.rst"
    generate_rst_file(
        filename=str(out_path),
        sections=[_minimal_section_with_subtitle()],
        filtered_genesets_df=df,
        call_ncbi_api=False,
        taxonomy_id="9606",
    )
    content = out_path.read_text()

    assert ":abbr:`Empirical" not in content
    assert "nan" not in content.lower()


def test_generate_rst_file_fdr_label_has_no_parens_regression(tmp_path):
    """Regression: 'FDR (parametric)' as an :abbr: label was misparsed by
    Sphinx because parens in the label conflict with the tooltip delimiter,
    producing 'FDR' as the visible label and a garbled tooltip. The label
    must be paren-free so 'first paren opens tooltip' is unambiguous."""
    out_path = tmp_path / "cluster_0.rst"
    generate_rst_file(
        filename=str(out_path),
        sections=[_minimal_section_with_subtitle()],
        filtered_genesets_df=_filtered_df_with_empirical(),
        call_ncbi_api=False,
        taxonomy_id="9606",
    )
    content = out_path.read_text()

    # The label chosen must not contain parens (would collide with :abbr:'s
    # tooltip delimiter). Concrete asserts on the current-choice label:
    assert ":abbr:`Parametric FDR (" in content
    # And guard against the broken form:
    assert ":abbr:`FDR (parametric)" not in content
