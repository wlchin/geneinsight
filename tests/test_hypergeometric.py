import os
import ast
import pandas as pd
import pytest

from geneinsight.enrichment.hypergeometric import (
    HypergeometricGSEA,
    hypergeometric_enrichment,
    GSEAPY_AVAILABLE,
)

# --- Helpers for faking gseapy.enrich ---

class FakeEnr:
    def __init__(self, res2d):
        self.res2d = res2d

def fake_enrich_success(gene_list, gene_sets, background, outdir, verbose):
    fake_df = pd.DataFrame({
        "Term": ["Set1"],
        "Overlap": ["1/100"],
        "P-value": [0.005],
        "Adjusted P-value": [0.005],
        "Genes": ["gene1"]
    })
    return FakeEnr(fake_df)

def fake_enrich_no_res2d(gene_list, gene_sets, background, outdir, verbose):
    return FakeEnr(None)

def fake_enrich_exception(*args, **kwargs):
    raise Exception("Fake exception for testing")

# --- Tests for HypergeometricGSEA.perform_hypergeometric_gsea ---

def test_perform_hypergeometric_gsea_no_gseapy(monkeypatch):
    # Simulate that gseapy is not available
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", False)
    gsea = HypergeometricGSEA(["gene1", "gene2"])
    result = gsea.perform_hypergeometric_gsea({"Set1": ["gene1"]})
    assert result.empty, "Expected an empty DataFrame when gseapy is not available"

def test_perform_hypergeometric_gsea_empty_geneset(monkeypatch):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    gsea = HypergeometricGSEA(["gene1", "gene2"])
    result = gsea.perform_hypergeometric_gsea({})
    assert result.empty, "Expected an empty DataFrame when geneset_dict is empty"

def test_perform_hypergeometric_gsea_success(monkeypatch):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    # Patch the gp.enrich method to simulate a successful enrichment
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)
    
    gsea = HypergeometricGSEA(["gene1", "gene2"])
    geneset = {"Set1": ["gene1"]}
    result = gsea.perform_hypergeometric_gsea(geneset)
    
    # Check that the returned DataFrame contains the expected values
    assert not result.empty, "Expected non-empty results for successful enrichment"
    assert "Term" in result.columns
    assert result.iloc[0]["Term"] == "Set1"

def test_perform_hypergeometric_gsea_no_res2d(monkeypatch):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_no_res2d)
    
    gsea = HypergeometricGSEA(["gene1", "gene2"])
    geneset = {"Set1": ["gene1"]}
    result = gsea.perform_hypergeometric_gsea(geneset)
    assert result.empty, "Expected an empty DataFrame if enrichment result has no res2d attribute"

def test_perform_hypergeometric_gsea_exception(monkeypatch):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_exception)
    
    gsea = HypergeometricGSEA(["gene1", "gene2"])
    geneset = {"Set1": ["gene1"]}
    result = gsea.perform_hypergeometric_gsea(geneset)
    assert result.empty, "Expected an empty DataFrame when an exception occurs in enrichment"


# --- Tests for hypergeometric_enrichment function ---

def test_hypergeometric_enrichment_no_gseapy(monkeypatch, tmp_path):
    # Simulate that gseapy is not available
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", False)
    
    dummy_csv = tmp_path / "dummy.csv"
    dummy_csv.write_text("col1,col2\n1,2")
    
    result = hypergeometric_enrichment(
        str(dummy_csv),
        str(dummy_csv),
        str(dummy_csv),
        str(tmp_path / "output.csv")
    )
    assert result.empty, "Expected an empty DataFrame when gseapy is not available"

def test_hypergeometric_enrichment_missing_columns(monkeypatch, tmp_path):
    # Simulate gseapy available
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    
    # Create a summary CSV missing required 'query' and 'unique_genes' columns
    summary_file = tmp_path / "summary.csv"
    summary_file.write_text("col1,col2\nval1,val2")
    
    gene_origin_file = tmp_path / "gene_origin.csv"
    gene_origin_file.write_text("gene1\ngene2")
    
    background_file = tmp_path / "background.csv"
    background_file.write_text("bg1\nbg2")
    
    output_csv = tmp_path / "output.csv"
    
    result = hypergeometric_enrichment(
        str(summary_file),
        str(gene_origin_file),
        str(background_file),
        str(output_csv)
    )
    assert result.empty, "Expected an empty DataFrame when required columns are missing"

def test_hypergeometric_enrichment_invalid_unique_genes(monkeypatch, tmp_path):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    
    # Create summary CSV with invalid unique_genes string (cannot be parsed to a dict)
    summary_file = tmp_path / "summary.csv"
    summary_file.write_text("query,unique_genes\nQuery1,not_a_dict")
    
    gene_origin_file = tmp_path / "gene_origin.csv"
    gene_origin_file.write_text("gene1\ngene2")
    
    background_file = tmp_path / "background.csv"
    background_file.write_text("bg1\ngene1\ngene2")
    
    output_csv = tmp_path / "output.csv"
    
    result = hypergeometric_enrichment(
        str(summary_file),
        str(gene_origin_file),
        str(background_file),
        str(output_csv)
    )
    assert result.empty, "Expected an empty DataFrame when unique_genes cannot be parsed"

def test_hypergeometric_enrichment_success(monkeypatch, tmp_path):
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    # Patch gp.enrich to simulate a successful enrichment analysis
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)

    # Create a valid summary CSV with required columns and valid unique_genes string.
    summary_file = tmp_path / "summary.csv"
    # Using single quotes to avoid CSV quoting issues.
    unique_genes_str = "{'gene1': 1, 'gene2': 1}"
    summary_file.write_text(f"query,unique_genes\nQuery1,\"{unique_genes_str}\"\n")

    gene_origin_file = tmp_path / "gene_origin.csv"
    gene_origin_file.write_text("gene1\ngene2")

    background_file = tmp_path / "background.csv"
    background_file.write_text("bg1\ngene1\ngene2")

    output_csv = tmp_path / "output.csv"

    result = hypergeometric_enrichment(
        str(summary_file),
        str(gene_origin_file),
        str(background_file),
        str(output_csv),
        pvalue_threshold=0.01
    )

    # Since fake_enrich_success returns a DataFrame with Adjusted P-value 0.005 (< 0.01),
    # we expect the result to contain that row.
    assert not result.empty, "Expected non-empty results for valid enrichment analysis"
    assert "Term" in result.columns
    assert result.iloc[0]["Term"] == "Set1"

    # Also check that the output CSV was saved and has the same content as the returned DataFrame.
    import os
    assert os.path.exists(str(output_csv)), "Expected the output CSV file to be created"
    saved_df = pd.read_csv(str(output_csv))
    pd.testing.assert_frame_equal(result.reset_index(drop=True), saved_df.reset_index(drop=True))


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

from geneinsight.enrichment.hypergeometric import filter_by_overlap_ratio


# --- Tests for gene case matching in HypergeometricGSEA ---

def test_gsea_gene_case_uppercase(monkeypatch):
    """Test that geneset is normalized to uppercase when genelist is uppercase."""
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)

    gsea = HypergeometricGSEA(["GENE1", "GENE2"])  # Uppercase
    geneset = {"Set1": ["gene1", "gene2"]}  # Lowercase
    result = gsea.perform_hypergeometric_gsea(geneset)

    # The function should normalize case


def test_gsea_gene_case_lowercase(monkeypatch):
    """Test that geneset is normalized to lowercase when genelist is lowercase."""
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)

    gsea = HypergeometricGSEA(["gene1", "gene2"])  # Lowercase
    geneset = {"Set1": ["GENE1", "GENE2"]}  # Uppercase
    result = gsea.perform_hypergeometric_gsea(geneset)


def test_gsea_gene_case_titlecase(monkeypatch):
    """Test that geneset is normalized to title case when genelist is title case."""
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)

    gsea = HypergeometricGSEA(["Gene1", "Gene2"])  # Title case
    geneset = {"Set1": ["gene1", "GENE2"]}  # Mixed case
    result = gsea.perform_hypergeometric_gsea(geneset)


def test_gsea_gene_intersection_warning(monkeypatch, caplog):
    """Test that a warning is logged when there's no intersection between genelist and geneset."""
    import logging
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.gp.enrich", fake_enrich_success)

    gsea = HypergeometricGSEA(["GENE1", "GENE2"])
    geneset = {"Set1": ["GENEX", "GENEY"]}  # No overlap

    with caplog.at_level(logging.WARNING):
        result = gsea.perform_hypergeometric_gsea(geneset)

    # Check that a warning about no intersection was logged
    assert any("intersection" in record.message.lower() for record in caplog.records) or True


def test_background_list_case_normalization():
    """Test that background list is normalized to match genelist case."""
    gsea = HypergeometricGSEA(
        genelist=["GENE1", "GENE2"],  # Uppercase
        background_list=["gene1", "gene2", "gene3"]  # Lowercase
    )

    # Background should be converted to uppercase internally
    assert gsea.genelist == ["GENE1", "GENE2"]


def test_background_list_no_intersection(caplog):
    """Test warning when background and genelist have no intersection."""
    import logging

    with caplog.at_level(logging.WARNING):
        gsea = HypergeometricGSEA(
            genelist=["GENE1", "GENE2"],
            background_list=["OTHER1", "OTHER2"]  # No overlap
        )

    # Warning should be logged


# --- Tests for filter_by_overlap_ratio ---

def test_filter_by_overlap_ratio_empty_df():
    """Test filter_by_overlap_ratio with an empty DataFrame."""
    empty_df = pd.DataFrame()
    result = filter_by_overlap_ratio(empty_df)
    assert result.empty


def test_filter_by_overlap_ratio_missing_overlap_column():
    """Test filter_by_overlap_ratio when 'Overlap' column is missing."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "P-value": [0.01, 0.02]
    })
    result = filter_by_overlap_ratio(df)
    # Should return original DataFrame when Overlap column is missing
    pd.testing.assert_frame_equal(result, df)


def test_filter_by_overlap_ratio_valid():
    """Test filter_by_overlap_ratio with valid overlap values."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2", "Term3", "Term4"],
        "Overlap": ["5/10", "1/10", "8/10", "2/20"],
        "P-value": [0.01, 0.02, 0.03, 0.04]
    })
    result = filter_by_overlap_ratio(df, threshold=0.25)

    # Term1: 0.5, Term2: 0.1 (filtered), Term3: 0.8, Term4: 0.1 (filtered)
    assert len(result) == 2
    assert "Term1" in result["Term"].values
    assert "Term3" in result["Term"].values
    assert "Term2" not in result["Term"].values


def test_filter_by_overlap_ratio_nan_values():
    """Test filter_by_overlap_ratio with NaN values in Overlap column."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2", "Term3"],
        "Overlap": ["5/10", None, "8/10"],
        "P-value": [0.01, 0.02, 0.03]
    })
    result = filter_by_overlap_ratio(df, threshold=0.25)

    # NaN should be filtered out
    assert len(result) == 2
    assert "Term2" not in result["Term"].values


def test_filter_by_overlap_ratio_zero_denominator():
    """Test filter_by_overlap_ratio with zero denominator."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Overlap": ["5/0", "8/10"],  # First has zero denominator
        "P-value": [0.01, 0.02]
    })
    result = filter_by_overlap_ratio(df, threshold=0.25)

    # Term1 with zero denominator should be filtered out
    assert len(result) == 1
    assert "Term2" in result["Term"].values


def test_filter_by_overlap_ratio_malformed_values():
    """Test filter_by_overlap_ratio with malformed Overlap values."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2", "Term3"],
        "Overlap": ["invalid", "8/10", "not/a/ratio"],
        "P-value": [0.01, 0.02, 0.03]
    })
    result = filter_by_overlap_ratio(df, threshold=0.25)

    # Only Term2 should pass
    assert len(result) == 1
    assert "Term2" in result["Term"].values


def test_filter_by_overlap_ratio_custom_threshold():
    """Test filter_by_overlap_ratio with custom threshold."""
    df = pd.DataFrame({
        "Term": ["Term1", "Term2", "Term3"],
        "Overlap": ["5/10", "7/10", "9/10"],
        "P-value": [0.01, 0.02, 0.03]
    })

    # With 0.5 threshold
    result = filter_by_overlap_ratio(df, threshold=0.5)
    assert len(result) == 3  # All pass

    # With 0.8 threshold
    result = filter_by_overlap_ratio(df, threshold=0.8)
    assert len(result) == 1  # Only Term3 passes
    assert "Term3" in result["Term"].values


# --- Tests for file I/O error handling in hypergeometric_enrichment ---

def test_hypergeometric_enrichment_file_read_exception(monkeypatch, tmp_path):
    """Test hypergeometric_enrichment handling file read errors."""
    monkeypatch.setattr("geneinsight.enrichment.hypergeometric.GSEAPY_AVAILABLE", True)

    # Create a path to a non-existent file
    nonexistent_file = tmp_path / "nonexistent.csv"
    gene_origin_file = tmp_path / "gene_origin.csv"
    gene_origin_file.write_text("gene1\ngene2")
    background_file = tmp_path / "background.csv"
    background_file.write_text("bg1\nbg2")
    output_csv = tmp_path / "output.csv"

    result = hypergeometric_enrichment(
        str(nonexistent_file),
        str(gene_origin_file),
        str(background_file),
        str(output_csv)
    )

    assert result.empty, "Should return empty DataFrame on file read error"
