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
