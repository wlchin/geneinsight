# Filename: test_ontology_enrichment.py

import os
import sys
import pytest
import tempfile
import pandas as pd
import torch
import logging
from unittest.mock import patch, MagicMock
from geneinsight.ontology.calculate_ontology_enrichment import (
    HypergeometricGSEA,
    OntologyReader,
    RAGModuleGSEAPY,
    main  # Import main() for testing the parser code
)

# ----------------------------
# Fixtures and OntologyReader tests
# ----------------------------
@pytest.fixture
def mock_genelist():
    return ["GENE1", "GENE2", "GENE3"]

@pytest.fixture
def mock_background_list():
    return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

@pytest.fixture
def mock_ontology_file_contents():
    return """TermA\t\tGENE1\tGENE2
TermB\t\tGENE3
"""

@pytest.fixture
def mock_ontology_file(mock_ontology_file_contents):
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write(mock_ontology_file_contents)
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def test_ontology_reader_init_and_dict(mock_ontology_file):
    ontology_reader = OntologyReader(mock_ontology_file, ontology_name="MockOntology")
    assert ontology_reader.file_path == mock_ontology_file
    assert ontology_reader.name == "MockOntology"
    gene_dict = ontology_reader.gene_dict
    assert "TermA" in gene_dict
    assert "TermB" in gene_dict
    assert gene_dict["TermA"] == ["GENE1", "GENE2"]
    assert gene_dict["TermB"] == ["GENE3"]

def test_ontology_reader_empty_file():
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        reader = OntologyReader(tmp_path, ontology_name="EmptyOntology")
        assert len(reader.gene_dict) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_ontology_reader_file_not_found():
    with pytest.raises(FileNotFoundError):
        OntologyReader("non_existent_file_path.tsv", "NonExistent")

def test_ontology_reader_inconsistent_format():
    contents = "TermA\tGENE1\nTermB\t\tGENE2\tGENE3"
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        reader = OntologyReader(tmp_path, ontology_name="Inconsistent")
        # With fallback parsing, we expect:
        # "TermA" -> ["GENE1"]
        # "TermB" -> ["GENE2", "GENE3"]
        assert "TermA" in reader.gene_dict
        assert reader.gene_dict["TermA"] == ["GENE1"]
        assert "TermB" in reader.gene_dict
        assert reader.gene_dict["TermB"] == ["GENE2", "GENE3"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ----------------------------
# Tests for HypergeometricGSEA
# ----------------------------
@pytest.mark.parametrize("background_list", [None, ["GENE4", "GENE5"]])
def test_hypergeometric_gsea_init(mock_genelist, background_list):
    gsea = HypergeometricGSEA(genelist=mock_genelist, background_list=background_list)
    assert gsea.genelist == mock_genelist
    assert gsea.background_list == background_list

@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_hypergeometric_gsea_perform(mock_enrich, mock_genelist, mock_background_list):
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["MockTerm"],
        "Adjusted P-value": [0.05],
        "Overlap": ["2/100"]
    })
    mock_enrich.return_value = mock_result
    gsea = HypergeometricGSEA(mock_genelist, mock_background_list)
    dummy_ontology_dict = {"TermSet1": ["GENE1", "GENE2"]}
    result_df = gsea.perform_hypergeometric_gsea(dummy_ontology_dict)
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["Term", "Adjusted P-value", "Overlap"]
    mock_enrich.assert_called_once_with(
        gene_list=mock_genelist,
        gene_sets=dummy_ontology_dict,
        background=mock_background_list,
        outdir=None,
        verbose=True
    )

@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_hypergeometric_gsea_perform_raises_exception(mock_enrich, mock_genelist):
    from geneinsight.ontology.calculate_ontology_enrichment import RAGModuleGSEAPY
    mock_enrich.side_effect = RuntimeError("Mock error in enrich")
    class MockOntologyReader:
        name = "MockOntology"
        gene_dict = {"TermSet1": ["GENE1", "GENE2"]}
    rag_module = RAGModuleGSEAPY([MockOntologyReader()])
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)
    df_out = rag_module.query_gse_single_ontology(MockOntologyReader())
    assert df_out.empty

# ----------------------------
# Tests for RAGModuleGSEAPY
# ----------------------------
@pytest.fixture
def mock_ontologies(mock_ontology_file):
    ont1 = OntologyReader(mock_ontology_file, "Ontology1")
    ont2 = OntologyReader(mock_ontology_file, "Ontology2")
    return [ont1, ont2]

@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
def test_ragmodule_init(mock_transformer_cls, mock_ontologies):
    mock_transformer_instance = MagicMock()
    mock_transformer_cls.return_value = mock_transformer_instance
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    assert len(rag_module.ontologies) == 2
    mock_transformer_cls.assert_called_once()

@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_enrichment(mock_enrich, mock_transformer_cls, mock_ontologies, mock_genelist):
    mock_transformer_cls.return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.2]
    })
    mock_enrich.return_value = mock_result
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)
    res_df = rag_module.get_enrichment(fdr_threshold=0.1)
    assert not res_df.empty
    assert list(res_df["Term"]) == ["Term1"]

@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
def test_ragmodule_get_enrichment_no_results_pre_filter(mock_transformer_cls, mock_ontologies, mock_genelist, caplog):
    mock_transformer_cls.return_value = MagicMock()
    class MockEmptyOntologyReader:
        name = "EmptyOntology"
        gene_dict = {}
    rag_module = RAGModuleGSEAPY([MockEmptyOntologyReader()])
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)
    with caplog.at_level(logging.INFO):
        res_df = rag_module.get_enrichment(fdr_threshold=0.1)
    assert res_df.empty
    assert "No results found for any ontology before FDR filtering." in caplog.text

@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
def test_ragmodule_get_enrichment_all_filtered_out(
    mock_transformer_cls, mock_enrich, mock_ontologies, mock_genelist, caplog
):
    mock_transformer_cls.return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.2, 0.3]
    })
    mock_enrich.return_value = mock_result
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)
    with caplog.at_level(logging.INFO):
        res_df = rag_module.get_enrichment(fdr_threshold=0.1)
    assert res_df.empty
    assert "All results filtered out by FDR threshold. No significant terms remain." in caplog.text

def test_ragmodule_query_gse_single_ontology_empty_dict(mock_genelist):
    class MockEmptyOntologyReader:
        name = "EmptyTest"
        gene_dict = {}
    rag_module = RAGModuleGSEAPY([MockEmptyOntologyReader()])
    rag_module.hypergeometric_gsea_obj = MagicMock()
    rag_module.hypergeometric_gsea_obj.perform_hypergeometric_gsea.return_value = pd.DataFrame()
    df_out = rag_module.query_gse_single_ontology(MockEmptyOntologyReader())
    assert df_out.empty
    rag_module.hypergeometric_gsea_obj.perform_hypergeometric_gsea.assert_called_once()

@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents(
    mock_enrich, mock_transformer_cls, mock_util, mock_ontologies, mock_genelist
):
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        [0.5, 0.5, 0.5],
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    ]
    mock_transformer_cls.return_value = mock_embedder_instance
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.01]
    })
    mock_enrich.return_value = mock_result
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[0.8, 0.2]])
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    (top_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output) = rag_module.get_top_documents(
        query="Sample query", gene_list=mock_genelist, background_list=None, N=5, fdr_threshold=0.1
    )
    assert top_indices == [0, 1]
    assert "Term1" in extracted_items
    assert "Term2" in extracted_items
    assert not enrichr_results.empty
    assert "cosine_score" in enrichr_results.columns
    assert not enrichr_df_filtered.empty
    assert "Term1" in formatted_output
    assert "Term2" in formatted_output

@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents_n_larger_than_docs(
    mock_enrich, mock_transformer_cls, mock_util, mock_ontologies, mock_genelist
):
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        [0.5, 0.5, 0.5],
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    ]
    mock_transformer_cls.return_value = mock_embedder_instance
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.01, 0.02]
    })
    mock_enrich.return_value = mock_result
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[0.8, 0.2]])
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    (top_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output) = rag_module.get_top_documents(
        query="Sample query", gene_list=mock_genelist, background_list=None, N=10, fdr_threshold=0.1
    )
    assert top_indices == [0, 1]
    assert len(enrichr_results) == 2
    assert "Term1" in extracted_items and "Term2" in extracted_items

@patch("geneinsight.ontology.calculate_ontology_enrichment.torch.topk")
@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents_raise_runtime_error_in_topk(
    mock_enrich, mock_transformer_cls, mock_util, mock_topk, mock_ontologies, mock_genelist, caplog
):
    import torch
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        [0.5, 0.5, 0.5],
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    ]
    mock_transformer_cls.return_value = mock_embedder_instance
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.01]
    })
    mock_enrich.return_value = mock_result
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[0.8, 0.2]])
    # Force torch.topk to raise RuntimeError
    mock_topk.side_effect = RuntimeError("Mock topk error")
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    with caplog.at_level(logging.WARNING):
        (top_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output) = rag_module.get_top_documents(
            query="Sample query", gene_list=mock_genelist, background_list=None, N=5, fdr_threshold=0.1
        )
    assert top_indices == []
    assert "RuntimeError occurred in topk computation: Mock topk error" in extracted_items
    assert "RuntimeError occurred in topk computation: Mock topk error" in caplog.text

@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents_empty_genelist(
    mock_enrich, mock_transformer_cls, mock_util, mock_ontologies
):
    import torch
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.return_value = torch.tensor([0.5, 0.5, 0.5])
    mock_transformer_cls.return_value = mock_embedder_instance
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": [],
        "Adjusted P-value": []
    })
    mock_enrich.return_value = mock_result
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[]])
    rag_module = RAGModuleGSEAPY(mock_ontologies)
    (top_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output) = rag_module.get_top_documents(
        query="Sample query", gene_list=[], background_list=None, N=5, fdr_threshold=0.1
    )
    assert len(top_indices) == 0
    assert enrichr_results.empty
    assert enrichr_df_filtered.empty
    assert "No significant terms found" in extracted_items
    assert "No significant terms found" in formatted_output

def test_ragmodule_get_top_documents_empty_ontologies(mock_genelist):
    rag_module = RAGModuleGSEAPY([])
    (top_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output) = rag_module.get_top_documents(
        query="Any query", gene_list=mock_genelist, background_list=None, N=5, fdr_threshold=0.1
    )
    assert len(top_indices) == 0
    assert "No significant terms found" in extracted_items
    assert enrichr_results.empty
    assert enrichr_df_filtered.empty

def test_ragmodule_format_top_documents(mock_genelist):
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame({
        "Term": ["TermA", "TermB"],
        "Adjusted P-value": [0.05, 0.01],
        "Gene_set": ["Ontology1", "Ontology2"]
    })
    top_results_indices = [1, 0]
    formatted_str, filtered_df = rag_module.format_top_documents(top_results_indices)
    assert "* `Ontology2: TermB - FDR: 0.0100`" in formatted_str
    assert "* `Ontology1: TermA - FDR: 0.0500`" in formatted_str
    assert filtered_df.shape[0] == 2
    assert filtered_df.iloc[0]["Term"] == "TermB"
    assert filtered_df.iloc[1]["Term"] == "TermA"

def test_ragmodule_format_top_documents_with_empty_indices():
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame({
        "Term": ["TermA", "TermB"],
        "Adjusted P-value": [0.05, 0.01],
        "Gene_set": ["Ontology1", "Ontology2"]
    })
    formatted_str, filtered_df = rag_module.format_top_documents([])
    assert formatted_str == ""
    assert filtered_df.empty

@pytest.mark.parametrize("invalid_indices", [[0, 999], [999], [-1]])
def test_ragmodule_format_top_documents_out_of_bounds(invalid_indices, caplog):
    """
    Test that format_top_documents logs appropriate warnings and returns empty results
    for negative or out-of-range indices.
    """
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame({
        "Term": ["TermA", "TermB"],
        "Adjusted P-value": [0.05, 0.01],
        "Gene_set": ["Ontology1", "Ontology2"]
    })
    with caplog.at_level(logging.WARNING):
        formatted_str, filtered_df = rag_module.format_top_documents(invalid_indices)
    
    # Check for either error message
    assert (
        "IndexError: Attempted to index out-of-bounds" in caplog.text or
        "IndexError: Negative indices are disallowed" in caplog.text
    )
    
    # Also verify the function returns empty results
    assert formatted_str == ""
    assert filtered_df.empty

# ----------------------------
# Tests for main() parser code
# ----------------------------
def test_main_parser_missing_args(monkeypatch, capsys):
    test_args = ["prog"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit) as e:
        main()
    assert e.type == SystemExit
    assert e.value.code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err.lower()

@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.OntologyReader")
@patch("geneinsight.ontology.calculate_ontology_enrichment.os.listdir")
@patch("geneinsight.ontology.calculate_ontology_enrichment.pd.read_csv")
@patch("geneinsight.ontology.calculate_ontology_enrichment.sys.exit")
def test_main_parser_success(
    mock_sys_exit,
    mock_read_csv,
    mock_listdir,
    mock_ontology_reader,
    mock_sentence_transformer,
    monkeypatch,
    tmp_path
):
    # Mock SentenceTransformer to prevent external calls
    mock_sentence_transformer_instance = MagicMock()
    mock_sentence_transformer.return_value = mock_sentence_transformer_instance
    # Create a fake ontology directory and add a dummy file so isfile() returns True.
    fake_ontology_dir = tmp_path / "ontology_dir"
    fake_ontology_dir.mkdir()
    dummy_file = fake_ontology_dir / "fake_ontology.tsv"
    dummy_file.write_text("Dummy content")
    
    mock_listdir.return_value = ["fake_ontology.tsv"]
    
    def mock_read_csv_side_effect(filepath, *args, **kwargs):
        if "filter.csv" in str(filepath):
            return pd.DataFrame({"Term": ["mockTerm"]})
        if "summary.csv" in str(filepath):
            return pd.DataFrame({
                "query": ["mockTerm"],
                "unique_genes": ['{"GENE1":1}']
            })
        return pd.DataFrame()
    
    mock_read_csv.side_effect = mock_read_csv_side_effect
    mock_ontology_instance = MagicMock()
    mock_ontology_instance.gene_dict = {"dummy": ["GENE1"]}
    mock_ontology_reader.return_value = mock_ontology_instance
    
    test_args = [
        "prog",
        "--summary_csv", "summary.csv",
        "--gene_origin", "genes.tsv",
        "--background_genes", "background.tsv",
        "--output_csv", "output.csv",
        "--ontology_folder", str(fake_ontology_dir),
        "--filter_csv", "filter.csv",
        "--fdr_threshold", "0.05"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    mock_sys_exit.side_effect = SystemExit("Called sys.exit unexpectedly!")
    
    try:
        main()
    except SystemExit as e:
        pytest.fail(f"main() exited unexpectedly with code {e.code} - {e}")
    
    assert mock_sys_exit.call_count == 0
    mock_listdir.assert_called_once_with(str(fake_ontology_dir))
    assert mock_read_csv.call_count >= 2
