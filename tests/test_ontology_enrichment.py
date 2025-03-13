# Filename: test_ontology_enrichment.py

import os
import pytest
import tempfile
import pandas as pd
import torch
import logging
from unittest.mock import patch, MagicMock
from geneinsight.ontology.calculate_ontology_enrichment import (
    HypergeometricGSEA,
    OntologyReader,
    RAGModuleGSEAPY
)


@pytest.fixture
def mock_genelist():
    return ["GENE1", "GENE2", "GENE3"]


@pytest.fixture
def mock_background_list():
    return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]


@pytest.fixture
def mock_ontology_file_contents():
    """
    Returns text mimicking a tab-delimited ontology file with two terms and associated genes.
    """
    return """TermA\t\tGENE1\tGENE2
TermB\t\tGENE3
"""


@pytest.fixture
def mock_ontology_file(mock_ontology_file_contents):
    """
    Creates a temporary file with ontology data and returns the file path.
    """
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write(mock_ontology_file_contents)
        tmp_path = tmp.name
    yield tmp_path

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# ------------------------------------------------------------------
# Tests for OntologyReader
# ------------------------------------------------------------------
def test_ontology_reader_init_and_dict(mock_ontology_file):
    """
    Test initialization of OntologyReader and check if gene_dict is read properly.
    """
    ontology_reader = OntologyReader(mock_ontology_file, ontology_name="MockOntology")
    assert ontology_reader.file_path == mock_ontology_file
    assert ontology_reader.name == "MockOntology"

    # Check gene_dict
    gene_dict = ontology_reader.gene_dict
    assert "TermA" in gene_dict
    assert "TermB" in gene_dict
    assert gene_dict["TermA"] == ["GENE1", "GENE2"]
    assert gene_dict["TermB"] == ["GENE3"]


def test_ontology_reader_empty_file():
    """
    Test that OntologyReader can handle an empty file gracefully (returns empty dictionary).
    """
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        reader = OntologyReader(tmp_path, ontology_name="EmptyOntology")
        assert len(reader.gene_dict) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ontology_reader_file_not_found():
    """
    Test that reading a non-existent file raises an exception.
    """
    with pytest.raises(FileNotFoundError):
        OntologyReader("non_existent_file_path.tsv", "NonExistent")


# ------------------------------------------------------------------
# Tests for HypergeometricGSEA
# ------------------------------------------------------------------
@pytest.mark.parametrize("background_list", [None, ["GENE4", "GENE5"]])
def test_hypergeometric_gsea_init(mock_genelist, background_list):
    """
    Test the initialization of HypergeometricGSEA with and without background list.
    """
    gsea = HypergeometricGSEA(genelist=mock_genelist, background_list=background_list)
    assert gsea.genelist == mock_genelist
    assert gsea.background_list == background_list


@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_hypergeometric_gsea_perform(mock_enrich, mock_genelist, mock_background_list):
    """
    Test perform_hypergeometric_gsea method, ensuring it calls gseapy.enrich correctly
    and returns a DataFrame.
    """
    # Mock the return of gp.enrich
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
    """
    Test behavior when gseapy.enrich raises an exception.
    We'll check that query_gse_single_ontology returns an empty DataFrame.
    """
    from geneinsight.ontology.calculate_ontology_enrichment import RAGModuleGSEAPY

    # Force gseapy.enrich to raise
    mock_enrich.side_effect = RuntimeError("Mock error in enrich")

    # Create a minimal RAGModuleGSEAPY
    class MockOntologyReader:
        name = "MockOntology"
        gene_dict = {"TermSet1": ["GENE1", "GENE2"]}

    rag_module = RAGModuleGSEAPY([MockOntologyReader()])
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)

    # Attempt to enrich for the single ontology
    df_out = rag_module.query_gse_single_ontology(MockOntologyReader())
    # Expect an empty DataFrame
    assert df_out.empty


# ------------------------------------------------------------------
# Tests for RAGModuleGSEAPY
# ------------------------------------------------------------------
@pytest.fixture
def mock_ontologies(mock_ontology_file):
    """
    Creates a small list of OntologyReader objects.
    """
    ont1 = OntologyReader(mock_ontology_file, "Ontology1")
    ont2 = OntologyReader(mock_ontology_file, "Ontology2")
    return [ont1, ont2]


@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
def test_ragmodule_init(mock_transformer_cls, mock_ontologies):
    """
    Test initialization of RAGModuleGSEAPY to ensure it sets up ontologies 
    and instantiates SentenceTransformer.
    """
    mock_transformer_instance = MagicMock()
    mock_transformer_cls.return_value = mock_transformer_instance

    rag_module = RAGModuleGSEAPY(mock_ontologies)
    assert len(rag_module.ontologies) == 2
    mock_transformer_cls.assert_called_once()


@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_enrichment(mock_enrich, mock_transformer_cls, mock_ontologies, mock_genelist):
    """
    Test get_enrichment method. Mocks out gp.enrich to ensure the pipeline flows.
    """
    # Set up mocks
    mock_transformer_cls.return_value = MagicMock()
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.2]
    })
    mock_enrich.return_value = mock_result

    # Initialize RAG module
    rag_module = RAGModuleGSEAPY(mock_ontologies)

    # Create a HypergeometricGSEA manually to test inside get_enrichment
    rag_module.hypergeometric_gsea_obj = HypergeometricGSEA(mock_genelist)

    # Run get_enrichment
    res_df = rag_module.get_enrichment(fdr_threshold=0.1)
    # Expect only Term1 to remain after filtering (Adjusted P-value < 0.1)
    assert not res_df.empty
    assert list(res_df["Term"]) == ["Term1"]


@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents(
    mock_enrich, mock_transformer_cls, mock_util, mock_ontologies, mock_genelist
):
    """
    Test get_top_documents returns the top N matches and includes 'cosine_score' column.
    """
    # Setup mock embedder
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        # query embedding
        [0.5, 0.5, 0.5],
        # document embeddings for each term
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ]
    ]
    mock_transformer_cls.return_value = mock_embedder_instance

    # Setup mock for gp.enrich
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.01]  # Both under threshold
    })
    mock_enrich.return_value = mock_result

    # Return a real tensor here
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[0.8, 0.2]])

    # Initialize the RAG module
    rag_module = RAGModuleGSEAPY(mock_ontologies)

    # Execute get_top_documents
    (
        top_indices,
        extracted_items,
        enrichr_results,
        enrichr_df_filtered,
        formatted_output
    ) = rag_module.get_top_documents(
        query="Sample query",
        gene_list=mock_genelist,
        background_list=None,
        N=5,
        fdr_threshold=0.1
    )

    # Both terms pass the FDR threshold; top_indices should reflect the highest similarity (Term1).
    assert top_indices == [0, 1]  # Term1 is index 0, Term2 index 1
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
    """
    Test get_top_documents when N is larger than the number of enriched documents.
    """
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        [0.5, 0.5, 0.5],
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ]
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
    (
        top_indices,
        extracted_items,
        enrichr_results,
        enrichr_df_filtered,
        formatted_output
    ) = rag_module.get_top_documents(
        query="Sample query",
        gene_list=mock_genelist,
        background_list=None,
        N=10,  # N is bigger than number of docs (2)
        fdr_threshold=0.1
    )

    # We expect top_indices to have exactly 2 (since that's all the docs)
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
    """
    Test get_top_documents scenario where torch.topk raises a RuntimeError.
    Ensures we catch the exception and return empty top_results_indices and the message.
    """
    import torch
    # Setup mocks
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.side_effect = [
        [0.5, 0.5, 0.5],
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ]
    ]
    mock_transformer_cls.return_value = mock_embedder_instance

    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["Term1", "Term2"],
        "Adjusted P-value": [0.05, 0.01]
    })
    mock_enrich.return_value = mock_result

    # Return a real tensor
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[0.8, 0.2]])

    # Make torch.topk raise a RuntimeError
    mock_topk.side_effect = RuntimeError("Mock topk error")

    rag_module = RAGModuleGSEAPY(mock_ontologies)
    with caplog.at_level(logging.WARNING):
        (
            top_indices,
            extracted_items,
            enrichr_results,
            enrichr_df_filtered,
            formatted_output
        ) = rag_module.get_top_documents(
            query="Sample query",
            gene_list=mock_genelist,
            background_list=None,
            N=5,
            fdr_threshold=0.1
        )

    # Should return an empty list for top_indices
    assert top_indices == []
    # We also expect the error message to appear in 'extracted_items'
    assert "RuntimeError occurred in topk computation: Mock topk error" in extracted_items
    # Confirm the log message was written
    assert "RuntimeError occurred in topk computation: Mock topk error" in caplog.text


@patch("geneinsight.ontology.calculate_ontology_enrichment.util")
@patch("geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer")
@patch("geneinsight.ontology.calculate_ontology_enrichment.gp.enrich")
def test_ragmodule_get_top_documents_empty_genelist(
    mock_enrich, mock_transformer_cls, mock_util, mock_ontologies
):
    """
    Test get_top_documents when the gene_list is empty,
    which typically results in no significant terms.
    """
    import torch

    # Setup mocks
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.encode.return_value = torch.tensor([0.5, 0.5, 0.5])
    mock_transformer_cls.return_value = mock_embedder_instance

    # gp.enrich mock returns no results
    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": [],
        "Adjusted P-value": []
    })
    mock_enrich.return_value = mock_result
    mock_util.pytorch_cos_sim.return_value = torch.tensor([[]])

    rag_module = RAGModuleGSEAPY(mock_ontologies)
    (
        top_indices,
        extracted_items,
        enrichr_results,
        enrichr_df_filtered,
        formatted_output
    ) = rag_module.get_top_documents(
        query="Sample query",
        gene_list=[],  # empty gene list
        background_list=None,
        N=5,
        fdr_threshold=0.1
    )

    assert len(top_indices) == 0
    assert enrichr_results.empty
    assert enrichr_df_filtered.empty
    assert "No significant terms found" in extracted_items
    assert "No significant terms found" in formatted_output


def test_ragmodule_get_top_documents_empty_ontologies(mock_genelist):
    """
    Test behavior when RAGModuleGSEAPY is created with an empty ontologies list.
    get_enrichment should produce no results.
    """
    rag_module = RAGModuleGSEAPY([])
    (
        top_indices,
        extracted_items,
        enrichr_results,
        enrichr_df_filtered,
        formatted_output
    ) = rag_module.get_top_documents(
        query="Any query",
        gene_list=mock_genelist,
        background_list=None,
        N=5,
        fdr_threshold=0.1
    )

    assert len(top_indices) == 0
    assert "No significant terms found" in extracted_items
    assert enrichr_results.empty
    assert enrichr_df_filtered.empty


def test_ragmodule_format_top_documents(mock_genelist):
    """
    Test that format_top_documents returns expected formatted output.
    """
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame({
        "Term": ["TermA", "TermB"],
        "Adjusted P-value": [0.05, 0.01],
        "Gene_set": ["Ontology1", "Ontology2"]
    })

    top_results_indices = [1, 0]  # B first, then A
    formatted_str, filtered_df = rag_module.format_top_documents(top_results_indices)
    # Expect 2 lines
    assert "* `Ontology2: TermB - FDR: 0.0100`" in formatted_str
    assert "* `Ontology1: TermA - FDR: 0.0500`" in formatted_str
    assert filtered_df.shape[0] == 2
    assert filtered_df.iloc[0]["Term"] == "TermB"
    assert filtered_df.iloc[1]["Term"] == "TermA"


def test_ragmodule_format_top_documents_with_empty_indices():
    """
    Test that format_top_documents returns empty results if top_results_indices is empty.
    """
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    # Suppose we have some enrichment results but the top indices are empty.
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
    Test that format_top_documents logs a warning and returns empty results 
    if we pass invalid index positions (e.g., out of range or negative).
    """
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame({
        "Term": ["TermA", "TermB"],
        "Adjusted P-value": [0.05, 0.01],
        "Gene_set": ["Ontology1", "Ontology2"]
    })

    with caplog.at_level(logging.WARNING):
        formatted_str, filtered_df = rag_module.format_top_documents(invalid_indices)

    # We should see a warning about out-of-bounds
    assert "IndexError: Attempted to index out-of-bounds" in caplog.text
    # Both return values should be empty
    assert formatted_str == ""
    assert filtered_df.empty


def test_ragmodule_format_top_documents_no_enrichr_results():
    """
    Test that format_top_documents returns empty results if self.enrichr_results is empty,
    even if indices are requested.
    """
    rag_module = RAGModuleGSEAPY(ontology_object_list=[])
    rag_module.enrichr_results = pd.DataFrame()  # empty

    # Even if we specify some indices, there's nothing to index
    formatted_str, filtered_df = rag_module.format_top_documents([0, 1])
    assert formatted_str == ""
    assert filtered_df.empty
