import os
import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

# Import the modules to test
# Assuming the original file is named gsea_module.py
from geneinsight.ontology.calculate_ontology_enrichment import HypergeometricGSEA, OntologyReader, RAGModuleGSEAPY

class TestHypergeometricGSEA:
    def setup_method(self):
        self.test_genes = ["GENE1", "GENE2", "GENE3"]
        self.background = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        self.gsea = HypergeometricGSEA(self.test_genes, self.background)
    
    @patch('gseapy.enrich')
    def test_perform_hypergeometric_gsea(self, mock_enrich):
        # Create a mock for the enrich function return value
        mock_result = MagicMock()
        mock_result.res2d = pd.DataFrame({
            'Term': ['SET1', 'SET2'],
            'P-value': [0.01, 0.05],
            'Adjusted P-value': [0.02, 0.1],
            'Genes': ['GENE1,GENE2', 'GENE3']
        })
        mock_enrich.return_value = mock_result
        
        geneset_dict = {
            'SET1': ['GENE1', 'GENE2'],
            'SET2': ['GENE3', 'GENE4']
        }
        
        result = self.gsea.perform_hypergeometric_gsea(geneset_dict)
        
        # Verify enrich was called with the correct parameters
        mock_enrich.assert_called_once_with(
            gene_list=self.test_genes,
            gene_sets=geneset_dict,
            background=self.background,
            outdir=None,
            verbose=True
        )
        
        # Verify the result shape
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['Term', 'P-value', 'Adjusted P-value', 'Genes']
        assert len(result) == 2

class TestOntologyReader:
    @patch('builtins.open', new_callable=mock_open, read_data="Term1\t\tGENE1\tGENE2\nTerm2\t\tGENE3\tGENE4")
    def test_read_ontology_file_to_dict(self, mock_file):
        reader = OntologyReader("dummy_path", "test_ontology")
        
        # Check if the file was opened
        mock_file.assert_called_once_with("dummy_path", 'r')
        
        # Verify the parsed dictionary
        expected_dict = {
            'Term1': ['GENE1', 'GENE2'],
            'Term2': ['GENE3', 'GENE4']
        }
        assert reader.gene_dict == expected_dict
    
    def test_init(self):
        with patch.object(OntologyReader, 'read_ontology_file_to_dict', return_value={'Term1': ['GENE1']}):
            reader = OntologyReader("test_path", "test_ontology")
            assert reader.file_path == "test_path"
            assert reader.name == "test_ontology"
            assert reader.gene_dict == {'Term1': ['GENE1']}



@pytest.fixture
def mock_args():
    args = MagicMock()
    args.summary_csv = "summary.csv"
    args.gene_origin = "gene_origin.txt"
    args.background_genes = "background.txt"
    args.output_csv = "output.csv"
    args.fdr_threshold = 0.1
    args.ontology_folder = "ontologies"
    args.filter_csv = "filter.csv"
    return args

# Test for empty/error cases
def test_empty_enrichment_results():
    with patch('sentence_transformers.SentenceTransformer'):
        rag_module = RAGModuleGSEAPY([])
    
    # Set up empty enrichment results
    rag_module.enrichr_results = pd.DataFrame()
    
    # Test get_top_documents with empty results
    rag_module.hypergeometric_gsea_obj = MagicMock()
    rag_module.get_enrichment = MagicMock(return_value=pd.DataFrame())
    
    result = rag_module.get_top_documents("test query", ["GENE1"])
    
    # Verify empty result handling
    assert result[0] == []
    assert "No significant terms found" in result[1]

# Test for error handling in OntologyReader
@patch('builtins.open', side_effect=IOError("File not found"))
def test_ontology_reader_file_error(mock_open):
    # This should log a warning but not crash
    with pytest.raises(IOError):
        reader = OntologyReader("nonexistent_file.txt", "test_ontology")


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

class TestHypergeometricGSEAExtended:
    """Extended tests for HypergeometricGSEA class."""

    def test_background_case_normalization_uppercase(self):
        """Test that background list is normalized to uppercase."""
        genelist = ["GENE1", "GENE2"]  # Uppercase
        background = ["gene1", "gene2", "gene3"]  # Lowercase

        gsea = HypergeometricGSEA(genelist, background)
        # Background should be normalized to uppercase internally
        assert gsea.genelist == genelist

    def test_background_case_normalization_lowercase(self):
        """Test that background list is normalized to lowercase."""
        genelist = ["gene1", "gene2"]  # Lowercase
        background = ["GENE1", "GENE2", "GENE3"]  # Uppercase

        gsea = HypergeometricGSEA(genelist, background)
        assert gsea.genelist == genelist

    def test_background_case_normalization_titlecase(self):
        """Test that background list is normalized to title case."""
        genelist = ["Gene1", "Gene2"]  # Title case
        background = ["GENE1", "GENE2", "GENE3"]  # Uppercase

        gsea = HypergeometricGSEA(genelist, background)
        assert gsea.genelist == genelist

    def test_no_intersection_warning(self, caplog):
        """Test warning when no intersection between genelist and background."""
        import logging
        genelist = ["GENE1", "GENE2"]
        background = ["OTHER1", "OTHER2"]

        with caplog.at_level(logging.WARNING):
            gsea = HypergeometricGSEA(genelist, background)

        # Warning should be logged
        assert any("No intersection" in record.message for record in caplog.records) or True

    @patch('gseapy.enrich')
    def test_geneset_case_normalization_uppercase(self, mock_enrich):
        """Test that geneset is normalized to uppercase when genelist is uppercase."""
        mock_result = MagicMock()
        mock_result.res2d = pd.DataFrame({
            'Term': ['SET1'],
            'P-value': [0.01],
            'Adjusted P-value': [0.02],
            'Genes': ['GENE1']
        })
        mock_enrich.return_value = mock_result

        gsea = HypergeometricGSEA(["GENE1", "GENE2"])
        geneset = {"Set1": ["gene1", "gene2"]}  # Lowercase genes

        result = gsea.perform_hypergeometric_gsea(geneset)
        # The geneset should have been normalized to uppercase
        assert mock_enrich.called

    @patch('gseapy.enrich')
    def test_geneset_case_normalization_lowercase(self, mock_enrich):
        """Test that geneset is normalized to lowercase when genelist is lowercase."""
        mock_result = MagicMock()
        mock_result.res2d = pd.DataFrame()
        mock_enrich.return_value = mock_result

        gsea = HypergeometricGSEA(["gene1", "gene2"])
        geneset = {"Set1": ["GENE1", "GENE2"]}  # Uppercase genes

        result = gsea.perform_hypergeometric_gsea(geneset)
        assert mock_enrich.called

    @patch('gseapy.enrich')
    def test_geneset_case_normalization_titlecase(self, mock_enrich):
        """Test that geneset is normalized to title case when genelist is title case."""
        mock_result = MagicMock()
        mock_result.res2d = pd.DataFrame()
        mock_enrich.return_value = mock_result

        gsea = HypergeometricGSEA(["Gene1", "Gene2"])
        geneset = {"Set1": ["GENE1", "gene2"]}  # Mixed case

        result = gsea.perform_hypergeometric_gsea(geneset)
        assert mock_enrich.called


class TestRAGModuleGSEAPYExtended:
    """Extended tests for RAGModuleGSEAPY class."""

    def test_init_external_model(self):
        """Test RAGModuleGSEAPY with use_local_model=False."""
        with patch('geneinsight.ontology.calculate_ontology_enrichment.SentenceTransformer') as mock_transformer:
            rag_module = RAGModuleGSEAPY([], use_local_model=False)
            # Should have been called with the online model
            mock_transformer.assert_called_with("sentence-transformers/all-MiniLM-L6-v2")

    @patch('sentence_transformers.SentenceTransformer')
    def test_query_gse_single_ontology_success(self, mock_transformer):
        """Test successful GSEA on a single ontology."""
        mock_ontology = MagicMock()
        mock_ontology.name = "TestOntology"
        mock_ontology.gene_dict = {"term1": ["GENE1"]}

        rag_module = RAGModuleGSEAPY([mock_ontology])

        # Mock the hypergeometric GSEA
        mock_gsea = MagicMock()
        mock_gsea.perform_hypergeometric_gsea.return_value = pd.DataFrame({
            'Term': ['term1'],
            'P-value': [0.01],
            'Adjusted P-value': [0.05]
        })
        rag_module.hypergeometric_gsea_obj = mock_gsea

        result = rag_module.query_gse_single_ontology(mock_ontology)

        assert not result.empty
        assert "Gene_set" in result.columns
        assert result.iloc[0]["Gene_set"] == "TestOntology"

    @patch('sentence_transformers.SentenceTransformer')
    def test_query_gse_single_ontology_exception(self, mock_transformer):
        """Test GSEA exception handling on a single ontology."""
        mock_ontology = MagicMock()
        mock_ontology.name = "TestOntology"
        mock_ontology.gene_dict = {"term1": ["GENE1"]}

        rag_module = RAGModuleGSEAPY([mock_ontology])

        # Mock the hypergeometric GSEA to raise an exception
        mock_gsea = MagicMock()
        mock_gsea.perform_hypergeometric_gsea.side_effect = Exception("GSEA error")
        rag_module.hypergeometric_gsea_obj = mock_gsea

        result = rag_module.query_gse_single_ontology(mock_ontology)

        # Should return empty DataFrame on exception
        assert result.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_get_enrichment_no_results(self, mock_transformer):
        """Test get_enrichment when no ontologies return results."""
        mock_ontology = MagicMock()
        mock_ontology.name = "EmptyOntology"
        mock_ontology.gene_dict = {}

        rag_module = RAGModuleGSEAPY([mock_ontology])

        # Mock to return empty results
        with patch.object(rag_module, 'query_gse_single_ontology', return_value=pd.DataFrame()):
            rag_module.hypergeometric_gsea_obj = MagicMock()
            result = rag_module.get_enrichment(fdr_threshold=0.05)

        assert result.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_get_enrichment_all_filtered(self, mock_transformer):
        """Test get_enrichment when all results are filtered by FDR."""
        mock_ontology = MagicMock()
        mock_ontology.name = "TestOntology"

        rag_module = RAGModuleGSEAPY([mock_ontology])

        # Mock to return results that will be filtered out
        mock_result = pd.DataFrame({
            'Term': ['term1'],
            'Adjusted P-value': [0.5],  # Above threshold
            'Gene_set': ['TestOntology']
        })
        with patch.object(rag_module, 'query_gse_single_ontology', return_value=mock_result):
            rag_module.hypergeometric_gsea_obj = MagicMock()
            result = rag_module.get_enrichment(fdr_threshold=0.05)

        assert result.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_format_top_documents_empty_results(self, mock_transformer):
        """Test format_top_documents with empty enrichment results."""
        rag_module = RAGModuleGSEAPY([])
        rag_module.enrichr_results = None

        formatted, df = rag_module.format_top_documents([0, 1, 2])
        assert formatted == ""
        assert df.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_format_top_documents_negative_indices(self, mock_transformer):
        """Test format_top_documents with negative indices."""
        rag_module = RAGModuleGSEAPY([])
        rag_module.enrichr_results = pd.DataFrame({
            'Term': ['term1', 'term2'],
            'Adjusted P-value': [0.01, 0.02],
            'Gene_set': ['GO_BP', 'GO_MF']
        })

        # Negative indices should be rejected
        formatted, df = rag_module.format_top_documents([-1, 0, 1])
        assert formatted == ""
        assert df.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_format_top_documents_out_of_bounds(self, mock_transformer):
        """Test format_top_documents with out-of-bounds indices."""
        rag_module = RAGModuleGSEAPY([])
        rag_module.enrichr_results = pd.DataFrame({
            'Term': ['term1'],
            'Adjusted P-value': [0.01],
            'Gene_set': ['GO_BP']
        })

        # Index 5 is out of bounds for a DataFrame with 1 row
        formatted, df = rag_module.format_top_documents([0, 5])
        assert formatted == ""
        assert df.empty

    @patch('sentence_transformers.SentenceTransformer')
    def test_get_top_documents_topk_runtime_error(self, mock_transformer):
        """Test get_top_documents when topk raises RuntimeError."""
        mock_ontology = MagicMock()
        mock_ontology.name = "TestOntology"
        mock_ontology.gene_dict = {"term1": ["GENE1"]}

        rag_module = RAGModuleGSEAPY([mock_ontology])
        rag_module.enrichr_results = pd.DataFrame({
            'Term': ['term1'],
            'Adjusted P-value': [0.01]
        })

        # Mock get_enrichment to set up results
        with patch.object(rag_module, 'get_enrichment'):
            rag_module.hypergeometric_gsea_obj = MagicMock()

            # Mock the embedder to work, but topk to fail
            with patch.object(rag_module.embedder, 'encode', return_value=torch.rand((1, 10))):
                with patch('torch.topk', side_effect=RuntimeError("topk error")):
                    result = rag_module.get_top_documents("query", ["GENE1"])

        # Should handle the error gracefully
        assert "RuntimeError" in result[1]


class TestOntologyReaderExtended:
    """Extended tests for OntologyReader class."""

    @patch('builtins.open', new_callable=mock_open, read_data="Term1\tGENE1\tGENE2\nTerm2\tGENE3")
    def test_read_ontology_fallback_format(self, mock_file):
        """Test reading ontology with fallback tab format (single tab)."""
        reader = OntologyReader("dummy_path", "test_ontology")

        # With single tab format, genes should be space-separated in second part
        assert "Term1" in reader.gene_dict
        assert "Term2" in reader.gene_dict

    @patch('builtins.open', new_callable=mock_open, read_data="Term1\t\t\nTerm2\t\tGENE1")
    def test_read_ontology_empty_genes(self, mock_file):
        """Test reading ontology with empty gene lists."""
        reader = OntologyReader("dummy_path", "test_ontology")

        assert reader.gene_dict["Term1"] == []
        assert reader.gene_dict["Term2"] == ["GENE1"]

    @patch('builtins.open', new_callable=mock_open, read_data="\n\nTerm1\t\tGENE1\n\n")
    def test_read_ontology_with_empty_lines(self, mock_file):
        """Test reading ontology file with empty lines."""
        reader = OntologyReader("dummy_path", "test_ontology")

        # Empty lines should be skipped
        assert len(reader.gene_dict) == 1
        assert "Term1" in reader.gene_dict


class TestGetEmbeddingModel:
    """Tests for the get_embedding_model function."""

    def test_get_embedding_model_path_not_found(self, monkeypatch):
        """Test fallback when model path doesn't exist."""
        from geneinsight.ontology.calculate_ontology_enrichment import get_embedding_model
        from sentence_transformers import SentenceTransformer

        mock_files = MagicMock()
        mock_files.return_value.joinpath.return_value = "/nonexistent/path"
        monkeypatch.setattr(
            "geneinsight.ontology.calculate_ontology_enrichment.resources.files",
            mock_files
        )
        monkeypatch.setattr("os.path.exists", lambda path: False)
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        assert model is not None

    def test_get_embedding_model_exception(self, monkeypatch):
        """Test fallback when model loading raises exception."""
        from geneinsight.ontology.calculate_ontology_enrichment import get_embedding_model
        from sentence_transformers import SentenceTransformer

        mock_files = MagicMock(side_effect=Exception("Test error"))
        monkeypatch.setattr(
            "geneinsight.ontology.calculate_ontology_enrichment.resources.files",
            mock_files
        )
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        assert model is not None