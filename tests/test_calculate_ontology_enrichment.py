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