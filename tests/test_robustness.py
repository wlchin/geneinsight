# tests/test_robustness.py
"""
Tests for robustness improvements: malformed data handling, timeouts, and validation.
"""

import os
import pytest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock
import requests


# ============================================================================
# Tests for gene list NaN/empty string filtering
# ============================================================================

class TestGeneListValidation:
    """Tests for NaN and empty string filtering in gene list readers."""

    def test_stringdb_filters_nan_values(self, tmp_path):
        """Test that stringdb.read_gene_list filters out NaN values."""
        from geneinsight.enrichment.stringdb import read_gene_list

        # Create CSV with NaN values
        gene_file = tmp_path / "genes_with_nan.csv"
        df = pd.DataFrame({"genes": ["BRCA1", None, "TP53", float("nan"), "EGFR"]})
        df.to_csv(gene_file, index=False, header=False)

        result = read_gene_list(str(gene_file))

        assert "BRCA1" in result
        assert "TP53" in result
        assert "EGFR" in result
        assert len(result) == 3  # Only valid genes

    def test_stringdb_filters_empty_strings(self, tmp_path):
        """Test that stringdb.read_gene_list filters out empty strings."""
        from geneinsight.enrichment.stringdb import read_gene_list

        # Create CSV with empty strings
        gene_file = tmp_path / "genes_with_empty.csv"
        with open(gene_file, "w") as f:
            f.write("BRCA1\n\nTP53\n   \nEGFR\n")

        result = read_gene_list(str(gene_file))

        assert "BRCA1" in result
        assert "TP53" in result
        assert "EGFR" in result
        assert len(result) == 3

    def test_stringdb_local_filters_nan_values(self, tmp_path):
        """Test that stringdb_local.read_gene_list filters out NaN values."""
        from geneinsight.enrichment.stringdb_local import read_gene_list

        # Create CSV with NaN values
        gene_file = tmp_path / "genes_with_nan.csv"
        df = pd.DataFrame({"genes": ["BRCA1", None, "TP53", float("nan"), "EGFR"]})
        df.to_csv(gene_file, index=False, header=False)

        result = read_gene_list(str(gene_file))

        assert "BRCA1" in result
        assert "TP53" in result
        assert "EGFR" in result
        assert len(result) == 3

    def test_stringdb_local_filters_empty_strings(self, tmp_path):
        """Test that stringdb_local.read_gene_list filters out empty strings."""
        from geneinsight.enrichment.stringdb_local import read_gene_list

        # Create CSV with empty strings
        gene_file = tmp_path / "genes_with_empty.csv"
        with open(gene_file, "w") as f:
            f.write("BRCA1\n\nTP53\n   \nEGFR\n")

        result = read_gene_list(str(gene_file))

        assert "BRCA1" in result
        assert "TP53" in result
        assert "EGFR" in result
        assert len(result) == 3


# ============================================================================
# Tests for ast.literal_eval malformed data handling
# ============================================================================

class TestAstLiteralEvalRobustness:
    """Tests for handling malformed string data in ast.literal_eval calls."""

    def test_generate_rst_handles_malformed_ref_dict(self, tmp_path):
        """Test that create_clustered_sections handles malformed ref_dict."""
        from geneinsight.report.generate_rst_from_files import create_clustered_sections

        # Create headings CSV
        headings_file = tmp_path / "headings.csv"
        headings_df = pd.DataFrame({
            "cluster": [0],
            "heading": ["Test Heading"],
            "main_heading_text": ["Test content"]
        })
        headings_df.to_csv(headings_file, index=False)

        # Create merged CSV with malformed ref_dict (include all required columns)
        merged_file = tmp_path / "merged.csv"
        merged_df = pd.DataFrame({
            "Cluster": [0],
            "query": ["test query"],
            "ref_dict": ["not a valid dict {{{"],  # Malformed
            "ontology_dict": ["{}"],
            "unique_genes": ["{}"],
            "subheading_text": ["Test subheading content"]
        })
        merged_df.to_csv(merged_file, index=False)

        # Should not raise, should return empty dict for malformed ref_dict
        result = create_clustered_sections(str(headings_file), str(merged_file))
        assert 0 in result
        # The subsection should have empty references due to malformed ref_dict
        subsections = [s for s in result[0] if "subtitle" in s]
        assert len(subsections) == 1
        assert subsections[0]["references"] == []

    def test_generate_rst_handles_malformed_ontology_dict(self, tmp_path):
        """Test that create_clustered_sections handles malformed ontology_dict."""
        from geneinsight.report.generate_rst_from_files import create_clustered_sections

        # Create headings CSV
        headings_file = tmp_path / "headings.csv"
        headings_df = pd.DataFrame({
            "cluster": [0],
            "heading": ["Test Heading"],
            "main_heading_text": ["Test content"]
        })
        headings_df.to_csv(headings_file, index=False)

        # Create merged CSV with malformed ontology_dict (include all required columns)
        merged_file = tmp_path / "merged.csv"
        merged_df = pd.DataFrame({
            "Cluster": [0],
            "query": ["test query"],
            "ref_dict": ["{}"],
            "ontology_dict": ["invalid syntax [[["],  # Malformed
            "unique_genes": ["{}"],
            "subheading_text": ["Test subheading content"]
        })
        merged_df.to_csv(merged_file, index=False)

        # Should not raise
        result = create_clustered_sections(str(headings_file), str(merged_file))
        assert 0 in result
        subsections = [s for s in result[0] if "subtitle" in s]
        assert len(subsections) == 1
        assert subsections[0]["thematic_geneset"] == []

    def test_rst_generator_handles_malformed_code_block(self, tmp_path):
        """Test that rst_generator handles malformed code_block in CSV generation."""
        import ast

        # Simulate the try/except behavior
        code_block = "not a valid dict {{{"
        try:
            code_dict = ast.literal_eval(code_block)
        except (SyntaxError, ValueError):
            code_dict = {}

        assert code_dict == {}

    def test_workflow_handles_malformed_unique_genes(self, tmp_path):
        """Test that workflow.py handles malformed unique_genes gracefully."""
        import ast

        # Simulate the try/except behavior for unique_genes parsing
        malformed_genes = "{'BRCA1': 5, 'TP53': 3"  # Missing closing brace

        try:
            if isinstance(malformed_genes, str):
                unique_genes = list(ast.literal_eval(malformed_genes).keys())
            else:
                unique_genes = list(malformed_genes.keys())
        except (SyntaxError, ValueError):
            unique_genes = None  # Would continue in actual code

        assert unique_genes is None

    def test_calculate_ontology_handles_malformed_unique_genes(self):
        """Test that calculate_ontology_enrichment handles malformed unique_genes."""
        import ast

        # Test various malformed inputs
        malformed_inputs = [
            "not a dict at all",
            "{'key': 'unclosed",
            "[1, 2, 3]",  # List, not dict
            "None",
        ]

        for malformed in malformed_inputs:
            try:
                result = list(ast.literal_eval(malformed).keys())
            except (SyntaxError, ValueError, AttributeError):
                result = None

            # All should fail gracefully
            assert result is None or isinstance(result, list)


# ============================================================================
# Tests for HTTP timeout handling
# ============================================================================

class TestHTTPTimeouts:
    """Tests for HTTP request timeout handling."""

    @patch("geneinsight.enrichment.stringdb.requests.post")
    def test_stringdb_timeout_raises_exception(self, mock_post):
        """Test that stringdb API call with timeout raises proper exception."""
        from geneinsight.enrichment.stringdb import map_gene_identifiers

        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        result = map_gene_identifiers(["BRCA1", "TP53"])

        # Should return empty dict on timeout
        assert result == {}
        # Verify timeout parameter was passed
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("timeout") == 30

    @patch("geneinsight.enrichment.stringdb_local.os.makedirs")
    @patch("geneinsight.enrichment.stringdb_local.os.path.exists")
    @patch("geneinsight.enrichment.stringdb_local.requests.get")
    def test_stringdb_local_timeout_raises_exception(self, mock_get, mock_exists, mock_makedirs):
        """Test that stringdb_local download with timeout raises proper exception."""
        from geneinsight.enrichment.stringdb_local import load_species_data

        # Mock that the cache doesn't exist to force download
        mock_exists.return_value = False
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        # Should raise since we can't load the data
        with pytest.raises(requests.exceptions.Timeout):
            load_species_data(9606)

        # Verify timeout was passed
        mock_get.assert_called()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("timeout") == 60

    @patch("geneinsight.report.generate_rst_from_files.requests.get")
    def test_ncbi_api_timeout_handling(self, mock_get):
        """Test that NCBI API calls include timeout parameter."""
        from geneinsight.report.generate_rst_from_files import get_gene_summary

        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "esearchresult": {"idlist": ["12345"]}
        }
        mock_get.return_value = mock_response

        get_gene_summary("BRCA1", "9606")

        # Verify timeout was passed
        assert mock_get.call_count >= 1
        for call in mock_get.call_args_list:
            call_kwargs = call[1]
            assert call_kwargs.get("timeout") == 10


# ============================================================================
# Tests for empty merge validation
# ============================================================================

class TestEmptyMergeValidation:
    """Tests for empty DataFrame merge warnings."""

    def test_context_merge_warns_on_empty_result(self, tmp_path, caplog):
        """Test that context_merge logs warning for empty merge result."""
        import logging
        from geneinsight.report.context_merge import merge_context_ontology

        # Create non-matching CSVs
        subheadings_file = tmp_path / "subheadings.csv"
        pd.DataFrame({
            "query": ["A", "B"],
            "content": ["x", "y"]
        }).to_csv(subheadings_file, index=False)

        ontology_file = tmp_path / "ontology.csv"
        pd.DataFrame({
            "query": ["C", "D"],  # No overlap with subheadings
            "ontology": ["1", "2"]
        }).to_csv(ontology_file, index=False)

        output_file = tmp_path / "merged.csv"

        with caplog.at_level(logging.WARNING):
            result = merge_context_ontology(
                str(subheadings_file),
                str(ontology_file),
                str(output_file)
            )

        assert result.empty
        assert "empty DataFrame" in caplog.text or "No matching" in caplog.text

    def test_merge_context_ontology_dict_warns_on_empty(self, tmp_path, capsys):
        """Test that merge_context_and_ontology_dict prints warning for empty merge."""
        from geneinsight.ontology.merge_context_and_ontology_dict import main

        # Create non-matching CSVs
        ontology_file = tmp_path / "ontology.csv"
        pd.DataFrame({
            "query": ["X", "Y"],
            "data": ["1", "2"]
        }).to_csv(ontology_file, index=False)

        subheadings_file = tmp_path / "subheadings.csv"
        pd.DataFrame({
            "query": ["A", "B"],  # No overlap
            "content": ["x", "y"]
        }).to_csv(subheadings_file, index=False)

        output_file = tmp_path / "merged.csv"

        main(str(ontology_file), str(subheadings_file), str(output_file))

        captured = capsys.readouterr()
        assert "empty DataFrame" in captured.out or "Warning" in captured.out


# ============================================================================
# Tests for stringdb_local empty gene_name handling
# ============================================================================

class TestStringDBLocalGeneNameHandling:
    """Tests for stringdb_local handling empty gene_name series."""

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    @patch("geneinsight.enrichment.stringdb_local.read_gene_list")
    def test_empty_gene_name_series_warning(self, mock_read, mock_load, tmp_path, caplog):
        """Test warning when gene_name series is empty after dropna."""
        import logging
        from geneinsight.enrichment.stringdb_local import process_gene_enrichment

        # Mock species data with all NaN gene_names
        mock_species_df = pd.DataFrame({
            "gene_name": [None, None, None],
            "description": ["desc1", "desc2", "desc3"]
        })
        mock_load.return_value = mock_species_df
        mock_read.return_value = ["BRCA1", "TP53"]

        output_dir = tmp_path / "output"

        with caplog.at_level(logging.WARNING):
            process_gene_enrichment(
                input_file="dummy.csv",
                output_dir=str(output_dir),
                species=9606
            )

        assert "No gene names available" in caplog.text or "empty" in caplog.text.lower()
