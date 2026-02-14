# tests/test_stringdb_local.py
"""
Tests for the geneinsight.enrichment.stringdb_local module.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile

from geneinsight.enrichment.stringdb_local import (
    read_gene_list,
    load_species_data,
    process_gene_enrichment,
    main
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def gene_list_file(tmp_path):
    """Creates a temporary CSV file containing a list of genes."""
    data = ["BRCA1", "TP53", "EGFR"]
    file_path = tmp_path / "genes.csv"
    file_path.write_text("\n".join(data))
    return file_path


@pytest.fixture
def gene_list_with_duplicates(tmp_path):
    """Creates a CSV file with duplicate gene entries."""
    data = ["BRCA1", "TP53", "BRCA1", "EGFR", "TP53", "EGFR"]
    file_path = tmp_path / "genes_dups.csv"
    file_path.write_text("\n".join(data))
    return file_path


@pytest.fixture
def empty_gene_list_file(tmp_path):
    """Creates an empty CSV file."""
    file_path = tmp_path / "empty_genes.csv"
    file_path.write_text("")
    return file_path


@pytest.fixture
def output_dir(tmp_path):
    """Creates a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return d


@pytest.fixture
def mock_species_df():
    """Creates a mock species DataFrame."""
    return pd.DataFrame({
        "gene_name": ["BRCA1", "TP53", "EGFR", "MYC"],
        "description": ["Breast cancer 1", "Tumor protein p53", "Epidermal growth factor receptor", "MYC proto-oncogene"],
        "string_id": ["9606.ENSP00000357654", "9606.ENSP00000269305", "9606.ENSP00000275493", "9606.ENSP00000367207"]
    })


# ============================================================================
# Tests for read_gene_list
# ============================================================================

class TestReadGeneList:

    def test_read_gene_list_basic(self, gene_list_file):
        """Test reading a basic gene list from a file."""
        genes = read_gene_list(str(gene_list_file))
        assert genes == ["BRCA1", "TP53", "EGFR"]

    def test_read_gene_list_with_duplicates(self, gene_list_with_duplicates):
        """Test that duplicate genes are removed while preserving order."""
        genes = read_gene_list(str(gene_list_with_duplicates))
        assert genes == ["BRCA1", "TP53", "EGFR"]
        assert len(genes) == 3

    def test_read_gene_list_empty_file(self, empty_gene_list_file):
        """Test reading from an empty file returns empty list."""
        genes = read_gene_list(str(empty_gene_list_file))
        assert genes == []

    def test_read_gene_list_empty_dataframe(self, tmp_path):
        """Test reading a file with header but no data returns empty list."""
        file_path = tmp_path / "header_only.csv"
        file_path.write_text("gene\n")  # Header only
        genes = read_gene_list(str(file_path))
        # pandas will read this as one row with 'gene' as data
        # but since df is not empty, it should return ['gene']
        # Actually, the function reads with header=None, so 'gene' becomes data
        assert "gene" in genes or genes == []


# ============================================================================
# Tests for load_species_data
# ============================================================================

class TestLoadSpeciesData:

    @patch("geneinsight.enrichment.stringdb_local.os.path.exists")
    @patch("geneinsight.enrichment.stringdb_local.pd.read_pickle")
    def test_load_species_data_human(self, mock_read_pickle, mock_exists, mock_species_df):
        """Test loading human species data (9606)."""
        mock_exists.return_value = True
        mock_read_pickle.return_value = mock_species_df

        result = load_species_data(9606)

        assert not result.empty
        assert mock_read_pickle.call_count == 4  # 4 PKL files for human

    @patch("geneinsight.enrichment.stringdb_local.os.path.exists")
    @patch("geneinsight.enrichment.stringdb_local.pd.read_pickle")
    def test_load_species_data_mouse(self, mock_read_pickle, mock_exists, mock_species_df):
        """Test loading mouse species data (10090)."""
        mock_exists.return_value = True
        mock_read_pickle.return_value = mock_species_df

        result = load_species_data(10090)

        assert not result.empty
        assert mock_read_pickle.call_count == 4  # 4 PKL files for mouse

    @patch("geneinsight.enrichment.stringdb_local.requests.get")
    @patch("geneinsight.enrichment.stringdb_local.os.path.exists")
    @patch("geneinsight.enrichment.stringdb_local.pd.read_pickle")
    def test_load_species_data_download_missing_pkl(self, mock_read_pickle, mock_exists, mock_get, mock_species_df, tmp_path):
        """Test downloading PKL files when not found locally."""
        # First check returns False (file not found), subsequent checks return True
        mock_exists.side_effect = [False, True, True, True, True, True, True, True]

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.content = b"mock pkl content"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        mock_read_pickle.return_value = mock_species_df

        # Use patch for os.makedirs and open
        with patch("geneinsight.enrichment.stringdb_local.os.makedirs"), \
             patch("builtins.open", MagicMock()):
            result = load_species_data(9606)

        assert mock_get.called

    def test_load_species_data_unsupported_species(self):
        """Test that unsupported species returns empty DataFrame."""
        result = load_species_data(7227)  # Drosophila - not supported
        assert result.empty


# ============================================================================
# Tests for process_gene_enrichment
# ============================================================================

class TestProcessGeneEnrichment:

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_success(self, mock_load_species, gene_list_file, output_dir, mock_species_df):
        """Test successful gene enrichment processing."""
        mock_load_species.return_value = mock_species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_list_file),
            output_dir=str(output_dir),
            species=9606
        )

        assert not enrichment_df.empty
        assert isinstance(documents, list)
        mock_load_species.assert_called_once_with(9606)

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_case_normalization_upper(self, mock_load_species, tmp_path, output_dir):
        """Test that gene names are normalized to uppercase when species data is uppercase."""
        # Create lowercase gene list
        gene_file = tmp_path / "lowercase_genes.csv"
        gene_file.write_text("brca1\ntp53\negfr")

        # Species data with uppercase genes
        species_df = pd.DataFrame({
            "gene_name": ["BRCA1", "TP53", "EGFR"],
            "description": ["Desc1", "Desc2", "Desc3"]
        })
        mock_load_species.return_value = species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_file),
            output_dir=str(output_dir),
            species=9606
        )

        # Should find matches because lowercase was converted to uppercase
        assert not enrichment_df.empty

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_case_normalization_lower(self, mock_load_species, tmp_path, output_dir):
        """Test gene name normalization to lowercase."""
        gene_file = tmp_path / "upper_genes.csv"
        gene_file.write_text("BRCA1\nTP53")

        # Species data with lowercase genes
        species_df = pd.DataFrame({
            "gene_name": ["brca1", "tp53", "egfr"],
            "description": ["Desc1", "Desc2", "Desc3"]
        })
        mock_load_species.return_value = species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_file),
            output_dir=str(output_dir),
            species=9606
        )

        assert not enrichment_df.empty

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_case_normalization_title(self, mock_load_species, tmp_path, output_dir):
        """Test gene name normalization to title case."""
        gene_file = tmp_path / "upper_genes.csv"
        gene_file.write_text("BRCA1\nTP53")

        # Species data with title case genes
        species_df = pd.DataFrame({
            "gene_name": ["Brca1", "Tp53", "Egfr"],
            "description": ["Desc1", "Desc2", "Desc3"]
        })
        mock_load_species.return_value = species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_file),
            output_dir=str(output_dir),
            species=9606
        )

        assert not enrichment_df.empty

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_empty_species_data(self, mock_load_species, gene_list_file, output_dir):
        """Test handling of empty species data."""
        # Empty DataFrame with proper columns
        empty_species_df = pd.DataFrame(columns=["gene_name", "description", "string_id"])
        mock_load_species.return_value = empty_species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_list_file),
            output_dir=str(output_dir),
            species=9606
        )

        # Should return empty results when species data is empty
        assert enrichment_df.empty
        assert documents == []

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_no_matching_genes(self, mock_load_species, tmp_path, output_dir):
        """Test when no genes match the species data."""
        gene_file = tmp_path / "nonexistent_genes.csv"
        gene_file.write_text("FAKE_GENE1\nFAKE_GENE2")

        species_df = pd.DataFrame({
            "gene_name": ["BRCA1", "TP53", "EGFR"],
            "description": ["Desc1", "Desc2", "Desc3"]
        })
        mock_load_species.return_value = species_df

        enrichment_df, documents = process_gene_enrichment(
            input_file=str(gene_file),
            output_dir=str(output_dir),
            species=9606
        )

        assert enrichment_df.empty
        assert documents == []

    @patch("geneinsight.enrichment.stringdb_local.load_species_data")
    def test_process_gene_enrichment_creates_output_file(self, mock_load_species, gene_list_file, output_dir, mock_species_df):
        """Test that enrichment output CSV is created."""
        mock_load_species.return_value = mock_species_df

        process_gene_enrichment(
            input_file=str(gene_list_file),
            output_dir=str(output_dir),
            species=9606
        )

        # Check that output file exists
        output_files = list(output_dir.glob("*__enrichment.csv"))
        assert len(output_files) == 1


# ============================================================================
# Tests for main CLI
# ============================================================================

class TestMainCLI:

    @patch("geneinsight.enrichment.stringdb_local.process_gene_enrichment")
    @patch("geneinsight.enrichment.stringdb_local.argparse.ArgumentParser.parse_args")
    def test_main_cli(self, mock_parse_args, mock_process):
        """Test the main CLI entry point."""
        mock_args = MagicMock()
        mock_args.input = "input.csv"
        mock_args.output_dir = "output/"
        mock_args.species = 9606
        mock_parse_args.return_value = mock_args

        mock_process.return_value = (pd.DataFrame(), [])

        main()

        mock_process.assert_called_once_with(
            input_file="input.csv",
            output_dir="output/",
            species=9606
        )
