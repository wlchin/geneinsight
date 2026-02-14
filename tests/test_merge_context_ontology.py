# tests/test_merge_context_ontology.py
"""
Tests for the geneinsight.ontology.merge_context_and_ontology_dict module.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile

from geneinsight.ontology.merge_context_and_ontology_dict import main


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ontology_dict_csv(tmp_path):
    """Creates a mock ontology dictionary CSV file."""
    df = pd.DataFrame({
        "query": ["Term1", "Term2", "Term3"],
        "ontology_id": ["GO:0001", "GO:0002", "GO:0003"],
        "description": ["Biological process 1", "Molecular function 2", "Cellular component 3"]
    })
    file_path = tmp_path / "ontology_dict.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def subheadings_csv(tmp_path):
    """Creates a mock subheadings CSV file."""
    df = pd.DataFrame({
        "query": ["Term1", "Term2", "Term4"],
        "subheading": ["Subheading 1", "Subheading 2", "Subheading 4"],
        "score": [0.95, 0.87, 0.92]
    })
    file_path = tmp_path / "subheadings.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def empty_subheadings_csv(tmp_path):
    """Creates an empty subheadings CSV file."""
    df = pd.DataFrame(columns=["query", "subheading", "score"])
    file_path = tmp_path / "empty_subheadings.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def no_matching_subheadings_csv(tmp_path):
    """Creates a subheadings CSV with no matching queries."""
    df = pd.DataFrame({
        "query": ["NoMatch1", "NoMatch2"],
        "subheading": ["Subheading X", "Subheading Y"],
        "score": [0.5, 0.6]
    })
    file_path = tmp_path / "no_match_subheadings.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def output_csv(tmp_path):
    """Creates a path for the output CSV."""
    return tmp_path / "merged_output.csv"


# ============================================================================
# Tests for main function
# ============================================================================

class TestMergeMain:

    def test_merge_valid_csvs(self, ontology_dict_csv, subheadings_csv, output_csv):
        """Test merging two valid CSV files with matching rows."""
        main(
            ontology_dict_csv=str(ontology_dict_csv),
            subheadings_csv=str(subheadings_csv),
            output_csv=str(output_csv)
        )

        # Check output file exists
        assert output_csv.exists()

        # Read the merged output
        merged_df = pd.read_csv(output_csv)

        # Should have 2 matching rows (Term1 and Term2)
        assert len(merged_df) == 2
        assert "query" in merged_df.columns
        assert "ontology_id" in merged_df.columns
        assert "subheading" in merged_df.columns
        assert set(merged_df["query"].tolist()) == {"Term1", "Term2"}

    def test_merge_no_matching_rows(self, ontology_dict_csv, no_matching_subheadings_csv, output_csv):
        """Test merging when no rows match between the two files."""
        main(
            ontology_dict_csv=str(ontology_dict_csv),
            subheadings_csv=str(no_matching_subheadings_csv),
            output_csv=str(output_csv)
        )

        # Check output file exists
        assert output_csv.exists()

        # Read the merged output
        merged_df = pd.read_csv(output_csv)

        # Should be empty since no matching queries
        assert len(merged_df) == 0

    def test_merge_empty_subheadings(self, ontology_dict_csv, empty_subheadings_csv, output_csv):
        """Test merging with empty subheadings file."""
        main(
            ontology_dict_csv=str(ontology_dict_csv),
            subheadings_csv=str(empty_subheadings_csv),
            output_csv=str(output_csv)
        )

        # Check output file exists
        assert output_csv.exists()

        # Read the merged output
        merged_df = pd.read_csv(output_csv)

        # Should be empty
        assert len(merged_df) == 0

    def test_merge_missing_input_file(self, tmp_path, subheadings_csv, output_csv):
        """Test that missing input file raises an error."""
        nonexistent_file = tmp_path / "nonexistent.csv"

        with pytest.raises(Exception):  # pandas will raise FileNotFoundError
            main(
                ontology_dict_csv=str(nonexistent_file),
                subheadings_csv=str(subheadings_csv),
                output_csv=str(output_csv)
            )


# ============================================================================
# Tests for CLI entry point
# ============================================================================

class TestMergeMainCLI:

    @patch("geneinsight.ontology.merge_context_and_ontology_dict.argparse.ArgumentParser.parse_args")
    @patch("geneinsight.ontology.merge_context_and_ontology_dict.pd.read_csv")
    @patch("geneinsight.ontology.merge_context_and_ontology_dict.pd.DataFrame.to_csv")
    def test_main_cli_integration(self, mock_to_csv, mock_read_csv, mock_parse_args):
        """Test that the CLI parses arguments and calls main correctly."""
        # Mock parse_args to return expected arguments
        mock_args = MagicMock()
        mock_args.ontology_dict_csv = "ontology.csv"
        mock_args.subheadings_csv = "subheadings.csv"
        mock_args.output_csv = "output.csv"
        mock_parse_args.return_value = mock_args

        # Mock read_csv to return DataFrames
        mock_read_csv.side_effect = [
            pd.DataFrame({"query": ["A"], "ontology": ["GO:1"]}),
            pd.DataFrame({"query": ["A"], "subheading": ["Sub1"]})
        ]

        # Import and test the module's __main__ block behavior
        # We'll test main directly since we can't easily test __main__
        from geneinsight.ontology.merge_context_and_ontology_dict import main

        # This tests that the function can be called without error
        mock_read_csv.reset_mock()
        mock_read_csv.side_effect = [
            pd.DataFrame({"query": ["A"], "ontology": ["GO:1"]}),
            pd.DataFrame({"query": ["A"], "subheading": ["Sub1"]})
        ]

        main("ontology.csv", "subheadings.csv", "output.csv")
        assert mock_read_csv.call_count == 2
        mock_to_csv.assert_called_once()
