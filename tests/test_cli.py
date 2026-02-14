#!/usr/bin/env python3
"""
Tests for the updated geneinsight.cli module.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from geneinsight.cli import main

class TestGeneInsightCLI:
    """Tests for the geneinsight.cli main() function."""

    @patch("geneinsight.cli.Pipeline")
    def test_minimal_args(self, mock_pipeline):
        """
        Test the CLI with only the required arguments:
        1) query_gene_set
        2) background_gene_list
        """
        test_args = [
            "geneinsight",          # Simulate script name
            "query_genes.txt",
            "background_genes.txt"
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Ensure Pipeline was called exactly once
        mock_pipeline.assert_called_once()

        # Unpack the Pipeline call
        call_args, call_kwargs = mock_pipeline.call_args

        # Pipeline is called only with keyword arguments, so `call_args` should be empty
        assert call_args == (), "Pipeline should only be called with keyword arguments."

        # Verify the keyword arguments match the defaults
        expected_kwargs = {
            'output_dir': './output',
            'temp_dir': None,
            'n_samples': 5,  # default
            'num_topics': None,
            'pvalue_threshold': 0.05,
            'api_service': 'openai',
            'api_model': 'gpt-4o-mini',
            'api_parallel_jobs': 1,
            'api_base_url': None,
            'target_filtered_topics': 25,
            'species': 9606,
            'filtered_n_samples': 10,  # Added new default parameter
            'api_temperature': 0.2,  # Added new default parameter
            'call_ncbi_api': False,  # Added new parameter
            'use_local_stringdb': False,  # Added new parameter
            'overlap_ratio_threshold': 0.25,  # Added new parameter
            'enable_metrics': True,  # Added new parameter
            'quiet_metrics': False,  # Added new parameter
            'metrics_output_path': None  # Added new parameter
        }
        assert call_kwargs == expected_kwargs, "Pipeline got unexpected init kwargs."

        # Check run() call
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.run.assert_called_once_with(
            query_gene_set="query_genes.txt",
            background_gene_list="background_genes.txt",
            generate_report=True,  # not passing --no-report
            report_title=None
        )

    @patch("geneinsight.cli.Pipeline")
    def test_full_args(self, mock_pipeline):
        """
        Test passing all known arguments to ensure they are handled correctly.
        """
        test_args = [
            "geneinsight",
            "query.txt",
            "background.txt",
            "-o", "my_output",
            "--no-report",
            "--n_samples", "10",
            "--num_topics", "20",
            "--pvalue_threshold", "0.01",
            "--api_service", "anthropic",
            "--api_model", "claude-3",
            "--api_parallel_jobs", "25",
            "--api_base_url", "https://example.com/api",
            "--target_filtered_topics", "30",
            "--temp_dir", "my_temp_dir",
            "--report_title", "My Custom Title",
            "--species", "10090",  # Mouse
            "--filtered_n_samples", "15",
            "--api_temperature", "0.5"
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_pipeline.assert_called_once()

        # Unpack the Pipeline call
        call_args, call_kwargs = mock_pipeline.call_args

        # Ensure no positional arguments
        assert call_args == (), "Pipeline should only be called with keyword arguments."

        # Check the keyword arguments
        expected_kwargs = {
            'output_dir': 'my_output',
            'temp_dir': 'my_temp_dir',
            'n_samples': 10,
            'num_topics': 20,
            'pvalue_threshold': 0.01,
            'api_service': 'anthropic',
            'api_model': 'claude-3',
            'api_parallel_jobs': 25,
            'api_base_url': 'https://example.com/api',
            'target_filtered_topics': 30,
            'species': 10090,
            'filtered_n_samples': 15,  # Added new parameter
            'api_temperature': 0.5,  # Added new parameter
            'call_ncbi_api': False,  # Added new parameter (default)
            'use_local_stringdb': False,  # Added new parameter (default)
            'overlap_ratio_threshold': 0.25,  # Added new parameter (default)
            'enable_metrics': True,  # Added new parameter (default)
            'quiet_metrics': False,  # Added new parameter (default)
            'metrics_output_path': None  # Added new parameter (default)
        }
        assert call_kwargs == expected_kwargs, (
            "Pipeline got unexpected init kwargs when passing all CLI parameters."
        )

        # Ensure run() is called correctly
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.run.assert_called_once_with(
            query_gene_set="query.txt",
            background_gene_list="background.txt",
            generate_report=False,  # we set --no-report
            report_title="My Custom Title"
        )

    @patch("geneinsight.cli.Pipeline")
    def test_missing_arguments(self, mock_pipeline):
        """
        Test behavior if one of the required args is missing.
        argparse should automatically raise a SystemExit error.
        """
        # Provide only one of the two required positional arguments
        test_args = [
            "geneinsight",
            "query_genes.txt",
            # Missing background_genes.txt
        ]
        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as excinfo:
            main()

        # Ensure it fails with a proper usage error (status code 2 for argparse)
        assert excinfo.type == SystemExit
        assert excinfo.value.code == 2
        mock_pipeline.assert_not_called()

    @patch("geneinsight.cli.Pipeline")
    def test_pipeline_exception(self, mock_pipeline):
        """
        Test that if the pipeline.run() raises an exception,
        it is caught and returns error code 1. The CLI now catches exceptions
        and returns an error code instead of raising.
        """
        # Provide both required arguments
        test_args = [
            "geneinsight",
            "query.txt",
            "background.txt"
        ]
        mock_pipeline_instance = mock_pipeline.return_value
        mock_pipeline_instance.run.side_effect = Exception("Pipeline error")

        with patch.object(sys, "argv", test_args):
            # main() now returns 1 on error instead of raising
            result = main()

        assert result == 1  # Error return code
        # Check we tried to run the pipeline
        mock_pipeline.assert_called_once()
        mock_pipeline_instance.run.assert_called_once()
        
    @patch("geneinsight.cli.Pipeline")
    def test_new_parameters(self, mock_pipeline):
        """
        Test the new parameters: filtered_n_samples and api_temperature
        """
        test_args = [
            "geneinsight",
            "query.txt",
            "background.txt",
            "--filtered_n_samples", "20",
            "--api_temperature", "0.7"
        ]
        with patch.object(sys, "argv", test_args):
            main()

        mock_pipeline.assert_called_once()
        
        # Check the new parameters were correctly passed
        call_args, call_kwargs = mock_pipeline.call_args
        assert call_kwargs['filtered_n_samples'] == 20
        assert call_kwargs['api_temperature'] == 0.7

if __name__ == "__main__":
    pytest.main(["-v", __file__])