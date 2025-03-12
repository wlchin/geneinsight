#!/usr/bin/env python3
"""
Tests for the CLI module of the GeneInsight package.
"""

import os
import sys
import json
import yaml
import pytest
from unittest.mock import patch, MagicMock, mock_open
from argparse import Namespace

# Import module to test
from geneinsight.cli import (
    DEFAULT_CONFIG,
    load_config,
    parse_args,
    main,
    cli_main
)

class TestLoadConfig:
    """Tests for the load_config function."""
    
    def test_load_json_config(self, tmp_path):
        """Test loading a JSON configuration file."""
        config = {"n_samples": 10, "api_model": "custom-model"}
        config_file = tmp_path / "config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config
    
    def test_load_yaml_config(self, tmp_path):
        """Test loading a YAML configuration file."""
        config = {"n_samples": 10, "api_model": "custom-model"}
        config_file = tmp_path / "config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        loaded_config = load_config(str(config_file))
        assert loaded_config == config
    
    def test_file_not_found(self):
        """Test handling of a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("/path/does/not/exist.json")
    
    def test_unsupported_format(self, tmp_path):
        """Test handling of an unsupported file format."""
        config_file = tmp_path / "config.txt"
        config_file.touch()
        
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(str(config_file))
    
    def test_invalid_json(self, tmp_path):
        """Test handling of an invalid JSON file."""
        config_file = tmp_path / "config.json"
        
        with open(config_file, 'w') as f:
            f.write("{invalid: json,")
        
        with pytest.raises(ValueError, match="Error loading configuration"):
            load_config(str(config_file))
    
    def test_invalid_yaml(self, tmp_path):
        """Test handling of an invalid YAML file."""
        config_file = tmp_path / "config.yaml"
        
        with open(config_file, 'w') as f:
            f.write("invalid: yaml:\n  - item")
        
        with pytest.raises(ValueError, match="Error loading configuration"):
            load_config(str(config_file))


class TestParseArgs:
    """Tests for the parse_args function."""
    
    def test_minimal_args(self):
        """Test parsing with only required arguments."""
        test_args = [
            "query_genes.txt",
            "background_genes.txt"
        ]
        
        with patch.object(sys, 'argv', ['geneinsight'] + test_args):
            args = parse_args()
        
        assert args.query_gene_set == "query_genes.txt"
        assert args.background_gene_list == "background_genes.txt"
        assert args.output_dir == "./output"
        assert args.temp_dir is None
        assert args.n_samples is None
        assert args.no_report is False
    
    def test_full_args(self):
        """Test parsing with all possible arguments."""
        test_args = [
            "query_genes.txt",
            "background_genes.txt",
            "--output-dir", "/custom/output",
            "--temp-dir", "/custom/temp",
            "--n-samples", "10",
            "--num-topics", "15",
            "--pvalue-threshold", "0.05",
            "--api-service", "anthropic",
            "--api-model", "claude-3-opus",
            "--api-parallel-jobs", "20",
            "--api-base-url", "https://api.anthropic.com",
            "--target-filtered-topics", "30",
            "--no-report",
            "--config", "config.yaml",
            "--report-title", "Custom Report Title"
        ]
        
        with patch.object(sys, 'argv', ['geneinsight'] + test_args):
            args = parse_args()
        
        assert args.query_gene_set == "query_genes.txt"
        assert args.background_gene_list == "background_genes.txt"
        assert args.output_dir == "/custom/output"
        assert args.temp_dir == "/custom/temp"
        assert args.n_samples == 10
        assert args.num_topics == 15
        assert args.pvalue_threshold == 0.05
        assert args.api_service == "anthropic"
        assert args.api_model == "claude-3-opus"
        assert args.api_parallel_jobs == 20
        assert args.api_base_url == "https://api.anthropic.com"
        assert args.target_filtered_topics == 30
        assert args.no_report is True
        assert args.config == "config.yaml"
        assert args.report_title == "Custom Report Title"


class TestMainFunction:
    """Tests for the main function."""
    
    @patch('geneinsight.cli.Pipeline')
    @patch('geneinsight.cli.load_config')
    @patch('geneinsight.cli.parse_args')
    @patch('os.path.exists')
    def test_main_success(self, mock_exists, mock_parse_args, mock_load_config, mock_pipeline):
        """Test successful execution of the main function."""
        # Set up mocks
        mock_exists.return_value = True
        
        mock_args = MagicMock()
        mock_args.query_gene_set = "query.txt"
        mock_args.background_gene_list = "background.txt"
        mock_args.output_dir = "output"
        mock_args.temp_dir = None
        mock_args.n_samples = 10
        mock_args.num_topics = 15
        mock_args.pvalue_threshold = 0.05
        mock_args.api_service = "openai"
        mock_args.api_model = "gpt-4"
        mock_args.api_parallel_jobs = 30
        mock_args.api_base_url = None
        mock_args.target_filtered_topics = 20
        mock_args.no_report = False
        mock_args.config = "config.json"
        mock_args.report_title = "Test Report"
        
        mock_parse_args.return_value = mock_args
        
        mock_config = {
            "n_samples": 5,
            "api_model": "default-model"
        }
        mock_load_config.return_value = mock_config
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = "output/results"
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Run the function
        main()
        
        # Check that the pipeline was initialized and run with correct parameters
        mock_pipeline.assert_called_once_with(
            output_dir="output",
            temp_dir=None,
            n_samples=10,  # From command-line args
            num_topics=15,
            pvalue_threshold=0.05,
            api_service="openai",
            api_model="gpt-4",
            api_parallel_jobs=30,
            api_base_url=None,
            target_filtered_topics=20,
        )
        
        mock_pipeline_instance.run.assert_called_once_with(
            query_gene_set="query.txt",
            background_gene_list="background.txt",
            generate_report=True,
            report_title="Test Report"
        )
    
    @patch('geneinsight.cli.Pipeline')
    @patch('geneinsight.cli.parse_args')
    @patch('os.path.exists')
    @patch('sys.exit')
    def test_main_missing_files(self, mock_exit, mock_exists, mock_parse_args, mock_pipeline):
        """Test main when input files are missing."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.query_gene_set = "query.txt"
        mock_args.background_gene_list = "background.txt"
        mock_args.config = None
        mock_args.no_report = False
        mock_args.report_title = None
        
        mock_parse_args.return_value = mock_args
        
        # First file exists, second doesn't
        mock_exists.side_effect = lambda path: path == "query.txt"
        
        # The function exits early when a file is missing, so we need to make sure
        # sys.exit actually stops execution
        mock_exit.side_effect = SystemExit(1)
        
        # Run the function
        with pytest.raises(SystemExit):
            main()
        
        # Check that sys.exit was called with an error code
        mock_exit.assert_called_once_with(1)
        
        # Pipeline should not have been initialized
        mock_pipeline.assert_not_called()
    
    @patch('geneinsight.cli.Pipeline')
    @patch('geneinsight.cli.parse_args')
    @patch('os.path.exists')
    @patch('sys.exit')
    def test_main_pipeline_exception(self, mock_exit, mock_exists, mock_parse_args, mock_pipeline):
        """Test main when the pipeline raises an exception."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.query_gene_set = "query.txt"
        mock_args.background_gene_list = "background.txt"
        mock_args.output_dir = "output"
        mock_args.temp_dir = None
        mock_args.config = None
        mock_args.no_report = False
        mock_args.report_title = None
        
        mock_parse_args.return_value = mock_args
        mock_exists.return_value = True
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.side_effect = Exception("Pipeline error")
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Run the function
        main()
        
        # Check that sys.exit was called with an error code
        mock_exit.assert_called_once_with(1)
    
    @patch('geneinsight.cli.main')
    def test_cli_main(self, mock_main):
        """Test the cli_main entry point."""
        cli_main()
        mock_main.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])