"""
Tests for the file_handlers module in the GeneInsight report package.
"""

import os
import shutil
import pandas as pd
import pytest
from pathlib import Path
from unittest import mock

# Import the module to be tested
from geneinsight.report.file_handlers import (
    create_folder_structure,
    copy_input_files,
    copy_logo,
    copy_scripts
)

# Constants for tests
TEST_GENE_SET = "test_geneset"
TEST_OUTPUT_DIR = "test_output"
TEST_INPUT_DIR = "test_input"

@pytest.fixture
def setup_test_dirs():
    """Create and clean up test directories."""
    # Create test directories
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_INPUT_DIR, exist_ok=True)
    
    yield
    
    # Clean up after tests
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    if os.path.exists(TEST_INPUT_DIR):
        shutil.rmtree(TEST_INPUT_DIR)

@pytest.fixture
def setup_test_files():
    """Create test input files."""
    # Create test files
    file_mapping = {
        "clustered.csv": pd.DataFrame({"Term": ["term1", "term2"], "Cluster": [1, 2]}),
        "summary.csv": pd.DataFrame({"Gene": ["gene1", "gene2"], "Score": [0.5, 0.7]}),
        "api_results.csv": pd.DataFrame({"Topic": ["topic1", "topic2"], "Hits": [10, 20]}),
        "topics.csv": pd.DataFrame({"Topic": ["topic1", "topic2"], "Genes": ["gene1", "gene2"]}),
        "enrichment.csv": pd.DataFrame({"Term": ["term1", "term2"], "P-value": [0.01, 0.02]}),
        "ontology_dict.csv": pd.DataFrame({"query": ["query1"], "ontology_dict": ["{'GO:1': 'GENE1'}"]}),
        "enriched.csv": pd.DataFrame({"Term": ["term1", "term2"], "Odds Ratio": [1.5, 2.0], "P-value": [0.01, 0.02], "Adjusted P-value": [0.01, 0.02], "Combined Score": [1.5, 2.0]}),
    }
    
    # Create input files
    for filename, df in file_mapping.items():
        file_path = os.path.join(TEST_INPUT_DIR, filename)
        df.to_csv(file_path, index=False)
    
    yield file_mapping
    
    # Cleanup happens in setup_test_dirs fixture

def test_create_folder_structure(setup_test_dirs):
    """Test creation of the folder structure."""
    base_path = create_folder_structure(TEST_OUTPUT_DIR, TEST_GENE_SET)
    
    # Check if base path is returned correctly
    assert base_path == Path(TEST_OUTPUT_DIR)
    
    # Check if required folders were created
    assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "results"))
    assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "logs"))
    
    # Check some nested folders
    assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "results", "summary"))
    assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "results", "heatmaps", TEST_GENE_SET))
    assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "results", "sphinx_builds", f"html_build_{TEST_GENE_SET}"))

def test_copy_input_files(setup_test_dirs, setup_test_files):
    """Test copying of input files."""
    # Create the folder structure first
    create_folder_structure(TEST_OUTPUT_DIR, TEST_GENE_SET)
    
    # Copy files
    copy_input_files(TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_GENE_SET)
    
    # Check if files were copied correctly
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/clustered_topics/{TEST_GENE_SET}_clustered_topics.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/summary/{TEST_GENE_SET}.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/minor_topics/{TEST_GENE_SET}_minor_topics.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/topics_for_genelists/{TEST_GENE_SET}_topic_model.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/enrichment_df/{TEST_GENE_SET}__enrichment.csv"))
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/ontology_dict/{TEST_GENE_SET}_ontology_dict.csv"))
    
    # Check if filtered geneset file was created
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, f"results/filtered_sets/{TEST_GENE_SET}_filtered_gene_sets.csv"))
    
    # Verify content of filtered geneset file
    filtered_df = pd.read_csv(os.path.join(TEST_OUTPUT_DIR, f"results/filtered_sets/{TEST_GENE_SET}_filtered_gene_sets.csv"))
    assert "Odds Ratio" in filtered_df.columns
    assert "Adjusted P-value" in filtered_df.columns
    assert "P-value" in filtered_df.columns
    assert "Combined Score" in filtered_df.columns

def test_copy_input_files_missing_file(setup_test_dirs):
    """Test handling of missing input files."""
    # Create the folder structure
    create_folder_structure(TEST_OUTPUT_DIR, TEST_GENE_SET)
    
    # Create empty input directory (no files)
    os.makedirs(TEST_INPUT_DIR, exist_ok=True)
    
    # Test should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        copy_input_files(TEST_INPUT_DIR, TEST_OUTPUT_DIR, TEST_GENE_SET)

@mock.patch("shutil.copy2")
def test_copy_logo_from_package(mock_copy2, setup_test_dirs):
    """Test copying logo from package."""
    # Create the data directory
    os.makedirs(os.path.join(TEST_OUTPUT_DIR, "data"), exist_ok=True)
    
    # Mock imports and existence checks
    with mock.patch("os.path.exists", return_value=True):
        # Import errors should trigger fallbacks, so we need to patch the import mechanism
        with mock.patch("importlib.import_module") as mock_import:
            # Create mock for geneinsight.report.assets
            mock_assets = mock.MagicMock()
            mock_assets.__file__ = "fake_path/assets/__init__.py"
            mock_import.return_value = mock_assets
            
            # Call the function
            logo_path = copy_logo(TEST_OUTPUT_DIR)
            
            # Check if copy2 was called
            mock_copy2.assert_called_once()
            # The first argument should end with logo.png or GeneInsight.png
            assert mock_copy2.call_args[0][0].endswith(("logo.png", "GeneInsight.png"))
            assert mock_copy2.call_args[0][1] == os.path.join(TEST_OUTPUT_DIR, "data", "GeneInsight.png")
            
            # Check if the returned path is correct
            assert logo_path == os.path.join(TEST_OUTPUT_DIR, "data", "GeneInsight.png")
            
            # Check if data directory was created
            assert os.path.isdir(os.path.join(TEST_OUTPUT_DIR, "data"))


@mock.patch("geneinsight.report.file_handlers.logging.info")
def test_copy_scripts(mock_logging_info, setup_test_dirs):
    """Test copy_scripts function (which is mostly a no-op)."""
    scripts_to_include = ["script1.py", "script2.py"]
    copy_scripts(TEST_OUTPUT_DIR, scripts_to_include)
    
    # Check if the logging message was called
    mock_logging_info.assert_called_once_with("Scripts are part of the package - no copying needed")