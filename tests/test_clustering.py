import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer
from geneinsight.analysis.clustering import run_clustering, get_embedding_model

# Dummy encode function to bypass the heavy model and return predictable embeddings.
def dummy_encode(self, terms):
    # Each term gets a simple 2D embedding [i, i] where i is the index.
    return np.array([[i, i] for i in range(len(terms))])

# Patch SentenceTransformer so that __init__ does nothing and encode uses our dummy function.
@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
    monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

def test_missing_term_column(tmp_path, capsys):
    # Create a CSV that lacks the required "Term" column.
    data = {'NotTerm': ['a', 'b', 'c']}
    input_csv = tmp_path / "input_missing_term.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)

    # The function now handles the error internally and returns early
    # instead of raising a KeyError
    run_clustering(str(input_csv), str(output_csv), min_clusters=1, max_clusters=1, n_trials=1)
    # The function should return without creating output since it can't find 'Term' column

def test_single_term(tmp_path, capsys):
    # Create CSV with a single term.
    data = {'Term': ['only_term']}
    input_csv = tmp_path / "input_single.csv"
    output_csv = tmp_path / "output_single.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    # For one term, clustering with 1 cluster should work.
    run_clustering(str(input_csv), str(output_csv), min_clusters=1, max_clusters=1, n_trials=1)
    
    captured = capsys.readouterr().out
    assert "Optimal clustering algorithm:" in captured
    # Read the output CSV and verify it has one row and cluster label 0.
    output_df = pd.read_csv(output_csv)
    assert len(output_df) == 1
    assert output_df['Cluster'].iloc[0] == 0

def test_cluster_sorting(tmp_path):
    # Create a CSV with several terms.
    data = {'Term': [f"term{i}" for i in range(6)]}
    input_csv = tmp_path / "input_sorting.csv"
    output_csv = tmp_path / "output_sorting.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    # Force clustering into 2 clusters.
    run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=2, n_trials=1)
    
    output_df = pd.read_csv(output_csv)
    # Check that the output is sorted by the "Cluster" column (non-decreasing order).
    clusters = output_df['Cluster'].tolist()
    assert clusters == sorted(clusters)

def test_n_trials_effect(tmp_path):
    # Test with a higher number of trials to simulate extended optimization.
    data = {'Term': [f"term{i}" for i in range(10)]}
    input_csv = tmp_path / "input_trials.csv"
    output_csv = tmp_path / "output_trials.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=3, n_trials=5)
    
    output_df = pd.read_csv(output_csv)
    assert 'Cluster' in output_df.columns
    assert len(output_df) == 10

def test_invalid_input_file(tmp_path):
    # Test that a non-existent input file is handled gracefully.
    # The function now catches the error and returns early instead of raising.
    input_csv = tmp_path / "non_existent.csv"
    output_csv = tmp_path / "output_invalid.csv"
    # The function should return without error (it logs the error internally)
    run_clustering(str(input_csv), str(output_csv), min_clusters=1, max_clusters=1, n_trials=1)
    # Output file should not exist since input file was not found
    assert not output_csv.exists(), "Output file should not be created when input file is missing"

def test_two_terms(tmp_path, capsys):
    # Create CSV with exactly 2 terms.
    data = {'Term': ['term1', 'term2']}
    input_csv = tmp_path / "input_two_terms.csv"
    output_csv = tmp_path / "output_two_terms.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    # With 2 samples, the function should use the "too few samples" branch.
    run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=2, n_trials=1)
    
    captured = capsys.readouterr().out
    assert "Optimal clustering algorithm: N/A (too few samples)" in captured
    output_df = pd.read_csv(output_csv)
    assert len(output_df) == 2
    # Both rows should be assigned to cluster 0.
    assert all(output_df['Cluster'] == 0)

def test_inconsistent_cluster_range(tmp_path, capsys):
    # Create CSV with 4 terms.
    data = {'Term': [f"term{i}" for i in range(4)]}
    input_csv = tmp_path / "input_inconsistent.csv"
    output_csv = tmp_path / "output_inconsistent.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    # Set min_clusters > max_clusters for a dataset with >= 3 samples.
    run_clustering(str(input_csv), str(output_csv), min_clusters=5, max_clusters=4, n_trials=1)
    
    captured = capsys.readouterr().out
    # The code should force clustering into 2 clusters.
    assert "Optimal number of clusters: 2" in captured
    output_df = pd.read_csv(output_csv)
    # Verify that the resulting clusters are either 0 or 1.
    assert set(output_df['Cluster'].unique()).issubset({0, 1})

def test_extra_columns_preserved(tmp_path):
    # Create CSV with extra columns alongside "Term".
    data = {'Term': ['term1', 'term2', 'term3'], 'Extra': ['A', 'B', 'C']}
    input_csv = tmp_path / "input_extra.csv"
    output_csv = tmp_path / "output_extra.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=2, n_trials=1)
    
    output_df = pd.read_csv(output_csv)
    # Check that the extra column "Extra" is preserved.
    assert 'Extra' in output_df.columns
    # Ensure the "Cluster" column exists and there are 3 rows.
    assert 'Cluster' in output_df.columns
    assert len(output_df) == 3

def test_empty_csv(tmp_path, capsys):
    # Create an empty CSV file with the required "Term" column.
    df_empty = pd.DataFrame(columns=['Term'])
    input_csv = tmp_path / "input_empty.csv"
    output_csv = tmp_path / "output_empty.csv"
    df_empty.to_csv(input_csv, index=False)

    run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=2, n_trials=1)

    captured = capsys.readouterr().out
    output_df = pd.read_csv(output_csv)
    # The output CSV should be empty.
    assert output_df.empty
    # The printed output should mention the "too few samples" condition.
    assert "Optimal clustering algorithm: N/A (too few samples)" in captured


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

class TestGetEmbeddingModel:
    """Tests for the get_embedding_model function."""

    def test_get_embedding_model_path_not_found(self, monkeypatch):
        """Test fallback when model path doesn't exist."""
        # Mock importlib.resources.files to return a mock path object
        mock_files = MagicMock()
        mock_files.return_value.joinpath.return_value = "/nonexistent/path"
        monkeypatch.setattr(
            "geneinsight.analysis.clustering.resources.files",
            mock_files
        )
        # Mock os.path.exists to return False
        monkeypatch.setattr("os.path.exists", lambda path: False)
        # Mock SentenceTransformer to avoid actual model loading
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        # Function should return without error (fallback to online model)
        assert model is not None

    def test_get_embedding_model_exception(self, monkeypatch):
        """Test fallback when model loading raises exception."""
        # Mock importlib.resources.files to raise an exception
        mock_files = MagicMock(side_effect=Exception("Test error"))
        monkeypatch.setattr(
            "geneinsight.analysis.clustering.resources.files",
            mock_files
        )
        # Mock SentenceTransformer to avoid actual model loading
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        # Function should return without error (fallback to online model)
        assert model is not None


class TestRunClusteringExtended:
    """Extended tests for run_clustering function."""

    def test_run_clustering_external_model(self, tmp_path, monkeypatch):
        """Test clustering with use_local_model=False."""
        # Patch SentenceTransformer
        monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
        monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

        data = {'Term': [f"term{i}" for i in range(5)]}
        input_csv = tmp_path / "input_external.csv"
        output_csv = tmp_path / "output_external.csv"
        pd.DataFrame(data).to_csv(input_csv, index=False)

        # Call with use_local_model=False to hit the external model path
        run_clustering(
            str(input_csv),
            str(output_csv),
            min_clusters=2,
            max_clusters=2,
            n_trials=1,
            use_local_model=False
        )

        output_df = pd.read_csv(output_csv)
        assert 'Cluster' in output_df.columns
        assert len(output_df) == 5

    def test_run_clustering_output_save_error(self, tmp_path, monkeypatch, capsys):
        """Test handling of output file save error."""
        monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
        monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

        data = {'Term': [f"term{i}" for i in range(5)]}
        input_csv = tmp_path / "input_save_error.csv"
        output_csv = tmp_path / "output_save_error.csv"
        pd.DataFrame(data).to_csv(input_csv, index=False)

        # Mock to_csv to raise an exception
        with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Permission denied")):
            # The function should handle the error gracefully (logs error but doesn't raise)
            run_clustering(
                str(input_csv),
                str(output_csv),
                min_clusters=2,
                max_clusters=2,
                n_trials=1
            )
        # Function should complete without raising exception

    def test_run_clustering_embedding_error(self, tmp_path, monkeypatch):
        """Test handling of embedding generation error."""
        monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
        # Make encode raise an exception
        monkeypatch.setattr(
            SentenceTransformer, 'encode',
            MagicMock(side_effect=Exception("Embedding error"))
        )

        data = {'Term': ['term1', 'term2', 'term3']}
        input_csv = tmp_path / "input_embed_error.csv"
        output_csv = tmp_path / "output_embed_error.csv"
        pd.DataFrame(data).to_csv(input_csv, index=False)

        # The function should handle the error gracefully
        run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=2, n_trials=1)
        # Output file should not exist since embedding failed
        assert not output_csv.exists()

    def test_run_clustering_spectral_algorithm(self, tmp_path, monkeypatch):
        """Test that spectral clustering algorithm can be selected."""
        monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
        monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

        # Mock optuna to force spectral clustering
        mock_study = MagicMock()
        mock_study.best_params = {'clustering_algorithm': 'spectral', 'n_clusters': 2}

        with patch('optuna.create_study', return_value=mock_study):
            data = {'Term': [f"term{i}" for i in range(6)]}
            input_csv = tmp_path / "input_spectral.csv"
            output_csv = tmp_path / "output_spectral.csv"
            pd.DataFrame(data).to_csv(input_csv, index=False)

            run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=3, n_trials=1)

            output_df = pd.read_csv(output_csv)
            assert 'Cluster' in output_df.columns

    def test_run_clustering_kmeans_algorithm(self, tmp_path, monkeypatch):
        """Test that kmeans clustering algorithm can be selected."""
        monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
        monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

        # Mock optuna to force kmeans clustering
        mock_study = MagicMock()
        mock_study.best_params = {'clustering_algorithm': 'kmeans', 'n_clusters': 2}

        with patch('optuna.create_study', return_value=mock_study):
            data = {'Term': [f"term{i}" for i in range(6)]}
            input_csv = tmp_path / "input_kmeans.csv"
            output_csv = tmp_path / "output_kmeans.csv"
            pd.DataFrame(data).to_csv(input_csv, index=False)

            run_clustering(str(input_csv), str(output_csv), min_clusters=2, max_clusters=3, n_trials=1)

            output_df = pd.read_csv(output_csv)
            assert 'Cluster' in output_df.columns
