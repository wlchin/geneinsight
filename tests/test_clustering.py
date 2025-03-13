import os
import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from geneinsight.analysis.clustering import run_clustering

# Dummy encode function to bypass the heavy model and return predictable embeddings.
def dummy_encode(self, terms):
    # Each term gets a simple 2D embedding [i, i] where i is the index.
    return np.array([[i, i] for i in range(len(terms))])

# Patch SentenceTransformer so that __init__ does nothing and encode uses our dummy function.
@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(SentenceTransformer, '__init__', lambda self, model_name: None)
    monkeypatch.setattr(SentenceTransformer, 'encode', dummy_encode)

def test_missing_term_column(tmp_path):
    # Create a CSV that lacks the required "Term" column.
    data = {'NotTerm': ['a', 'b', 'c']}
    input_csv = tmp_path / "input_missing_term.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    
    # Expect a KeyError since "Term" is missing.
    with pytest.raises(KeyError):
        run_clustering(str(input_csv), str(output_csv), min_clusters=1, max_clusters=1, n_trials=1)

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
    # Test that a non-existent input file raises a FileNotFoundError.
    input_csv = tmp_path / "non_existent.csv"
    output_csv = tmp_path / "output_invalid.csv"
    with pytest.raises(FileNotFoundError):
        run_clustering(str(input_csv), str(output_csv), min_clusters=1, max_clusters=1, n_trials=1)

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
