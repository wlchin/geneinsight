#!/usr/bin/env python3
"""
Tests for the BERTopic module implementation using pytest
"""

import os
import tempfile
import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# Import the module to test
from geneinsight.models.bertopic import (
    load_csv_data,
    initialize_bertopic,
    initialize_kmeans_topic_model,
    initialize_model_and_fit_documents,
    run_topic_modeling_return_df,
    run_multiple_seed_topic_modeling,
    signal_handler,
    main
)


@pytest.fixture
def sample_data():
    """Create sample documents for testing"""
    return [
        "This is a document about genetics and DNA sequencing.",
        "Machine learning models can help analyze genetic data.",
        "Protein folding is a complex biological process.",
        "CRISPR technology allows for gene editing.",
        "Bioinformatics combines biology and computer science."
    ]


@pytest.fixture
def sample_csv(tmp_path, sample_data):
    """Create a sample CSV file for testing"""
    sample_df = pd.DataFrame({"description": sample_data})
    input_csv_path = tmp_path / "test_input.csv"
    sample_df.to_csv(input_csv_path, index=False)
    return str(input_csv_path)


@pytest.fixture
def output_csv(tmp_path):
    """Create an output CSV path for testing"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir / "test_output.csv")


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing"""
    return np.random.rand(5, 10)


def test_load_csv_data(sample_csv, sample_data):
    """Test loading data from CSV file"""
    documents = load_csv_data(sample_csv)
    assert len(documents) == 5
    assert documents == sample_data


def test_load_csv_data_with_missing_file():
    """Test loading data from a non-existent CSV file"""
    with pytest.raises(FileNotFoundError):
        load_csv_data("nonexistent_file.csv")


def test_load_csv_data_with_missing_column(tmp_path):
    """Test loading data from a CSV file with missing 'description' column"""
    # Create a CSV without a description column
    sample_df = pd.DataFrame({"text": ["sample text"]})
    csv_path = tmp_path / "no_description.csv"
    sample_df.to_csv(csv_path, index=False)
    
    with pytest.raises(KeyError):
        load_csv_data(str(csv_path))


@patch('geneinsight.models.bertopic.BERTopic')
def test_initialize_bertopic(mock_bertopic_class, sample_data, mock_embeddings):
    """Test initializing BERTopic model"""
    # Setup mock
    mock_bertopic_instance = mock_bertopic_class.return_value
    mock_bertopic_instance.fit_transform.return_value = ([0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test without embeddings
    model, topics, probs = initialize_bertopic(sample_data, num_topics=3)
    
    # Assert calls
    mock_bertopic_class.assert_called_once_with(nr_topics=3)
    mock_bertopic_instance.fit_transform.assert_called_once_with(sample_data)
    assert model == mock_bertopic_instance
    assert topics == [0, 1, 2, 0, 1]
    assert probs == [0.8, 0.7, 0.6, 0.9, 0.7]
    
    # Reset mocks
    mock_bertopic_class.reset_mock()
    mock_bertopic_instance.fit_transform.reset_mock()
    
    # Test with embeddings
    model, topics, probs = initialize_bertopic(sample_data, num_topics=3, embeddings=mock_embeddings)
    
    # Assert calls
    mock_bertopic_class.assert_called_once_with(nr_topics=3)
    mock_bertopic_instance.fit_transform.assert_called_once_with(sample_data, mock_embeddings)


@patch('geneinsight.models.bertopic.BERTopic')
def test_initialize_bertopic_with_default_topics(mock_bertopic_class, sample_data):
    """Test initializing BERTopic model with default number of topics"""
    # Setup mock
    mock_bertopic_instance = mock_bertopic_class.return_value
    mock_bertopic_instance.fit_transform.return_value = ([0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test with default num_topics (None)
    model, topics, probs = initialize_bertopic(sample_data)
    
    # Assert calls
    mock_bertopic_class.assert_called_once_with(nr_topics=None)
    mock_bertopic_instance.fit_transform.assert_called_once_with(sample_data)


@patch('geneinsight.models.bertopic.BERTopic')
@patch('geneinsight.models.bertopic.KMeans')
@patch('geneinsight.models.bertopic.PCA')
@patch('geneinsight.models.bertopic.CountVectorizer')
def test_initialize_kmeans_topic_model(
    mock_vectorizer_class, mock_pca_class, mock_kmeans_class, mock_bertopic_class,
    sample_data, mock_embeddings
):
    """Test initializing KMeans-based topic model"""
    # Setup mocks
    mock_vectorizer = mock_vectorizer_class.return_value
    mock_kmeans = mock_kmeans_class.return_value
    mock_pca = mock_pca_class.return_value
    mock_bertopic_instance = mock_bertopic_class.return_value
    mock_bertopic_instance.fit_transform.return_value = ([0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test without embeddings
    model, topics, probs = initialize_kmeans_topic_model(
        sample_data, num_topics=3, ncomp=2, seed_value=42
    )
    
    # Assert calls
    mock_vectorizer_class.assert_called_once_with(stop_words="english")
    mock_kmeans_class.assert_called_once_with(n_clusters=3, random_state=42)
    mock_pca_class.assert_called_once_with(n_components=2)
    mock_bertopic_class.assert_called_once_with(
        hdbscan_model=mock_kmeans,
        umap_model=mock_pca,
        vectorizer_model=mock_vectorizer
    )
    mock_bertopic_instance.fit_transform.assert_called_once_with(sample_data)
    
    # Reset mocks
    mock_bertopic_class.reset_mock()
    mock_bertopic_instance.fit_transform.reset_mock()
    
    # Test with embeddings
    model, topics, probs = initialize_kmeans_topic_model(
        sample_data, num_topics=3, ncomp=2, seed_value=42, embeddings=mock_embeddings
    )
    
    # Assert calls
    mock_bertopic_instance.fit_transform.assert_called_once_with(sample_data, mock_embeddings)


@patch('geneinsight.models.bertopic.BERTopic')
@patch('geneinsight.models.bertopic.KMeans')
@patch('geneinsight.models.bertopic.PCA')
@patch('geneinsight.models.bertopic.CountVectorizer')
def test_initialize_kmeans_topic_model_default_params(
    mock_vectorizer_class, mock_pca_class, mock_kmeans_class, mock_bertopic_class,
    sample_data
):
    """Test initializing KMeans-based topic model with default parameters"""
    # Setup mocks
    mock_vectorizer = mock_vectorizer_class.return_value
    mock_kmeans = mock_kmeans_class.return_value
    mock_pca = mock_pca_class.return_value
    mock_bertopic_instance = mock_bertopic_class.return_value
    mock_bertopic_instance.fit_transform.return_value = ([0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test with default parameters
    model, topics, probs = initialize_kmeans_topic_model(sample_data)
    
    # Assert calls with default parameters
    mock_kmeans_class.assert_called_once_with(n_clusters=10, random_state=0)
    mock_pca_class.assert_called_once_with(n_components=2)


def test_initialize_model_and_fit_documents_invalid_method(sample_data):
    """Test that initialize_model_and_fit_documents raises ValueError for invalid method"""
    with pytest.raises(ValueError, match="Unknown method: invalid. Supported methods are 'bertopic' and 'kmeans'."):
        initialize_model_and_fit_documents(sample_data, method="invalid")


@patch('geneinsight.models.bertopic.initialize_bertopic')
@patch('geneinsight.models.bertopic.initialize_kmeans_topic_model')
def test_initialize_model_and_fit_documents(
    mock_initialize_kmeans, mock_initialize_bertopic, sample_data, mock_embeddings
):
    """Test initialize_model_and_fit_documents with different methods"""
    # Setup mocks
    mock_topic_model = MagicMock()
    mock_initialize_bertopic.return_value = (mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    mock_initialize_kmeans.return_value = (mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test bertopic method
    model, topics, probs = initialize_model_and_fit_documents(
        sample_data, method="bertopic", num_topics=3, embeddings=mock_embeddings
    )
    mock_initialize_bertopic.assert_called_once_with(
        sample_data, num_topics=3, embeddings=mock_embeddings
    )
    assert model == mock_topic_model
    
    # Reset mocks
    mock_initialize_bertopic.reset_mock()
    
    # Test kmeans method
    model, topics, probs = initialize_model_and_fit_documents(
        sample_data, method="kmeans", num_topics=3, ncomp=2, seed_value=42, embeddings=mock_embeddings
    )
    mock_initialize_kmeans.assert_called_once_with(
        sample_data, num_topics=3, ncomp=2, seed_value=42, embeddings=mock_embeddings
    )
    assert model == mock_topic_model


@patch('geneinsight.models.bertopic.initialize_bertopic')
@patch('geneinsight.models.bertopic.initialize_kmeans_topic_model')
def test_initialize_model_and_fit_documents_default_params(
    mock_initialize_kmeans, mock_initialize_bertopic, sample_data
):
    """Test initialize_model_and_fit_documents with default parameters"""
    # Setup mocks
    mock_topic_model = MagicMock()
    mock_initialize_bertopic.return_value = (mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    mock_initialize_kmeans.return_value = (mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7])
    
    # Test with default parameters
    model, topics, probs = initialize_model_and_fit_documents(sample_data)
    
    # Default method should be "bertopic"
    mock_initialize_bertopic.assert_called_once_with(
        sample_data, num_topics=10, embeddings=None
    )


@patch('geneinsight.models.bertopic.initialize_model_and_fit_documents')
def test_run_topic_modeling_return_df(mock_initialize_model, sample_data):
    """Test run_topic_modeling_return_df function"""
    # Setup mock
    mock_topic_model = MagicMock()
    mock_initialize_model.return_value = (
        mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7]
    )
    
    # Create a sample document info DataFrame
    sample_doc_info = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    mock_topic_model.get_document_info.return_value = sample_doc_info
    
    # Test the function
    result_df = run_topic_modeling_return_df(
        sample_data, method="bertopic", num_topics=3, ncomp=2, seed_value=42
    )
    
    # Assert calls
    mock_initialize_model.assert_called_once_with(
        sample_data, method="bertopic", num_topics=3, ncomp=2, seed_value=42, embeddings=None
    )
    mock_topic_model.get_document_info.assert_called_once_with(sample_data)
    pd.testing.assert_frame_equal(result_df, sample_doc_info)


@patch('geneinsight.models.bertopic.initialize_model_and_fit_documents')
def test_run_topic_modeling_return_df_default_params(mock_initialize_model, sample_data):
    """Test run_topic_modeling_return_df function with default parameters"""
    # Setup mock
    mock_topic_model = MagicMock()
    mock_initialize_model.return_value = (
        mock_topic_model, [0, 1, 2, 0, 1], [0.8, 0.7, 0.6, 0.9, 0.7]
    )
    
    # Create a sample document info DataFrame
    sample_doc_info = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    mock_topic_model.get_document_info.return_value = sample_doc_info
    
    # Test with default parameters
    result_df = run_topic_modeling_return_df(sample_data)
    
    # Assert calls with default parameters
    mock_initialize_model.assert_called_once_with(
        sample_data, method="bertopic", num_topics=10, ncomp=2, seed_value=0, embeddings=None
    )


@patch('geneinsight.models.bertopic.SentenceTransformer')
@patch('geneinsight.models.bertopic.run_topic_modeling_return_df')
@patch('geneinsight.models.bertopic.load_csv_data')
def test_run_multiple_seed_topic_modeling(
    mock_load_csv_data, mock_run_topic_modeling, mock_sentence_transformer,
    sample_data, sample_csv, output_csv
):
    """Test run_multiple_seed_topic_modeling function"""
    # Setup mocks
    mock_load_csv_data.return_value = sample_data
    
    mock_encoder = mock_sentence_transformer.return_value
    mock_embeddings = np.random.rand(5, 10)
    mock_encoder.encode.return_value = mock_embeddings
    
    # Create sample document info DataFrames for each seed
    sample_doc_info_1 = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    
    sample_doc_info_2 = pd.DataFrame({
        'Document': sample_data,
        'Topic': [1, 0, 2, 1, 0],
        'Name': ['Topic 1', 'Topic 0', 'Topic 2', 'Topic 1', 'Topic 0'],
        'Probability': [0.9, 0.8, 0.7, 0.8, 0.9]
    })
    
    # Mock return values for the two seeds
    mock_run_topic_modeling.side_effect = [sample_doc_info_1, sample_doc_info_2]
    
    # Test with bertopic method and sentence embeddings
    result_df = run_multiple_seed_topic_modeling(
        sample_csv, output_csv, method="bertopic", num_topics=3, ncomp=2, 
        seed_value=0, n_samples=2, use_sentence_embeddings=True
    )
    
    # Assert calls
    mock_load_csv_data.assert_called_once_with(sample_csv)
    mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
    mock_encoder.encode.assert_called_once_with(sample_data, show_progress_bar=True)
    
    # Assert run_topic_modeling calls
    assert mock_run_topic_modeling.call_count == 2
    mock_run_topic_modeling.assert_any_call(
        sample_data, method="bertopic", num_topics=3, ncomp=2, seed_value=0, embeddings=mock_embeddings
    )
    mock_run_topic_modeling.assert_any_call(
        sample_data, method="bertopic", num_topics=3, ncomp=2, seed_value=10, embeddings=mock_embeddings
    )
    
    # Assert DataFrame structure
    assert len(result_df) == 10  # 5 documents x 2 seeds
    assert "seed" in result_df.columns
    assert set(result_df["seed"].unique()) == {0, 10}
    
    # Test file was saved
    assert os.path.exists(output_csv)


@patch('geneinsight.models.bertopic.run_topic_modeling_return_df')
@patch('geneinsight.models.bertopic.load_csv_data')
def test_run_multiple_seed_topic_modeling_kmeans(
    mock_load_csv_data, mock_run_topic_modeling,
    sample_data, sample_csv, output_csv
):
    """Test run_multiple_seed_topic_modeling with kmeans method"""
    # Setup mocks
    mock_load_csv_data.return_value = sample_data
    
    # Create sample document info DataFrames for each seed
    sample_doc_info = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    
    # Mock return values
    mock_run_topic_modeling.return_value = sample_doc_info
    
    # Test with kmeans method
    result_df = run_multiple_seed_topic_modeling(
        sample_csv, output_csv, method="kmeans", num_topics=3, ncomp=2, 
        seed_value=0, n_samples=1, use_sentence_embeddings=False
    )
    
    # Assert embeddings are not used for kmeans method
    mock_run_topic_modeling.assert_called_once_with(
        sample_data, method="kmeans", num_topics=3, ncomp=2, seed_value=0, embeddings=None
    )


@patch('geneinsight.models.bertopic.run_topic_modeling_return_df')
@patch('geneinsight.models.bertopic.load_csv_data')
def test_run_multiple_seed_topic_modeling_with_directory_creation(
    mock_load_csv_data, mock_run_topic_modeling,
    sample_data, sample_csv, tmp_path
):
    """Test run_multiple_seed_topic_modeling creates directory if it doesn't exist"""
    # Setup mocks
    mock_load_csv_data.return_value = sample_data
    
    # Create sample document info DataFrame
    sample_doc_info = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    
    # Mock return value
    mock_run_topic_modeling.return_value = sample_doc_info
    
    # Create a deep nested directory path that doesn't exist
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    output_csv = str(nested_dir / "test_output.csv")
    
    # Test function creates directories
    result_df = run_multiple_seed_topic_modeling(
        sample_csv, output_csv, method="bertopic", n_samples=1
    )
    
    # Assert directory was created
    assert os.path.exists(str(nested_dir))
    assert os.path.exists(output_csv)


@patch('geneinsight.models.bertopic.run_topic_modeling_return_df')
@patch('geneinsight.models.bertopic.load_csv_data')
def test_run_multiple_seed_topic_modeling_default_params(
    mock_load_csv_data, mock_run_topic_modeling,
    sample_data, sample_csv, output_csv
):
    """Test run_multiple_seed_topic_modeling with default parameters"""
    # Setup mocks
    mock_load_csv_data.return_value = sample_data
    mock_run_topic_modeling.return_value = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7]
    })
    
    # Test with default parameters but explicitly set use_sentence_embeddings=False
    # to ensure SentenceTransformer is not used
    result_df = run_multiple_seed_topic_modeling(
        sample_csv, output_csv, use_sentence_embeddings=False
    )
    
    # Assert run_topic_modeling was called with default parameters
    mock_run_topic_modeling.assert_called_once_with(
        sample_data, method="bertopic", num_topics=None, ncomp=2, seed_value=0, embeddings=None
    )
    
    # Assert run_topic_modeling was called with default parameters
    mock_run_topic_modeling.assert_called_once_with(
        sample_data, method="bertopic", num_topics=None, ncomp=2, seed_value=0, embeddings=None
    )


@patch('sys.exit')
def test_signal_handler(mock_exit):
    """Test signal handler function"""
    # Call signal handler
    signal_handler(None, None)
    
    # Assert sys.exit was called
    mock_exit.assert_called_once_with(0)


@patch('geneinsight.models.bertopic.run_multiple_seed_topic_modeling')
@patch('argparse.ArgumentParser.parse_args')
def test_main(mock_parse_args, mock_run_multiple_seed_topic_modeling):
    """Test main function"""
    # Setup mock for parse_args
    mock_args = MagicMock()
    mock_args.input_file = "input.csv"
    mock_args.output_file = "output.csv"
    mock_args.method = "bertopic"
    mock_args.num_topics = 5
    mock_args.ncomp = 3
    mock_args.seed_value = 42
    mock_args.n_samples = 2
    mock_args.use_sentence_embeddings = True
    mock_parse_args.return_value = mock_args
    
    # Call main function
    main()
    
    # Assert run_multiple_seed_topic_modeling was called with correct arguments
    mock_run_multiple_seed_topic_modeling.assert_called_once_with(
        input_file="input.csv",
        output_file="output.csv",
        method="bertopic",
        num_topics=5,
        ncomp=3,
        seed_value=42,
        n_samples=2,
        use_sentence_embeddings=True
    )


@patch('geneinsight.models.bertopic.run_multiple_seed_topic_modeling')
@patch('argparse.ArgumentParser.parse_args')
def test_main_default_args(mock_parse_args, mock_run_multiple_seed_topic_modeling):
    """Test main function with default arguments"""
    # Setup mock for parse_args with minimal required arguments
    mock_args = MagicMock()
    mock_args.input_file = "input.csv"
    mock_args.output_file = "output.csv"
    mock_args.method = "bertopic"
    mock_args.num_topics = None
    mock_args.ncomp = 2
    mock_args.seed_value = 0
    mock_args.n_samples = 1
    mock_args.use_sentence_embeddings = False
    mock_parse_args.return_value = mock_args
    
    # Call main function
    main()
    
    # Assert run_multiple_seed_topic_modeling was called with correct arguments
    mock_run_multiple_seed_topic_modeling.assert_called_once_with(
        input_file="input.csv",
        output_file="output.csv",
        method="bertopic",
        num_topics=None,
        ncomp=2,
        seed_value=0,
        n_samples=1,
        use_sentence_embeddings=False
    )


def test_signal_handler_registration():
    """Test that signal handler function exists and has the expected signature"""
    # In a test environment, we can't easily verify the signal handler registration
    # since it happens at module import time. Instead, we can verify that the
    # signal_handler function exists and has the correct signature.
    
    # Check that signal_handler is a function
    assert callable(signal_handler)
    
    # Check that signal_handler accepts two parameters
    import inspect
    sig = inspect.signature(signal_handler)
    assert len(sig.parameters) == 2


@patch('geneinsight.models.bertopic.SentenceTransformer')
@patch('geneinsight.models.bertopic.run_topic_modeling_return_df')
@patch('geneinsight.models.bertopic.load_csv_data')
def test_run_multiple_seed_topic_modeling_with_large_n_samples(
    mock_load_csv_data, mock_run_topic_modeling, mock_sentence_transformer,
    sample_data, sample_csv, output_csv
):
    """Test run_multiple_seed_topic_modeling with a large number of samples"""
    # Setup mocks
    mock_load_csv_data.return_value = sample_data
    
    # Mock SentenceTransformer and encode
    mock_encoder = mock_sentence_transformer.return_value
    mock_embeddings = np.random.rand(5, 10)
    mock_encoder.encode.return_value = mock_embeddings
    
    # Create sample document info DataFrame
    sample_doc_info = pd.DataFrame({
        'Document': sample_data,
        'Topic': [0, 1, 2, 0, 1],
        'Name': ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 0', 'Topic 1'],
        'Probability': [0.8, 0.7, 0.6, 0.9, 0.7],
        'seed': None  # Add seed column that will be filled by the function
    })
    
    # Mock return values for different seeds
    expected_seeds = [0, 10, 20, 30, 40]
    n_samples = len(expected_seeds)
    
    # Create a list of dataframes with different seed values
    def side_effect(*args, **kwargs):
        # Get the seed value from the kwargs
        seed_value = kwargs.get('seed_value', 0)
        # Create a copy of the sample dataframe with the seed value
        df_copy = sample_doc_info.copy()
        df_copy['seed'] = seed_value
        return df_copy
    
    mock_run_topic_modeling.side_effect = side_effect
    
    # Test with multiple samples
    result_df = run_multiple_seed_topic_modeling(
        sample_csv, output_csv, method="bertopic", num_topics=3, ncomp=2, 
        seed_value=0, n_samples=n_samples, use_sentence_embeddings=True
    )
    
    # Assert run_topic_modeling was called the correct number of times
    assert mock_run_topic_modeling.call_count == n_samples
    
    # Check that each seed value was used (starting with 0, incrementing by 10)
    expected_calls = [
        call(sample_data, method="bertopic", num_topics=3, ncomp=2, seed_value=seed, embeddings=mock_embeddings)
        for seed in expected_seeds
    ]
    mock_run_topic_modeling.assert_has_calls(expected_calls, any_order=True)
    
    # Assert DataFrame has the correct structure
    assert len(result_df) == 5 * n_samples  # 5 documents x n_samples
    assert "seed" in result_df.columns
    # Use a set to compare the unique values regardless of type (int, np.int64, etc.)
    assert {int(seed) for seed in result_df["seed"].unique()} == set(expected_seeds)