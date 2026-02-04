"""
Tests for the geneinsight.models.meta module.
"""

import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Import the module to test
from geneinsight.models.meta import (
    load_csv_data,
    initialize_bertopic,
    initialize_kmeans_topic_model,
    run_multiple_seed_topic_modeling,
    DEPS_AVAILABLE
)

# Mock data for testing
MOCK_TERMS = ["gene1 pathway", "gene2 regulation", "gene3 expression", "gene4 transcription", "gene5 signaling"]

@pytest.fixture
def mock_csv_file():
    """Create a mock CSV file content."""
    return "Term\n" + "\n".join(MOCK_TERMS)

@pytest.fixture
def mock_csv_file_alt_column():
    """Create a mock CSV file with alternative column name."""
    return "Description\n" + "\n".join(MOCK_TERMS)

@pytest.fixture
def mock_empty_csv_file():
    """Create a mock empty CSV file."""
    return "Term\n"

class TestLoadCSVData:
    
    def test_load_csv_data_success(self, mock_csv_file):
        """Test successfully loading terms from a CSV file."""
        with patch("builtins.open", mock_open(read_data=mock_csv_file)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame({"Term": MOCK_TERMS})
                mock_read_csv.return_value = mock_df
                
                result = load_csv_data("test.csv")
                
                assert result == MOCK_TERMS
                mock_read_csv.assert_called_once_with("test.csv")
    
    def test_load_csv_data_alt_column(self, mock_csv_file_alt_column):
        """Test loading terms from a CSV file with an alternative column name."""
        with patch("builtins.open", mock_open(read_data=mock_csv_file_alt_column)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame({"Description": MOCK_TERMS})
                mock_read_csv.return_value = mock_df
                
                result = load_csv_data("test.csv")
                
                assert result == MOCK_TERMS
                mock_read_csv.assert_called_once_with("test.csv")
    
    def test_load_csv_data_empty(self, mock_empty_csv_file):
        """Test loading from an empty CSV file."""
        with patch("builtins.open", mock_open(read_data=mock_empty_csv_file)):
            with patch("pandas.read_csv") as mock_read_csv:
                mock_df = pd.DataFrame({"Term": []})
                mock_read_csv.return_value = mock_df
                
                result = load_csv_data("test.csv")
                
                assert result == []
                mock_read_csv.assert_called_once_with("test.csv")
    
    def test_load_csv_data_exception(self):
        """Test handling an exception when loading CSV data."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = Exception("CSV error")
            
            result = load_csv_data("test.csv")
            
            assert result == []
            mock_read_csv.assert_called_once_with("test.csv")

@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
class TestBERTopicModel:
    
    @patch("geneinsight.models.meta.BERTopic")
    def test_initialize_bertopic_success(self, mock_bertopic):
        """Test successfully initializing a BERTopic model."""
        # Set up mock BERTopic behavior
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = (
            [0, 1, 2, 0, 1],  # topics
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]  # probabilities
        )
        mock_bertopic.return_value = mock_model
        
        # Call the function to test
        topic_model, topics, probs = initialize_bertopic(
            documents=MOCK_TERMS, 
            num_topics=3
        )
        
        # Assertions
        assert topic_model == mock_model
        assert topics == [0, 1, 2, 0, 1]
        assert len(probs) == 5
        mock_bertopic.assert_called_once_with(nr_topics=3)
        mock_model.fit_transform.assert_called_once_with(MOCK_TERMS)
    
    @patch("geneinsight.models.meta.BERTopic")
    def test_initialize_bertopic_with_embeddings(self, mock_bertopic):
        """Test initializing a BERTopic model with pre-computed embeddings."""
        # Set up mock BERTopic behavior
        mock_model = MagicMock()
        mock_model.fit_transform.return_value = (
            [0, 1, 2, 0, 1],  # topics
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]  # probabilities
        )
        mock_bertopic.return_value = mock_model
        
        # Mock embeddings
        mock_embeddings = np.random.rand(5, 384)  # 5 documents, 384-dimensional embeddings
        
        # Call the function to test
        topic_model, topics, probs = initialize_bertopic(
            documents=MOCK_TERMS, 
            num_topics=3,
            embeddings=mock_embeddings
        )
        
        # Assertions
        assert topic_model == mock_model
        assert topics == [0, 1, 2, 0, 1]
        assert len(probs) == 5
        mock_bertopic.assert_called_once_with(nr_topics=3)
        mock_model.fit_transform.assert_called_once_with(MOCK_TERMS, mock_embeddings)
    
    def test_initialize_bertopic_empty(self):
        """Test initializing a BERTopic model with no documents."""
        topic_model, topics, probs = initialize_bertopic(
            documents=[], 
            num_topics=3
        )
        
        # Assertions
        assert topic_model is None
        assert topics == []
        assert probs == []
    
    @patch("geneinsight.models.meta.BERTopic")
    def test_initialize_bertopic_exception(self, mock_bertopic):
        """Test handling an exception when initializing a BERTopic model."""
        # Set up mock BERTopic to raise an exception
        mock_bertopic.return_value = MagicMock()
        mock_bertopic.return_value.fit_transform.side_effect = Exception("BERTopic error")
        
        # Call the function to test
        topic_model, topics, probs = initialize_bertopic(
            documents=MOCK_TERMS, 
            num_topics=3
        )
        
        # Assertions
        assert topic_model is None
        assert topics == []
        assert probs == []
        mock_bertopic.assert_called_once_with(nr_topics=3)

@pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
class TestKMeansTopicModel:
    
    @patch("geneinsight.models.meta.BERTopic")
    @patch("geneinsight.models.meta.CountVectorizer")
    @patch("geneinsight.models.meta.KMeans")
    @patch("geneinsight.models.meta.PCA")
    def test_initialize_kmeans_topic_model_success(self, mock_pca, mock_kmeans, mock_vectorizer, mock_bertopic):
        """Test successfully initializing a KMeans topic model."""
        # Set up mocks
        mock_vectorizer_inst = MagicMock()
        mock_kmeans_inst = MagicMock()
        mock_pca_inst = MagicMock()
        mock_bertopic_inst = MagicMock()
        
        mock_vectorizer.return_value = mock_vectorizer_inst
        mock_kmeans.return_value = mock_kmeans_inst
        mock_pca.return_value = mock_pca_inst
        mock_bertopic.return_value = mock_bertopic_inst
        
        # Set up return value for fit_transform
        mock_bertopic_inst.fit_transform.return_value = (
            [0, 1, 2, 0, 1],  # topics
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]  # probabilities
        )
        
        # Call the function to test
        topic_model, topics, probs = initialize_kmeans_topic_model(
            documents=MOCK_TERMS, 
            num_topics=3,
            ncomp=2,
            seed_value=42
        )
        
        # Assertions
        assert topic_model == mock_bertopic_inst
        assert topics == [0, 1, 2, 0, 1]
        assert len(probs) == 5
        
        # Verify mock calls
        mock_vectorizer.assert_called_once_with(stop_words="english")
        mock_kmeans.assert_called_once_with(n_clusters=3, random_state=42)
        mock_pca.assert_called_once_with(n_components=2)
        mock_bertopic.assert_called_once_with(
            hdbscan_model=mock_kmeans_inst, 
            umap_model=mock_pca_inst, 
            vectorizer_model=mock_vectorizer_inst
        )
        mock_bertopic_inst.fit_transform.assert_called_once_with(MOCK_TERMS)
    
    @patch("geneinsight.models.meta.BERTopic")
    @patch("geneinsight.models.meta.CountVectorizer")
    @patch("geneinsight.models.meta.KMeans")
    @patch("geneinsight.models.meta.PCA")
    def test_initialize_kmeans_with_embeddings(self, mock_pca, mock_kmeans, mock_vectorizer, mock_bertopic):
        """Test initializing a KMeans topic model with pre-computed embeddings."""
        # Set up mocks
        mock_vectorizer_inst = MagicMock()
        mock_kmeans_inst = MagicMock()
        mock_pca_inst = MagicMock()
        mock_bertopic_inst = MagicMock()
        
        mock_vectorizer.return_value = mock_vectorizer_inst
        mock_kmeans.return_value = mock_kmeans_inst
        mock_pca.return_value = mock_pca_inst
        mock_bertopic.return_value = mock_bertopic_inst
        
        # Set up return value for fit_transform
        mock_bertopic_inst.fit_transform.return_value = (
            [0, 1, 2, 0, 1],  # topics
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]]  # probabilities
        )
        
        # Mock embeddings
        mock_embeddings = np.random.rand(5, 384)  # 5 documents, 384-dimensional embeddings
        
        # Call the function to test
        topic_model, topics, probs = initialize_kmeans_topic_model(
            documents=MOCK_TERMS, 
            num_topics=3,
            ncomp=2,
            seed_value=42,
            embeddings=mock_embeddings
        )
        
        # Assertions
        assert topic_model == mock_bertopic_inst
        assert topics == [0, 1, 2, 0, 1]
        assert len(probs) == 5
        
        # Verify mock calls
        mock_bertopic_inst.fit_transform.assert_called_once_with(MOCK_TERMS, mock_embeddings)
    
    def test_initialize_kmeans_topic_model_empty(self):
        """Test initializing a KMeans topic model with no documents."""
        topic_model, topics, probs = initialize_kmeans_topic_model(
            documents=[], 
            num_topics=3
        )
        
        # Assertions
        assert topic_model is None
        assert topics == []
        assert probs == []
    
    @patch("geneinsight.models.meta.BERTopic")
    @patch("geneinsight.models.meta.CountVectorizer")
    @patch("geneinsight.models.meta.KMeans")
    @patch("geneinsight.models.meta.PCA")
    def test_initialize_kmeans_topic_model_exception(self, mock_pca, mock_kmeans, mock_vectorizer, mock_bertopic):
        """Test handling an exception when initializing a KMeans topic model."""
        # Set up mocks
        mock_vectorizer_inst = MagicMock()
        mock_kmeans_inst = MagicMock()
        mock_pca_inst = MagicMock()
        mock_bertopic_inst = MagicMock()
        
        mock_vectorizer.return_value = mock_vectorizer_inst
        mock_kmeans.return_value = mock_kmeans_inst
        mock_pca.return_value = mock_pca_inst
        mock_bertopic.return_value = mock_bertopic_inst
        
        # Set up exception for fit_transform
        mock_bertopic_inst.fit_transform.side_effect = Exception("KMeans error")
        
        # Call the function to test
        topic_model, topics, probs = initialize_kmeans_topic_model(
            documents=MOCK_TERMS, 
            num_topics=3
        )
        
        # Assertions
        assert topic_model is None
        assert topics == []
        assert probs == []

class TestRunMultipleSeedTopicModeling:
    
    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_bertopic")
    @patch("geneinsight.models.meta.SentenceTransformer")
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    @patch("os.remove")
    def test_run_multiple_seed_topic_modeling_bertopic(
        self, mock_remove, mock_makedirs, mock_to_csv, mock_load_csv, mock_transformer, mock_initialize_bertopic
    ):
        """Test running multiple seed topic modeling with BERTopic."""
        # Set up mocks
        mock_load_csv.return_value = MOCK_TERMS
        
        # Mock SentenceTransformer
        mock_transformer_inst = MagicMock()
        mock_transformer_inst.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
        mock_transformer.return_value = mock_transformer_inst
        
        # Mock BERTopic models
        mock_topic_models = []
        mock_topics_list = []
        mock_probs_list = []
        mock_df_list = []
        
        for i in range(5):  # 5 samples
            mock_model = MagicMock()
            mock_topics = [i % 3, (i+1) % 3, (i+2) % 3, i % 3, (i+1) % 3]
            mock_probs = np.random.rand(5, 3).tolist()
            
            # Create mock document info dataframe
            mock_df = pd.DataFrame({
                "Document": MOCK_TERMS,
                "Topic": mock_topics,
                "Name": [f"Topic {t}" for t in mock_topics],
                "Probability": [0.7, 0.6, 0.8, 0.7, 0.6],
                "Representative_document": [True, False, True, False, True],
                "Top_n_words": ["word1, word2", "word3, word4", "word5, word6", "word1, word2", "word3, word4"]
            })
            
            mock_model.get_document_info.return_value = mock_df
            
            mock_topic_models.append(mock_model)
            mock_topics_list.append(mock_topics)
            mock_probs_list.append(mock_probs)
            mock_df_list.append(mock_df)
        
        # Set up initialize_bertopic to return different models for each call
        mock_initialize_bertopic.side_effect = [(m, t, p) for m, t, p in zip(mock_topic_models, mock_topics_list, mock_probs_list)]
        
        # Call the function to test
        result_df = run_multiple_seed_topic_modeling(
            input_file="input.csv",
            output_file="output/results.csv",
            method="bertopic",
            num_topics=3,
            n_samples=5
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 25  # 5 samples × 5 terms
        assert "seed" in result_df.columns
        assert set(result_df["seed"].unique()) == {0, 1, 2, 3, 4}  # 5 different seeds
        
        # Verify mock calls
        mock_load_csv.assert_called_once_with("input.csv")
        assert mock_initialize_bertopic.call_count == 5
        mock_makedirs.assert_called()
        mock_to_csv.assert_called()
        mock_remove.assert_called_once()
    
    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_kmeans_topic_model")
    @patch("geneinsight.models.meta.SentenceTransformer")
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    @patch("os.remove")
    def test_run_multiple_seed_topic_modeling_kmeans(
        self, mock_remove, mock_makedirs, mock_to_csv, mock_load_csv, mock_transformer, mock_initialize_kmeans
    ):
        """Test running multiple seed topic modeling with KMeans."""
        # Set up mocks
        mock_load_csv.return_value = MOCK_TERMS
        
        # Mock SentenceTransformer
        mock_transformer_inst = MagicMock()
        mock_transformer_inst.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
        mock_transformer.return_value = mock_transformer_inst
        
        # Mock KMeans models
        mock_topic_models = []
        mock_topics_list = []
        mock_probs_list = []
        mock_df_list = []
        
        for i in range(5):  # 5 samples
            mock_model = MagicMock()
            mock_topics = [i % 3, (i+1) % 3, (i+2) % 3, i % 3, (i+1) % 3]
            mock_probs = np.random.rand(5, 3).tolist()
            
            # Create mock document info dataframe
            mock_df = pd.DataFrame({
                "Document": MOCK_TERMS,
                "Topic": mock_topics,
                "Name": [f"Topic {t}" for t in mock_topics],
                "Probability": [0.7, 0.6, 0.8, 0.7, 0.6],
                "Representative_document": [True, False, True, False, True],
                "Top_n_words": ["word1, word2", "word3, word4", "word5, word6", "word1, word2", "word3, word4"]
            })
            
            mock_model.get_document_info.return_value = mock_df
            
            mock_topic_models.append(mock_model)
            mock_topics_list.append(mock_topics)
            mock_probs_list.append(mock_probs)
            mock_df_list.append(mock_df)
        
        # Set up initialize_kmeans_topic_model to return different models for each call
        mock_initialize_kmeans.side_effect = [(m, t, p) for m, t, p in zip(mock_topic_models, mock_topics_list, mock_probs_list)]
        
        # Call the function to test
        result_df = run_multiple_seed_topic_modeling(
            input_file="input.csv",
            output_file="output/results.csv",
            method="kmeans",
            num_topics=3,
            ncomp=2,
            n_samples=5
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 25  # 5 samples × 5 terms
        assert "seed" in result_df.columns
        assert set(result_df["seed"].unique()) == {0, 1, 2, 3, 4}  # 5 different seeds
        
        # Verify mock calls
        mock_load_csv.assert_called_once_with("input.csv")
        assert mock_initialize_kmeans.call_count == 5
        mock_makedirs.assert_called()
        mock_to_csv.assert_called()
        mock_remove.assert_called_once()
    
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    def test_run_multiple_seed_topic_modeling_no_deps(self, mock_makedirs, mock_to_csv, mock_load_csv):
        """Test running multiple seed topic modeling when dependencies are not available."""
        # Temporarily patch DEPS_AVAILABLE to False
        with patch("geneinsight.models.meta.DEPS_AVAILABLE", False):
            result_df = run_multiple_seed_topic_modeling(
                input_file="input.csv",
                output_file="output/results.csv",
                n_samples=5
            )
            
            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert "Dependencies not available" in result_df["Document"].values
            assert all(result_df["Topic"] == -1)
            
            # Verify mock calls
            mock_load_csv.assert_not_called()
            mock_makedirs.assert_called()
            mock_to_csv.assert_called_once()
    
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    def test_run_multiple_seed_topic_modeling_empty_input(self, mock_makedirs, mock_to_csv, mock_load_csv):
        """Test running multiple seed topic modeling with an empty input file."""
        # Setup mock to return empty list
        mock_load_csv.return_value = []
        
        result_df = run_multiple_seed_topic_modeling(
            input_file="empty.csv",
            output_file="output/results.csv",
            n_samples=5
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
        
        # Verify mock calls
        mock_load_csv.assert_called_once_with("empty.csv")
        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()
    
    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_bertopic")
    @patch("geneinsight.models.meta.get_embedding_model")
    @patch("geneinsight.models.meta.load_csv_data")
    def test_run_multiple_seed_topic_modeling_transformer_exception(
        self, mock_load_csv, mock_get_embedding_model, mock_initialize_bertopic
    ):
        """Test handling SentenceTransformer exception in multiple seed topic modeling."""
        # Setup mock to return terms
        mock_load_csv.return_value = MOCK_TERMS

        # Mock get_embedding_model to return a model that raises exception on encode
        mock_model_inst = MagicMock()
        mock_model_inst.encode.side_effect = Exception("Encoding error")
        mock_get_embedding_model.return_value = mock_model_inst

        # Create mock BERTopic model that works without embeddings
        mock_model = MagicMock()
        mock_topics = [0, 1, 2, 0, 1]
        mock_probs = np.random.rand(5, 3).tolist()

        # Mock document info
        mock_df = pd.DataFrame({
            "Document": MOCK_TERMS,
            "Topic": mock_topics,
            "Name": [f"Topic {t}" for t in mock_topics],
            "Probability": [0.7, 0.6, 0.8, 0.7, 0.6],
            "Representative_document": [True, False, True, False, True],
            "Top_n_words": ["word1, word2", "word3, word4", "word5, word6", "word1, word2", "word3, word4"]
        })

        mock_model.get_document_info.return_value = mock_df
        mock_initialize_bertopic.return_value = (mock_model, mock_topics, mock_probs)

        with patch("pandas.DataFrame.to_csv"), patch("os.makedirs"), patch("os.remove"):
            result_df = run_multiple_seed_topic_modeling(
                input_file="input.csv",
                output_file="output/results.csv",
                method="bertopic",
                num_topics=3,
                n_samples=5
            )

            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 25  # 5 samples × 5 terms

            # Verify that models were initialized without embeddings (due to encoding error)
            for call_args in mock_initialize_bertopic.call_args_list:
                kwargs = call_args[1]
                assert "embeddings" not in kwargs or kwargs["embeddings"] is None
    
    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_bertopic")
    @patch("geneinsight.models.meta.SentenceTransformer")
    @patch("geneinsight.models.meta.load_csv_data")
    def test_run_multiple_seed_topic_modeling_model_exception(
        self, mock_load_csv, mock_transformer, mock_initialize_bertopic
    ):
        """Test handling topic model exception in multiple seed topic modeling."""
        # Setup mock to return terms
        mock_load_csv.return_value = MOCK_TERMS

        # Mock SentenceTransformer
        mock_transformer_inst = MagicMock()
        mock_transformer_inst.encode.return_value = np.random.rand(5, 384)  # Mock embeddings
        mock_transformer.return_value = mock_transformer_inst

        # Mock model initialization to fail
        mock_initialize_bertopic.return_value = (None, [], [])

        with patch("pandas.DataFrame.to_csv"), patch("os.makedirs"), patch("os.remove"):
            result_df = run_multiple_seed_topic_modeling(
                input_file="input.csv",
                output_file="output/results.csv",
                method="bertopic",
                num_topics=3,
                n_samples=5
            )

            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 0  # No successful models


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

from geneinsight.models.meta import get_embedding_model, main


class TestGetEmbeddingModel:

    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.resources.files")
    @patch("geneinsight.models.meta.os.path.exists")
    @patch("geneinsight.models.meta.SentenceTransformer")
    def test_get_embedding_model_path_not_found(self, mock_transformer, mock_exists, mock_files):
        """Test fallback when embedding model path doesn't exist."""
        mock_files.return_value.joinpath.return_value = "/nonexistent/path"
        mock_exists.return_value = False  # Path doesn't exist

        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        result = get_embedding_model()

        # Should fall back to online model
        mock_transformer.assert_called_with("all-MiniLM-L6-v2")
        assert result is mock_model

    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.resources.files")
    @patch("geneinsight.models.meta.os.path.exists")
    @patch("geneinsight.models.meta.SentenceTransformer")
    def test_get_embedding_model_exception_fallback(self, mock_transformer, mock_exists, mock_files):
        """Test fallback when loading model raises exception."""
        mock_files.return_value.joinpath.return_value = "/some/path"
        mock_exists.return_value = True

        # First call raises exception, second call (fallback) succeeds
        mock_model = MagicMock()
        mock_transformer.side_effect = [Exception("Model load error"), mock_model]

        result = get_embedding_model()

        # Should fall back to online model after exception
        assert mock_transformer.call_count == 2
        assert result is mock_model

    def test_get_embedding_model_no_deps(self):
        """Test get_embedding_model when dependencies are not available."""
        with patch("geneinsight.models.meta.DEPS_AVAILABLE", False):
            result = get_embedding_model()
            assert result is None


class TestMainCLI:

    @patch("geneinsight.models.meta.run_multiple_seed_topic_modeling")
    def test_main_cli(self, mock_run):
        """Test the main CLI entry point."""
        import argparse

        mock_run.return_value = pd.DataFrame()

        # Simulate command-line arguments
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_args = argparse.Namespace(
                input_file="input.csv",
                output_file="output/results.csv",
                method="bertopic",
                num_topics=5,
                ncomp=2,
                seed_value=42,
                n_samples=10,
                use_external_model=False
            )
            mock_parse_args.return_value = mock_args

            main()

        mock_run.assert_called_once_with(
            input_file="input.csv",
            output_file="output/results.csv",
            method="bertopic",
            num_topics=5,
            ncomp=2,
            seed_value=42,
            n_samples=10,
            use_local_model=True  # opposite of use_external_model
        )

    @patch("geneinsight.models.meta.run_multiple_seed_topic_modeling")
    def test_main_cli_external_model(self, mock_run):
        """Test main CLI with external model flag."""
        import argparse

        mock_run.return_value = pd.DataFrame()

        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_args = argparse.Namespace(
                input_file="input.csv",
                output_file="output/results.csv",
                method="kmeans",
                num_topics=None,
                ncomp=3,
                seed_value=0,
                n_samples=5,
                use_external_model=True  # Use external model
            )
            mock_parse_args.return_value = mock_args

            main()

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["use_local_model"] is False


class TestRunMultipleSeedExtended:

    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_bertopic")
    @patch("geneinsight.models.meta.get_embedding_model")
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    @patch("os.remove")
    def test_run_multiple_seed_external_model(
        self, mock_remove, mock_makedirs, mock_to_csv, mock_load_csv,
        mock_get_model, mock_initialize_bertopic
    ):
        """Test running with external model (use_local_model=False)."""
        mock_load_csv.return_value = MOCK_TERMS

        # Mock external SentenceTransformer
        mock_model_inst = MagicMock()
        mock_model_inst.encode.return_value = np.random.rand(5, 384)

        # get_embedding_model shouldn't be called when use_local_model=False
        mock_get_model.return_value = mock_model_inst

        # Mock BERTopic model
        mock_model = MagicMock()
        mock_topics = [0, 1, 2, 0, 1]
        mock_probs = np.random.rand(5, 3).tolist()
        mock_df = pd.DataFrame({
            "Document": MOCK_TERMS,
            "Topic": mock_topics,
            "Name": [f"Topic {t}" for t in mock_topics],
            "Probability": [0.7, 0.6, 0.8, 0.7, 0.6],
            "Representative_document": [True, False, True, False, True],
            "Top_n_words": ["word1", "word2", "word3", "word4", "word5"]
        })
        mock_model.get_document_info.return_value = mock_df
        mock_initialize_bertopic.return_value = (mock_model, mock_topics, mock_probs)

        with patch("geneinsight.models.meta.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_model_inst

            result_df = run_multiple_seed_topic_modeling(
                input_file="input.csv",
                output_file="output/results.csv",
                method="bertopic",
                num_topics=3,
                n_samples=2,
                use_local_model=False  # Use external model
            )

        assert isinstance(result_df, pd.DataFrame)

    @pytest.mark.skipif(not DEPS_AVAILABLE, reason="Required dependencies not available")
    @patch("geneinsight.models.meta.initialize_bertopic")
    @patch("geneinsight.models.meta.get_embedding_model")
    @patch("geneinsight.models.meta.load_csv_data")
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    def test_run_multiple_seed_get_document_info_exception(
        self, mock_makedirs, mock_to_csv, mock_load_csv,
        mock_get_model, mock_initialize_bertopic
    ):
        """Test handling exception in get_document_info."""
        mock_load_csv.return_value = MOCK_TERMS

        mock_model_inst = MagicMock()
        mock_model_inst.encode.return_value = np.random.rand(5, 384)
        mock_get_model.return_value = mock_model_inst

        # Mock BERTopic model that raises exception on get_document_info
        mock_model = MagicMock()
        mock_model.get_document_info.side_effect = Exception("Document info error")
        mock_topics = [0, 1, 2, 0, 1]
        mock_probs = np.random.rand(5, 3).tolist()
        mock_initialize_bertopic.return_value = (mock_model, mock_topics, mock_probs)

        with patch("os.remove"):
            result_df = run_multiple_seed_topic_modeling(
                input_file="input.csv",
                output_file="output/results.csv",
                method="bertopic",
                num_topics=3,
                n_samples=2
            )

        # Should return empty DataFrame when all models fail
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0