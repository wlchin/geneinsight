import pytest
import pandas as pd
import torch
import json
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from sentence_transformers import util

# Import the module to be tested
# Assuming the module is in a package called geneinsight.analysis
from geneinsight.analysis.summary import RAGModule, read_input_files, initialize_rag_module, get_topics_of_interest, generate_results, save_results, main, create_summary


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "term": ["GO:0001", "GO:0002", "GO:0003", "GO:0004", "GO:0005"],
        "description": ["Process 1", "Process 2", "Process 3", "Process 4", "Process 5"],
        "inputGenes": ["GENE1,GENE2", "GENE2,GENE3", "GENE3,GENE4", "GENE4,GENE5", "GENE1,GENE5"]
    })


@pytest.fixture
def topics_df():
    """Create a sample topics DataFrame for testing."""
    return pd.DataFrame({
        "prompt_type": ["subtopic_BERT", "other", "subtopic_BERT", "subtopic_BERT"],
        "generated_result": ["Topic 1", "Topic X", "Topic 2", "Topic 3"]
    })


@pytest.fixture
def mock_embedder():
    """Create a mock SentenceTransformer object."""
    embedder = MagicMock()
    embedder.encode.side_effect = lambda texts, convert_to_tensor=False, **kwargs: torch.rand(
        (len(texts) if isinstance(texts, list) else 1, 10)
    )
    return embedder


@pytest.fixture
def rag_module(sample_df, mock_embedder):
    """Create a RAGModule instance with mocked embedder for testing."""
    with patch('geneinsight.analysis.summary.SentenceTransformer', return_value=mock_embedder):
        module = RAGModule(sample_df)
        # Replace the real embeddings with mock ones
        module.document_embeddings = torch.rand((5, 10))
        return module


class TestRAGModule:
    
    def test_init(self, sample_df):
        """Test RAGModule initialization."""
        with patch('geneinsight.analysis.summary.SentenceTransformer') as mock_transformer:
            mock_transformer.return_value.encode.return_value = torch.rand((5, 10))
            
            # Test with no provided embeddings
            module = RAGModule(sample_df)
            assert len(module.documents) == 5
            assert module.df_metadata.equals(sample_df)
            assert mock_transformer.return_value.encode.called
            
            # Test with provided embeddings
            embeddings = torch.rand((5, 10))
            module = RAGModule(sample_df, embeddings=embeddings)
            assert torch.equal(module.document_embeddings, embeddings)

    def test_get_top_documents(self, rag_module):
        """Test get_top_documents method."""
        with patch('geneinsight.analysis.summary.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.5, 0.7, 0.3, 0.9, 0.1]])
            
            # Test with N=3
            result = rag_module.get_top_documents("test query", N=3)
            assert len(result) == 3
            # The indices should be sorted by cosine similarity (highest first)
            # In this case: 0.9 (index 3), 0.7 (index 1), 0.5 (index 0)
            assert result.tolist() == [3, 1, 0]
            
            # Test with default N=5
            mock_cos_sim.return_value = torch.tensor([[0.5, 0.7, 0.3, 0.9, 0.1]])
            result = rag_module.get_top_documents("test query")
            assert len(result) == 5
            assert result.tolist() == [3, 1, 0, 2, 4]

    def test_get_context(self, rag_module):
        """Test get_context method."""
        # Mock the top indices
        indices = torch.tensor([2, 0, 4])
        
        context = rag_module.get_context("test query", indices)
        assert "Title: test query" in context
        assert "Document 1: Process 3" in context
        assert "Document 2: Process 1" in context
        assert "Document 3: Process 5" in context

    def test_get_text_response(self, rag_module):
        """Test get_text_response method."""
        with patch.object(rag_module, 'get_top_documents') as mock_get_top:
            mock_get_top.return_value = torch.tensor([2, 0, 4])
            
            response_dict, response_text = rag_module.get_text_response("test query", num_results=3)
            assert response_dict["query"] == "test query"
            assert "Document 1: Process 3" in response_dict["context"]
            assert "Document 2: Process 1" in response_dict["context"]
            assert "Document 3: Process 5" in response_dict["context"]
            assert response_dict["text"] == ""
            assert "rouge" in response_dict
            assert response_text == ""

    def test_format_references_and_genes(self, rag_module):
        """Test format_references_and_genes method."""
        indices = torch.tensor([0, 1, 2])
        
        references, unique_genes = rag_module.format_references_and_genes(indices)
        
        # Check references
        assert len(references) == 3
        assert references[0]["term"] == "GO:0001"
        assert references[0]["description"] == "Process 1"
        assert references[0]["genes"] == ["GENE1", "GENE2"]
        
        # Check unique genes
        assert len(unique_genes) == 4
        assert unique_genes["GENE1"] == 1
        assert unique_genes["GENE2"] == 2
        assert unique_genes["GENE3"] == 2
        assert unique_genes["GENE4"] == 1

    def test_get_summary_to_query(self, rag_module):
        """Test get_summary_to_query method."""
        with patch.object(rag_module, 'get_text_response') as mock_get_text:
            mock_get_text.return_value = ({"text": "Sample response", "context": "Sample context", "rouge": {}}, "")
            with patch.object(rag_module, 'get_top_documents') as mock_get_top:
                mock_get_top.return_value = torch.tensor([0, 1])
                with patch.object(rag_module, 'format_references_and_genes') as mock_format:
                    mock_format.return_value = (
                        [{"term": "GO:0001", "description": "Process 1", "genes": ["GENE1", "GENE2"]}],
                        {"GENE1": 1, "GENE2": 1}
                    )
                    
                    output = rag_module.get_summary_to_query("test query", num_results=2)
                    assert output["response"] == "Sample response"
                    assert output["context"] == "Sample context"
                    assert len(output["references"]) == 1
                    assert output["references"][0]["term"] == "GO:0001"
                    assert output["unique_genes"]["GENE1"] == 1
                    assert output["unique_genes"]["GENE2"] == 1

    def test_get_summary_to_query_df(self, rag_module):
        """Test get_summary_to_query_df method."""
        with patch.object(rag_module, 'get_summary_to_query') as mock_get_summary:
            mock_get_summary.return_value = {
                "response": "Sample response",
                "context": "Sample context",
                "rouge": {
                    "rouge1": {"fmeasure": 0.8},
                    "rouge2": {"fmeasure": 0.7},
                    "rougeL": {"fmeasure": 0.75}
                },
                "references": [
                    {"term": "GO:0001", "description": "Process 1", "genes": ["GENE1", "GENE2"]}
                ],
                "unique_genes": {"GENE1": 1, "GENE2": 1}
            }
            
            df = rag_module.get_summary_to_query_df("test query", num_results=2)
            assert df.shape[0] == 1
            assert df.iloc[0]["query"] == "test query"
            assert df.iloc[0]["response"] == "Sample response"
            assert df.iloc[0]["reference_term"] == "GO:0001"
            assert df.iloc[0]["reference_description"] == "Process 1"
            assert df.iloc[0]["reference_genes"] == "GENE1, GENE2"
            assert json.loads(df.iloc[0]["unique_genes"]) == {"GENE1": 1, "GENE2": 1}
            assert df.iloc[0]["rouge1_fmeasure"] == 0.8
            assert df.iloc[0]["rouge2_fmeasure"] == 0.7
            assert df.iloc[0]["rougeL_fmeasure"] == 0.75
            
            # Test with no references
            mock_get_summary.return_value = {
                "response": "Sample response",
                "context": "Sample context",
                "rouge": {},
                "references": [],
                "unique_genes": {}
            }
            
            df = rag_module.get_summary_to_query_df("test query")
            assert df.shape[0] == 1
            assert df.iloc[0]["reference_term"] is None
            assert df.iloc[0]["reference_description"] is None
            assert df.iloc[0]["reference_genes"] is None

    def test_save_output_to_json(self, rag_module, tmp_path):
        """Test save_output_to_json method."""
        output = {
            "response": "Sample response",
            "context": "Sample context",
            "references": [{"term": "GO:0001", "description": "Process 1", "genes": ["GENE1", "GENE2"]}],
            "unique_genes": {"GENE1": 1, "GENE2": 1}
        }
        
        filename = tmp_path / "test_output.json"
        rag_module.save_output_to_json(output, filename)
        
        # Verify the file was created and contains the expected content
        assert filename.exists()
        with open(filename, "r") as f:
            saved_output = json.load(f)
            assert saved_output == output

    def test_to_markdown(self, rag_module):
        """Test to_markdown method."""
        output = {
            "response": "Sample response",
            "rouge": {
                "rouge1": {"precision": 0.8, "recall": 0.7, "fmeasure": 0.75}
            },
            "context": "Sample context",
            "references": [{"term": "GO:0001", "description": "Process 1", "genes": ["GENE1", "GENE2"]}],
            "unique_genes": {"GENE1": 1, "GENE2": 1}
        }
        
        markdown = rag_module.to_markdown(output)
        assert "# Summary" in markdown
        assert "## Response" in markdown
        assert "Sample response" in markdown
        assert "## Rouge Scores" in markdown
        assert "- F-measure: 0.7500" in markdown
        assert "## Context" in markdown
        assert "Sample context" in markdown
        assert "## References" in markdown
        assert "**Term**: GO:0001" in markdown
        assert "Description: Process 1" in markdown
        assert "Genes: GENE1, GENE2" in markdown
        assert "## Unique Genes" in markdown
        assert "- GENE1: 1" in markdown
        assert "- GENE2: 1" in markdown


class TestUtilityFunctions:
    
    def test_read_input_files(self, sample_df, topics_df, tmp_path):
        """Test read_input_files function."""
        # Create temporary CSV files
        enrichment_path = tmp_path / "enrichment.csv"
        topics_path = tmp_path / "topics.csv"
        
        sample_df.to_csv(enrichment_path, index=False)
        topics_df.to_csv(topics_path, index=False)
        
        df, topics = read_input_files(str(enrichment_path), str(topics_path))
        
        assert df.shape == sample_df.shape
        assert topics.shape == topics_df.shape
        assert all(df.columns == sample_df.columns)
        assert all(topics.columns == topics_df.columns)

    def test_initialize_rag_module(self, sample_df):
        """Test initialize_rag_module function."""
        with patch('geneinsight.analysis.summary.RAGModule') as mock_rag:
            initialize_rag_module(sample_df)
            mock_rag.assert_called_once_with(sample_df, use_local_model=True)

    def test_get_topics_of_interest(self, topics_df):
        """Test get_topics_of_interest function."""
        result = get_topics_of_interest(topics_df)
        assert result == ["Topic 1", "Topic 2", "Topic 3"]
        assert "Topic X" not in result

    def test_generate_results(self, rag_module, topics_df):
        """Test generate_results function."""
        topics = get_topics_of_interest(topics_df)
        
        with patch.object(rag_module, 'get_summary_to_query_df') as mock_get_df:
            # Set up the mock to return a different DataFrame for each topic
            results = []
            for i, topic in enumerate(topics):
                df = pd.DataFrame({
                    "query": [topic],
                    "response": [f"Response {i}"],
                    "reference_term": [f"Term {i}"]
                })
                results.append(df)
                
            mock_get_df.side_effect = results
            
            final_df = generate_results(rag_module, topics, num_results=3)
            
            # Check that get_summary_to_query_df was called for each topic
            assert mock_get_df.call_count == len(topics)
            
            # Check that the results were concatenated correctly
            assert final_df.shape[0] == len(topics)
            for i, topic in enumerate(topics):
                assert final_df.iloc[i]["query"] == topic
                assert final_df.iloc[i]["response"] == f"Response {i}"
                assert final_df.iloc[i]["reference_term"] == f"Term {i}"

    def test_save_results(self, tmp_path):
        """Test save_results function."""
        df = pd.DataFrame({
            "query": ["Topic 1", "Topic 2"],
            "response": ["Response 1", "Response 2"]
        })
        
        output_path = tmp_path / "results.csv"
        save_results(df, str(output_path))
        
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert saved_df.shape == df.shape
        assert all(saved_df.columns == df.columns)
        assert saved_df.iloc[0]["query"] == "Topic 1"
        assert saved_df.iloc[1]["response"] == "Response 2"

    def test_main(self, sample_df, topics_df, tmp_path):
        """Test main function."""
        # Create temporary files
        enrichment_path = tmp_path / "enrichment.csv"
        topics_path = tmp_path / "topics.csv"
        output_path = tmp_path / "output.csv"
        
        sample_df.to_csv(enrichment_path, index=False)
        topics_df.to_csv(topics_path, index=False)
        
        # Create mock args
        args = MagicMock()
        args.enrichment_csv = str(enrichment_path)
        args.minor_topics_csv = str(topics_path)
        args.output_csv = str(output_path)
        args.num_results = 3
        args.use_external_model = False  # Default to use local model
        
        # Mock the functions
        with patch('geneinsight.analysis.summary.read_input_files', return_value=(sample_df, topics_df)) as mock_read:
            with patch('geneinsight.analysis.summary.initialize_rag_module') as mock_init:
                mock_init.return_value = "mock_rag_module"
                with patch('geneinsight.analysis.summary.get_topics_of_interest', return_value=["Topic 1", "Topic 2"]) as mock_get_topics:
                    with patch('geneinsight.analysis.summary.generate_results') as mock_generate:
                        mock_generate.return_value = pd.DataFrame({"a": [1, 2]})
                        with patch('geneinsight.analysis.summary.save_results') as mock_save:
                            
                            main(args)

                            # Check that all functions were called with correct arguments
                            mock_read.assert_called_once_with(str(enrichment_path), str(topics_path))
                            mock_init.assert_called_once_with(sample_df, use_local_model=True)
                            mock_get_topics.assert_called_once_with(topics_df)
                            mock_generate.assert_called_once_with("mock_rag_module", ["Topic 1", "Topic 2"], num_results=3)
                            mock_save.assert_called_once()
                            assert mock_save.call_args[0][1] == str(output_path)


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

class TestGetEmbeddingModel:
    """Tests for the get_embedding_model function."""

    def test_get_embedding_model_path_not_found(self, monkeypatch):
        """Test fallback when model path doesn't exist."""
        from geneinsight.analysis.summary import get_embedding_model
        from sentence_transformers import SentenceTransformer

        mock_files = MagicMock()
        mock_files.return_value.joinpath.return_value = "/nonexistent/path"
        monkeypatch.setattr(
            "geneinsight.analysis.summary.resources.files",
            mock_files
        )
        monkeypatch.setattr("os.path.exists", lambda path: False)
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        assert model is not None

    def test_get_embedding_model_exception(self, monkeypatch):
        """Test fallback when model loading raises exception."""
        from geneinsight.analysis.summary import get_embedding_model
        from sentence_transformers import SentenceTransformer

        mock_files = MagicMock(side_effect=Exception("Test error"))
        monkeypatch.setattr(
            "geneinsight.analysis.summary.resources.files",
            mock_files
        )
        monkeypatch.setattr(
            SentenceTransformer, '__init__',
            lambda self, model_name: None
        )

        model = get_embedding_model()
        assert model is not None


class TestRAGModuleExtended:
    """Extended tests for RAGModule."""

    def test_rag_module_external_model(self, sample_df):
        """Test RAGModule with use_local_model=False."""
        with patch('geneinsight.analysis.summary.SentenceTransformer') as mock_transformer:
            mock_transformer.return_value.encode.return_value = torch.rand((5, 10))

            # Test with use_local_model=False
            module = RAGModule(sample_df, use_local_model=False)
            assert len(module.documents) == 5
            # Should have been called with the online model name
            mock_transformer.assert_called_with("sentence-transformers/all-MiniLM-L6-v2")

    def test_format_references_no_gene_column(self, sample_df):
        """Test format_references_and_genes when neither inputGenes nor preferred_names exists."""
        # Create a DataFrame without gene columns
        df_no_genes = pd.DataFrame({
            "term": ["GO:0001", "GO:0002", "GO:0003"],
            "description": ["Process 1", "Process 2", "Process 3"]
        })

        with patch('geneinsight.analysis.summary.SentenceTransformer') as mock_transformer:
            mock_transformer.return_value.encode.return_value = torch.rand((3, 10))

            module = RAGModule(df_no_genes)
            module.document_embeddings = torch.rand((3, 10))

            indices = torch.tensor([0, 1])
            references, unique_genes = module.format_references_and_genes(indices)

            # Should return empty lists when no gene column exists
            assert references == []
            assert unique_genes == {}

    def test_format_references_with_preferred_names(self, sample_df):
        """Test format_references_and_genes with preferred_names column."""
        df_preferred = pd.DataFrame({
            "term": ["GO:0001", "GO:0002"],
            "description": ["Process 1", "Process 2"],
            "preferred_names": ["GENE1,GENE2", "GENE2,GENE3"]
        })

        with patch('geneinsight.analysis.summary.SentenceTransformer') as mock_transformer:
            mock_transformer.return_value.encode.return_value = torch.rand((2, 10))

            module = RAGModule(df_preferred)
            module.document_embeddings = torch.rand((2, 10))

            indices = torch.tensor([0, 1])
            references, unique_genes = module.format_references_and_genes(indices)

            assert len(references) == 2
            assert "GENE1" in unique_genes
            assert "GENE2" in unique_genes


class TestCreateSummary:
    """Tests for the create_summary function."""

    def test_create_summary_basic(self, sample_df):
        """Test create_summary function with valid inputs."""
        api_results_df = pd.DataFrame({
            "query": ["Topic 1", "Topic 2"]
        })

        with patch('geneinsight.analysis.summary.RAGModule') as mock_rag:
            mock_instance = MagicMock()
            mock_instance.get_summary_to_query_df.return_value = pd.DataFrame({
                "query": ["Topic 1"],
                "response": ["Response 1"]
            })
            mock_rag.return_value = mock_instance

            result = create_summary(api_results_df, sample_df)

            assert mock_rag.called
            assert isinstance(result, pd.DataFrame)

    def test_create_summary_with_generated_result_column(self, sample_df):
        """Test create_summary when 'query' column is missing but 'generated_result' exists."""
        api_results_df = pd.DataFrame({
            "generated_result": ["Topic 1", "Topic 2"]
        })

        with patch('geneinsight.analysis.summary.RAGModule') as mock_rag:
            mock_instance = MagicMock()
            mock_instance.get_summary_to_query_df.return_value = pd.DataFrame({
                "query": ["Topic 1"],
                "response": ["Response 1"]
            })
            mock_rag.return_value = mock_instance

            result = create_summary(api_results_df, sample_df)

            assert mock_rag.called
            assert isinstance(result, pd.DataFrame)

    def test_create_summary_missing_columns_error(self, sample_df):
        """Test create_summary raises ValueError when required columns missing."""
        api_results_df = pd.DataFrame({
            "other_column": ["Value 1", "Value 2"]
        })

        with pytest.raises(ValueError, match="must contain a 'query' column"):
            create_summary(api_results_df, sample_df)

    def test_create_summary_with_output_file(self, sample_df, tmp_path):
        """Test create_summary with summary_output parameter."""
        api_results_df = pd.DataFrame({
            "query": ["Topic 1"]
        })

        output_path = tmp_path / "summary_output.csv"

        with patch('geneinsight.analysis.summary.RAGModule') as mock_rag:
            mock_instance = MagicMock()
            mock_instance.get_summary_to_query_df.return_value = pd.DataFrame({
                "query": ["Topic 1"],
                "response": ["Response 1"]
            })
            mock_rag.return_value = mock_instance

            result = create_summary(api_results_df, sample_df, summary_output=str(output_path))

            assert output_path.exists()

    def test_create_summary_external_model(self, sample_df):
        """Test create_summary with use_local_model=False."""
        api_results_df = pd.DataFrame({
            "query": ["Topic 1"]
        })

        with patch('geneinsight.analysis.summary.RAGModule') as mock_rag:
            mock_instance = MagicMock()
            mock_instance.get_summary_to_query_df.return_value = pd.DataFrame({
                "query": ["Topic 1"],
                "response": ["Response 1"]
            })
            mock_rag.return_value = mock_instance

            result = create_summary(api_results_df, sample_df, use_local_model=False)

            # Check that RAGModule was called with use_local_model=False
            mock_rag.assert_called_once_with(sample_df, use_local_model=False)