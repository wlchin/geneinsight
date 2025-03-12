import pytest
import pandas as pd
import torch
import json
import os
import sys
from unittest.mock import patch, MagicMock, mock_open

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the RAGModule class
# Assuming the module is named geneinsight.api.get_summary
from geneinsight.api.get_summary import RAGModule

# Global patches to avoid actual model loading
@pytest.fixture(autouse=True)
def mock_environment():
    with patch('sentence_transformers.SentenceTransformer'), \
         patch('geneinsight.api.get_summary.chat'), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "mock-api-key"}):
        yield

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "term": ["Gene A", "Gene B", "Gene C"],
        "description": ["Description of Gene A", "Description of Gene B", "Description of Gene C"],
        "inputGenes": ["BRCA1,TP53", "EGFR,KRAS", "PTEN,AKT1"]
    })

@pytest.fixture
def mock_embedder():
    """Create a mock SentenceTransformer."""
    mock = MagicMock()
    # Mock encode method to return tensor of correct shape
    mock.encode.return_value = torch.rand(3, 384)  # Assuming embedding size of 384
    return mock

@pytest.fixture
def rag_module(sample_df, monkeypatch):
    """Create a RAGModule instance with mocked embedder."""
    # Mock SentenceTransformer
    mock_sentence_transformer = MagicMock()
    mock_sentence_transformer.return_value = MagicMock()
    mock_sentence_transformer.return_value.encode.return_value = torch.rand(3, 384)
    monkeypatch.setattr("geneinsight.api.get_summary.SentenceTransformer", mock_sentence_transformer)
    
    # Create RAGModule with sample data
    module = RAGModule(sample_df)
    
    # Set document_embeddings directly
    module.document_embeddings = torch.rand(3, 384)
    
    return module

def test_initialization(sample_df, monkeypatch):
    """Test RAGModule initialization."""
    # Mock SentenceTransformer
    mock_transformer = MagicMock()
    mock_transformer.return_value.encode.return_value = torch.rand(3, 384)
    monkeypatch.setattr("geneinsight.api.get_summary.SentenceTransformer", mock_transformer)
    
    # Initialize with sample data
    rag = RAGModule(sample_df)
    
    # Check if documents are loaded correctly
    assert len(rag.documents) == 3
    assert rag.documents == sample_df["description"].to_list()
    
    # Check if metadata is stored
    assert rag.df_metadata.equals(sample_df)
    
    # Check if embedder is initialized
    assert rag.embedder is not None
    
    # Check if document_embeddings are created
    assert rag.document_embeddings is not None
    
    # Test initialization with provided embeddings
    custom_embeddings = torch.rand(3, 384)
    rag_with_embeddings = RAGModule(sample_df, embeddings=custom_embeddings)
    assert rag_with_embeddings.document_embeddings is custom_embeddings

def test_get_top_documents(rag_module, monkeypatch):
    """Test get_top_documents method."""
    # Mock cos_sim to return predictable values
    mock_cos_sim = MagicMock()
    mock_cos_sim.return_value = torch.tensor([[0.8, 0.6, 0.4]])
    monkeypatch.setattr("geneinsight.api.get_summary.util.pytorch_cos_sim", mock_cos_sim)
    
    # Test with N=2
    result = rag_module.get_top_documents("test query", N=2)
    
    # Should return indices [0, 1] (top 2 scores)
    assert len(result) == 2
    assert result[0] == 0  # Highest score
    assert result[1] == 1  # Second highest score
    
    # Test with different N
    result = rag_module.get_top_documents("test query", N=1)
    assert len(result) == 1
    assert result[0] == 0  # Only highest score

def test_get_context(rag_module):
    """Test get_context method."""
    # Create dummy indices
    indices = torch.tensor([0, 2])
    
    # Get context
    context = rag_module.get_context("test query", indices)
    
    # Check if context contains the query
    assert "Title: test query" in context
    
    # Check if context contains the selected documents
    assert "Document 1: Description of Gene A" in context
    assert "Document 2: Description of Gene C" in context
    
    # Make sure document 2 is not included
    assert "Description of Gene B" not in context

def test_get_chat_response(rag_module):
    """Test get_chat_response method."""
    # Mock chat response
    mock_response = MagicMock()
    mock_response.message.content = "This is a test response."
    
    with patch("geneinsight.api.get_summary.chat", return_value=mock_response):
        # Call the method
        response = rag_module.get_chat_response("Test context")
        
        # Check returned response
        assert response == "This is a test response."

def test_calculate_rouge(rag_module):
    """Test calculate_rouge method."""
    # This method is disabled in the code
    result = rag_module.calculate_rouge("reference", "hypothesis")
    assert result == {}

def test_get_text_response(rag_module):
    """Test get_text_response method."""
    # Mock dependencies
    with patch.object(rag_module, "get_top_documents", return_value=torch.tensor([0, 1])) as mock_top_docs, \
         patch.object(rag_module, "get_context", return_value="Test context") as mock_context, \
         patch.object(rag_module, "get_chat_response", return_value="Test response") as mock_chat_response:
        
        # Call the method
        response_dict, response_text = rag_module.get_text_response("test query", num_results=2)
        
        # Check method calls
        mock_top_docs.assert_called_once_with("test query", N=2)
        mock_context.assert_called_once()
        mock_chat_response.assert_called_once_with("Test context")
        
        # Check return values
        assert response_text == "Test response"
        assert response_dict["query"] == "test query"
        assert response_dict["context"] == "Test context"
        assert response_dict["text"] == "Test response"
        assert "rouge" in response_dict

def test_format_references_and_genes(rag_module):
    """Test format_references_and_genes method."""
    # Test with example indices
    indices = torch.tensor([0, 2])
    references, unique_genes = rag_module.format_references_and_genes(indices)
    
    # Check references
    assert len(references) == 2
    assert references[0]["term"] == "Gene A"
    assert references[0]["description"] == "Description of Gene A"
    assert set(references[0]["genes"]) == {"BRCA1", "TP53"}
    
    assert references[1]["term"] == "Gene C"
    assert references[1]["description"] == "Description of Gene C"
    assert set(references[1]["genes"]) == {"PTEN", "AKT1"}
    
    # Check unique genes
    assert len(unique_genes) == 4
    assert unique_genes["BRCA1"] == 1
    assert unique_genes["TP53"] == 1
    assert unique_genes["PTEN"] == 1
    assert unique_genes["AKT1"] == 1

def test_get_summary_to_query(rag_module):
    """Test get_summary_to_query method."""
    # Mock dependencies
    with patch.object(rag_module, "get_text_response", return_value=(
        {"text": "Test response", "rouge": {}, "context": "Test context"},
        "Test response"
    )) as mock_text_response, \
    patch.object(rag_module, "get_top_documents", return_value=torch.tensor([0, 1])) as mock_top_docs, \
    patch.object(rag_module, "format_references_and_genes", return_value=(
        [{"term": "Gene A", "description": "Desc A", "genes": ["BRCA1"]}],
        {"BRCA1": 1}
    )) as mock_format_refs:
        
        # Call the method
        output = rag_module.get_summary_to_query("test query")
        
        # Check method calls
        mock_text_response.assert_called_once_with("test query", num_results=5, calculate_rouge=True)
        mock_top_docs.assert_called_once_with("test query", N=5)
        mock_format_refs.assert_called_once()
        
        # Check output
        assert output["response"] == "Test response"
        assert output["context"] == "Test context"
        assert output["references"] == [{"term": "Gene A", "description": "Desc A", "genes": ["BRCA1"]}]
        assert output["unique_genes"] == {"BRCA1": 1}

def test_get_summary_to_query_df(rag_module):
    """Test get_summary_to_query_df method."""
    # Mock get_summary_to_query
    with patch.object(rag_module, "get_summary_to_query", return_value={
        "response": "Test response",
        "rouge": {},
        "context": "Test context",
        "references": [
            {"term": "Gene A", "description": "Desc A", "genes": ["BRCA1", "TP53"]}
        ],
        "unique_genes": {"BRCA1": 1, "TP53": 1}
    }) as mock_summary:
        
        # Call the method
        df = rag_module.get_summary_to_query_df("test query")
        
        # Check method call
        mock_summary.assert_called_once_with("test query", num_results=5, calculate_rouge=True)
        
        # Check DataFrame
        assert len(df) == 1
        assert df.iloc[0]["query"] == "test query"
        assert df.iloc[0]["response"] == "Test response"
        assert df.iloc[0]["context"] == "Test context"
        assert df.iloc[0]["reference_term"] == "Gene A"
        assert df.iloc[0]["reference_description"] == "Desc A"
        assert df.iloc[0]["reference_genes"] == "BRCA1, TP53"
        assert json.loads(df.iloc[0]["unique_genes"]) == {"BRCA1": 1, "TP53": 1}

def test_get_summary_to_query_df_empty_references(rag_module):
    """Test get_summary_to_query_df method with empty references."""
    # Mock get_summary_to_query
    with patch.object(rag_module, "get_summary_to_query", return_value={
        "response": "Test response",
        "rouge": {},
        "context": "Test context",
        "references": [],
        "unique_genes": {}
    }):
        
        # Call the method
        df = rag_module.get_summary_to_query_df("test query")
        
        # Check DataFrame
        assert len(df) == 1
        assert df.iloc[0]["query"] == "test query"
        assert df.iloc[0]["response"] == "Test response"
        assert df.iloc[0]["reference_term"] is None
        assert df.iloc[0]["reference_description"] is None
        assert df.iloc[0]["reference_genes"] is None

def test_save_output_to_json(rag_module):
    """Test save_output_to_json method."""
    output = {"key": "value"}
    
    # Mock json.dump instead of trying to check the file contents
    with patch("json.dump") as mock_dump:
        with patch("builtins.open", mock_open()) as m:
            rag_module.save_output_to_json(output, "test.json")
        
        # Check if file was opened correctly
        m.assert_called_once_with("test.json", "w")
        
        # Check if json.dump was called with correct arguments
        mock_dump.assert_called_once()
        args, kwargs = mock_dump.call_args
        assert args[0] == output  # First argument should be our output dict
        assert "indent" in kwargs  # Should have indent parameter

def test_to_markdown(rag_module):
    """Test to_markdown method."""
    output = {
        "response": "Test response",
        "rouge": {
            "rouge1": {"precision": 0.5, "recall": 0.6, "fmeasure": 0.55}
        },
        "context": "Test context",
        "references": [
            {"term": "Gene A", "description": "Desc A", "genes": ["BRCA1", "TP53"]}
        ],
        "unique_genes": {"BRCA1": 1, "TP53": 1}
    }
    
    markdown = rag_module.to_markdown(output)
    
    # Check markdown content
    assert "# Summary" in markdown
    assert "## Response" in markdown
    assert "Test response" in markdown
    assert "## Rouge Scores" in markdown
    assert "**rouge1**:" in markdown
    assert "- Precision: 0.5000" in markdown
    assert "## Context" in markdown
    assert "Test context" in markdown
    assert "## References" in markdown
    assert "**Term**: Gene A" in markdown
    assert "Description: Desc A" in markdown
    assert "Genes: BRCA1, TP53" in markdown
    assert "## Unique Genes" in markdown
    assert "- BRCA1: 1" in markdown
    assert "- TP53: 1" in markdown

def test_save_to_pickle(rag_module):
    """Test save_to_pickle method."""
    # Mock pickle.dump
    with patch("pickle.dump") as mock_dump:
        rag_module.save_to_pickle("test.pkl")
        
        # Check if pickle.dump was called
        mock_dump.assert_called_once()
        args, _ = mock_dump.call_args
        assert args[0] == rag_module  # First arg should be the object

def test_load_from_pickle():
    """Test load_from_pickle static method."""
    # Mock pickle.load to return a mock object
    mock_object = MagicMock()
    
    with patch("pickle.load", return_value=mock_object), \
         patch("builtins.open", mock_open()):
        loaded = RAGModule.load_from_pickle("test.pkl")
        
        # Check if correct object was returned
        assert loaded == mock_object

# Test main functions from the module
@pytest.fixture
def mock_rag_module():
    return MagicMock(spec=RAGModule)

def test_read_input_files():
    """Test read_input_files function from the module."""
    from geneinsight.api.get_summary import read_input_files
    
    # Mock pd.read_csv
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = [pd.DataFrame({"col1": [1, 2]}), pd.DataFrame({"col2": [3, 4]})]
        
        # Call the function
        df, topics = read_input_files("enrichment.csv", "topics.csv")
        
        # Check calls to pd.read_csv
        assert mock_read_csv.call_count == 2
        mock_read_csv.assert_any_call("enrichment.csv")
        mock_read_csv.assert_any_call("topics.csv")
        
        # Check return values
        assert isinstance(df, pd.DataFrame)
        assert isinstance(topics, pd.DataFrame)

def test_initialize_rag_module():
    """Test initialize_rag_module function."""
    from geneinsight.api.get_summary import initialize_rag_module
    
    # Mock RAGModule
    with patch("geneinsight.api.get_summary.RAGModule") as mock_rag:
        # Call the function
        df = pd.DataFrame({"col": [1, 2]})
        result = initialize_rag_module(df)
        
        # Check if RAGModule was initialized with df
        mock_rag.assert_called_once_with(df)
        
        # Check return value
        assert result == mock_rag.return_value

def test_get_topics_of_interest():
    """Test get_topics_of_interest function."""
    from geneinsight.api.get_summary import get_topics_of_interest
    
    # Create test DataFrame
    topics_df = pd.DataFrame({
        "prompt_type": ["subtopic_BERT", "other", "subtopic_BERT"],
        "generated_result": ["Topic 1", "Topic 2", "Topic 3"]
    })
    
    # Call the function
    topics = get_topics_of_interest(topics_df)
    
    # Check return values - should only include subtopic_BERT types
    assert topics == ["Topic 1", "Topic 3"]

def test_generate_results(mock_rag_module):
    """Test generate_results function."""
    from geneinsight.api.get_summary import generate_results
    
    # Mock get_summary_to_query_df method
    mock_rag_module.get_summary_to_query_df.return_value = pd.DataFrame({"response": ["Test"]})
    
    # Call the function
    topics = ["Topic 1", "Topic 2"]
    result = generate_results(mock_rag_module, topics, num_results=3)
    
    # Check method calls
    assert mock_rag_module.get_summary_to_query_df.call_count == 2
    for topic in topics:
        mock_rag_module.get_summary_to_query_df.assert_any_call(
            topic, num_results=3, calculate_rouge=False
        )
    
    # Check returned DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two rows for two topics

def test_save_results():
    """Test save_results function."""
    from geneinsight.api.get_summary import save_results
    
    # Create test DataFrame
    df = pd.DataFrame({"col": [1, 2]})
    
    # Mock DataFrame.to_csv
    with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        # Call the function
        save_results(df, "output.csv")
        
        # Check if to_csv was called with correct args
        mock_to_csv.assert_called_once_with("output.csv", index=False)

def test_main():
    """Test main function."""
    from geneinsight.api.get_summary import main
    
    # Create mock ArgumentParser
    args = MagicMock()
    args.enrichment_csv = "enrichment.csv"
    args.minor_topics_csv = "topics.csv"
    args.output_csv = "output.csv"
    args.num_results = 3
    
    # Mock all functions used in main
    with patch("geneinsight.api.get_summary.read_input_files") as mock_read_files, \
         patch("geneinsight.api.get_summary.initialize_rag_module") as mock_init_rag, \
         patch("geneinsight.api.get_summary.get_topics_of_interest") as mock_get_topics, \
         patch("geneinsight.api.get_summary.generate_results") as mock_generate, \
         patch("geneinsight.api.get_summary.save_results") as mock_save:
        
        # Set return values
        mock_read_files.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_get_topics.return_value = ["Topic 1"]
        mock_generate.return_value = pd.DataFrame()
        
        # Call the function
        main(args)
        
        # Check function calls
        mock_read_files.assert_called_once_with("enrichment.csv", "topics.csv")
        mock_init_rag.assert_called_once()
        mock_get_topics.assert_called_once()
        mock_generate.assert_called_once_with(
            mock_init_rag.return_value,
            ["Topic 1"],
            num_results=3
        )
        mock_save.assert_called_once_with(mock_generate.return_value, "output.csv")