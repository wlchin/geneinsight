import os
import tempfile
import shutil
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path

# Import the module to test
from geneinsight.pipeline import Pipeline

# Fixtures for common test setup
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def output_dir(temp_dir):
    """Create an output directory for the pipeline."""
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.fixture
def mock_files(temp_dir):
    """Create mock input files for testing."""
    # Create a mock query gene set file
    query_file = os.path.join(temp_dir, "query_genes.txt")
    with open(query_file, "w") as f:
        f.write("GENE1\nGENE2\nGENE3\n")
    
    # Create a mock background gene list file
    background_file = os.path.join(temp_dir, "background_genes.txt")
    with open(background_file, "w") as f:
        f.write("GENE1\nGENE2\nGENE3\nGENE4\nGENE5\n")
    
    return {"query": query_file, "background": background_file}

@pytest.fixture
def pipeline(output_dir):
    """Create a Pipeline instance for testing."""
    pipeline = Pipeline(
        output_dir=output_dir,
        n_samples=2,  # Reduce for faster tests
        num_topics=5,  # Small number for testing
        pvalue_threshold=0.05,
        api_service="mock_service",
        api_model="mock_model",
        api_parallel_jobs=1
    )
    return pipeline

@pytest.fixture
def mock_enrichment_df():
    """Create a mock enrichment DataFrame."""
    return pd.DataFrame({
        "gene": ["GENE1", "GENE2", "GENE3"],
        "description": ["Description 1", "Description 2", "Description 3"],
        "pvalue": [0.01, 0.02, 0.03]
    })

@pytest.fixture
def mock_documents_df():
    """Create a mock documents DataFrame."""
    return pd.DataFrame({
        "description": [
            "Sample text for topic modeling 1",
            "Sample text for topic modeling 2",
            "Sample text for topic modeling 3"
        ]
    })

@pytest.fixture
def mock_topics_df():
    """Create a mock topics DataFrame."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "words": ["gene, protein, pathway", "disease, cell, tissue", "mutation, cancer, therapy"],
        "count": [10, 8, 6]
    })

@pytest.fixture
def mock_prompts_df():
    """Create a mock prompts DataFrame."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "prompt": ["Summarize gene, protein, pathway", "Summarize disease, cell, tissue", "Summarize mutation, cancer, therapy"]
    })

@pytest.fixture
def mock_api_results_df():
    """Create a mock API results DataFrame."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "response": ["Response for topic 0", "Response for topic 1", "Response for topic 2"],
        "subtopics": ["subtopic1, subtopic2", "subtopic3, subtopic4", "subtopic5, subtopic6"]
    })

@pytest.fixture
def mock_summary_df():
    """Create a mock summary DataFrame."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "subtopic": ["subtopic1", "subtopic3", "subtopic5"],
        "genes": ["GENE1,GENE2", "GENE2,GENE3", "GENE1,GENE3"]
    })

@pytest.fixture
def mock_enriched_df():
    """Create a mock enriched DataFrame."""
    return pd.DataFrame({
        "topic_id": [0, 1, 2],
        "subtopic": ["subtopic1", "subtopic3", "subtopic5"],
        "genes": ["GENE1,GENE2", "GENE2,GENE3", "GENE1,GENE3"],
        "pvalue": [0.01, 0.02, 0.03],
        "adjusted_pvalue": [0.03, 0.04, 0.05]
    })

@pytest.fixture
def mock_key_topics_df():
    """Create a mock key topics DataFrame."""
    return pd.DataFrame({
        "term": ["gene", "protein", "disease", "cell", "mutation"],
        "count": [15, 12, 10, 8, 6]
    })

@pytest.fixture
def mock_clustered_df():
    """Create a mock clustered DataFrame."""
    return pd.DataFrame({
        "term": ["gene", "protein", "disease", "cell", "mutation"],
        "count": [15, 12, 10, 8, 6],
        "cluster": [0, 0, 1, 1, 2]
    })

@pytest.fixture
def mock_ontology_dict_df():
    """Create a mock ontology dictionary DataFrame."""
    return pd.DataFrame({
        "ontology": ["GO", "KEGG", "REACTOME"],
        "term": ["Term1", "Term2", "Term3"],
        "pvalue": [0.01, 0.02, 0.03]
    })


# Tests for Pipeline initialization
class TestPipelineInit:
    def test_pipeline_init(self, output_dir):
        """Test that pipeline initializes correctly with provided parameters."""
        pipeline = Pipeline(
            output_dir=output_dir,
            n_samples=5,
            num_topics=10,
            pvalue_threshold=0.01,
            api_service="openai",
            api_model="gpt-4",
            api_parallel_jobs=4
        )
        
        # Check that parameters are set correctly
        assert pipeline.output_dir == os.path.abspath(output_dir)
        assert pipeline.n_samples == 5
        assert pipeline.num_topics == 10
        assert pipeline.pvalue_threshold == 0.01
        assert pipeline.api_service == "openai"
        assert pipeline.api_model == "gpt-4"
        assert pipeline.api_parallel_jobs == 4
        
        # Check that directories are created
        assert os.path.exists(pipeline.output_dir)
        assert os.path.exists(pipeline.temp_dir)
        assert os.path.exists(pipeline.dirs["enrichment"])
        assert os.path.exists(pipeline.dirs["topics"])
        assert os.path.exists(pipeline.dirs["final"])
        assert os.path.exists(pipeline.dirs["sphinx_builds"])
    
    def test_pipeline_init_with_temp_dir(self, temp_dir, output_dir):
        """Test pipeline initialization with custom temp_dir."""
        custom_temp = os.path.join(temp_dir, "custom_temp")
        
        pipeline = Pipeline(
            output_dir=output_dir,
            temp_dir=custom_temp
        )
        
        assert pipeline.temp_dir == os.path.abspath(custom_temp)
        assert os.path.exists(custom_temp)
        assert pipeline._temp_is_system is False

    def test_pipeline_init_with_system_temp(self, output_dir):
        """Test pipeline initialization with system temp directory."""
        pipeline = Pipeline(output_dir=output_dir)
        
        assert pipeline.temp_dir.startswith(tempfile.gettempdir())
        assert pipeline._temp_is_system is True


# Tests for individual pipeline methods
class TestPipelineMethods:
    @patch("geneinsight.pipeline.process_gene_enrichment")
    def test_get_stringdb_enrichment(self, mock_process, pipeline, mock_files, mock_enrichment_df, mock_documents_df):
        """Test the _get_stringdb_enrichment method."""
        # Setup the mock
        mock_process.return_value = (mock_enrichment_df, ["Description 1", "Description 2", "Description 3"])
        
        # Call the method
        enrichment_df, documents_df = pipeline._get_stringdb_enrichment(mock_files["query"])
        
        # Verify the call
        mock_process.assert_called_once_with(
            input_file=mock_files["query"],
            output_dir=pipeline.dirs["enrichment"],
            mode="single"
        )
        
        # Check results
        pd.testing.assert_frame_equal(enrichment_df, mock_enrichment_df)
        assert documents_df.shape[0] == 3
        assert "description" in documents_df.columns
    
    @patch("geneinsight.pipeline.run_multiple_seed_topic_modeling")
    def test_run_topic_modeling(self, mock_topic_modeling, pipeline, mock_documents_df, mock_topics_df):
        """Test the _run_topic_modeling method."""
        # Setup the mock
        mock_topic_modeling.return_value = mock_topics_df
        
        # Call the method
        result = pipeline._run_topic_modeling(mock_documents_df)
        
        # Verify the call
        mock_topic_modeling.assert_called_once()
        assert mock_topic_modeling.call_args[1]["method"] == "bertopic"
        assert mock_topic_modeling.call_args[1]["num_topics"] == pipeline.num_topics
        assert mock_topic_modeling.call_args[1]["n_samples"] == pipeline.n_samples
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_topics_df)
    
    @patch("geneinsight.pipeline.generate_prompts")
    def test_generate_prompts(self, mock_generate_prompts, pipeline, mock_topics_df, mock_prompts_df):
        """Test the _generate_prompts method."""
        # Setup the mock
        mock_generate_prompts.return_value = mock_prompts_df
        
        # Call the method
        result = pipeline._generate_prompts(mock_topics_df)
        
        # Verify the call
        mock_generate_prompts.assert_called_once()
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_prompts_df)
    
    @patch("geneinsight.pipeline.batch_process_api_calls")
    def test_process_api_calls(self, mock_api_calls, pipeline, mock_prompts_df, mock_api_results_df):
        """Test the _process_api_calls method."""
        # Setup the mock
        mock_api_calls.return_value = mock_api_results_df
        
        # Call the method
        result = pipeline._process_api_calls(mock_prompts_df)
        
        # Verify the call
        mock_api_calls.assert_called_once()
        assert mock_api_calls.call_args[1]["service"] == pipeline.api_service
        assert mock_api_calls.call_args[1]["model"] == pipeline.api_model
        assert mock_api_calls.call_args[1]["n_jobs"] == pipeline.api_parallel_jobs
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_api_results_df)
    
    @patch("geneinsight.pipeline.create_summary")
    def test_create_summary(self, mock_create_summary, pipeline, mock_api_results_df, mock_enrichment_df, mock_summary_df):
        """Test the _create_summary method."""
        # Setup the mock
        mock_create_summary.return_value = mock_summary_df
        
        # Call the method
        result = pipeline._create_summary(mock_api_results_df, mock_enrichment_df)
        
        # Verify the call
        mock_create_summary.assert_called_once_with(mock_api_results_df, mock_enrichment_df, ANY)
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_summary_df)
    
    @patch("geneinsight.pipeline.hypergeometric_enrichment")
    def test_perform_hypergeometric_enrichment(self, mock_enrichment, pipeline, mock_summary_df, mock_files, mock_enriched_df):
        """Test the _perform_hypergeometric_enrichment method."""
        # Setup the mock
        mock_enrichment.return_value = mock_enriched_df
        
        # Call the method
        result = pipeline._perform_hypergeometric_enrichment(
            mock_summary_df, mock_files["query"], mock_files["background"]
        )
        
        # Verify the call
        mock_enrichment.assert_called_once()
        assert mock_enrichment.call_args[1]["pvalue_threshold"] == pipeline.pvalue_threshold
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_enriched_df)
    
    @patch("geneinsight.pipeline.run_multiple_seed_topic_modeling")
    def test_run_topic_modeling_on_filtered_sets(self, mock_topic_modeling, pipeline, mock_enriched_df, mock_topics_df):
        """Test the _run_topic_modeling_on_filtered_sets method."""
        # Setup the mock
        mock_topic_modeling.return_value = mock_topics_df
        
        # Call the method
        result = pipeline._run_topic_modeling_on_filtered_sets(mock_enriched_df)
        
        # Verify the call
        mock_topic_modeling.assert_called_once()
        assert mock_topic_modeling.call_args[1]["method"] == "bertopic"
        assert mock_topic_modeling.call_args[1]["num_topics"] is None  # Auto-determine
        assert mock_topic_modeling.call_args[1]["n_samples"] == 10
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_topics_df)
    
    @patch("geneinsight.pipeline.count_top_terms")
    def test_get_key_topics(self, mock_count_terms, pipeline, mock_topics_df, mock_key_topics_df):
        """Test the _get_key_topics method."""
        # Setup the mock
        mock_count_terms.return_value = mock_key_topics_df
        
        # Call the method
        result = pipeline._get_key_topics(mock_topics_df)
        
        # Verify the call
        mock_count_terms.assert_called_once()
        assert mock_count_terms.call_args[1]["top_n"] is None
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_key_topics_df)
    
    @patch("geneinsight.pipeline.filter_terms_by_similarity")
    def test_filter_topics(self, mock_filter, pipeline, mock_key_topics_df, mock_enriched_df):
        """Test the _filter_topics method."""
        # Setup the mock
        mock_filter.return_value = mock_enriched_df
        
        # Call the method
        result = pipeline._filter_topics(mock_key_topics_df)
        
        # Verify the call
        mock_filter.assert_called_once()
        assert mock_filter.call_args[1]["target_rows"] == pipeline.target_filtered_topics
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_enriched_df)
    
    @patch("geneinsight.pipeline.run_clustering")
    def test_run_clustering(self, mock_clustering, pipeline, mock_enriched_df, mock_clustered_df):
        """Test the _run_clustering method."""
        # Setup the mocks
        mock_clustering.return_value = None  # Function doesn't return the DataFrame
        
        # Create a patch for pd.read_csv
        with patch('pandas.read_csv', return_value=mock_clustered_df):
            # Call the method
            result = pipeline._run_clustering(mock_enriched_df)
            
            # Verify the call
            mock_clustering.assert_called_once()
            
            # Check results
            pd.testing.assert_frame_equal(result, mock_clustered_df)
    
    def test_finalize_outputs(self, pipeline):
        """Test the _finalize_outputs method."""
        run_id = "test_run_123"
        dataframes = {
            "enrichment": pd.DataFrame({"col1": [1, 2, 3]}),
            "topics": pd.DataFrame({"col2": [4, 5, 6]}),
            "summary": pd.DataFrame({"col3": [7, 8, 9]})
        }
        
        # Call the method
        output_path = pipeline._finalize_outputs(run_id, dataframes)
        
        # Check results
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(output_path, "enrichment.csv"))
        assert os.path.exists(os.path.join(output_path, "topics.csv"))
        assert os.path.exists(os.path.join(output_path, "summary.csv"))
        assert os.path.exists(os.path.join(output_path, "metadata.csv"))
        
        # Check metadata
        metadata = pd.read_csv(os.path.join(output_path, "metadata.csv"))
        assert metadata["run_id"][0] == run_id
        assert metadata["n_samples"][0] == pipeline.n_samples
        assert metadata["api_service"][0] == pipeline.api_service
    
    @patch("geneinsight.pipeline.OntologyWorkflow")
    def test_perform_ontology_enrichment(self, mock_ontology_workflow, pipeline, mock_summary_df, mock_clustered_df, mock_files, mock_ontology_dict_df):
        """Test the _perform_ontology_enrichment method."""
        # Setup the mock
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.process_dataframes.return_value = (mock_clustered_df, mock_ontology_dict_df)
        mock_ontology_workflow.return_value = mock_workflow_instance
        
        # Call the method
        result = pipeline._perform_ontology_enrichment(
            mock_summary_df, mock_clustered_df, mock_files["query"], mock_files["background"]
        )
        
        # Verify the calls
        mock_ontology_workflow.assert_called_once()
        mock_workflow_instance.process_dataframes.assert_called_once()
        
        # Check results
        pd.testing.assert_frame_equal(result, mock_ontology_dict_df)


# Tests for directory and file operations
class TestPipelineDirectoryOperations:
    def test_cleanup_temp(self, temp_dir, output_dir):
        """Test the _cleanup_temp method with system temp directory."""
        # Create pipeline with system temp
        pipeline = Pipeline(output_dir=output_dir)
        temp_path = pipeline.temp_dir
        
        # Verify temp directory exists
        assert os.path.exists(temp_path)
        
        # Call cleanup method
        pipeline._cleanup_temp()
        
        # Verify temp directory is removed
        assert not os.path.exists(temp_path)
    
    def test_cleanup_temp_with_custom_dir(self, temp_dir, output_dir):
        """Test the _cleanup_temp method with custom temp directory."""
        custom_temp = os.path.join(temp_dir, "custom_temp")
        
        # Create pipeline with custom temp
        pipeline = Pipeline(output_dir=output_dir, temp_dir=custom_temp)
        
        # Verify temp directory exists
        assert os.path.exists(custom_temp)
        
        # Call cleanup method (should not remove custom directory)
        pipeline._cleanup_temp()
        
        # Verify temp directory still exists
        assert os.path.exists(custom_temp)
    
    @patch("geneinsight.pipeline.shutil.rmtree")
    @patch("geneinsight.pipeline.shutil.copytree")
    @patch("geneinsight.pipeline.os.rename")
    @patch("os.path.exists", return_value=True)
    def test_reorganize_output_directory(self, mock_exists, mock_rename, mock_copytree, mock_rmtree, pipeline):
        """Test the _reorganize_output_directory method."""
        output_path = os.path.join(pipeline.dirs["final"], "test_run")
        
        # Call the method
        pipeline._reorganize_output_directory(output_path)
        
        # Verify the calls
        assert mock_rmtree.call_count >= 1  # Should be called for ontology folder and original sphinx_builds
        assert mock_copytree.call_count == 1  # Should be called to copy sphinx_builds
        assert mock_rename.call_count == 1  # Should be called to rename temp sphinx_builds
    
    @patch("geneinsight.pipeline.zip_directory")
    def test_zip_results_folders(self, mock_zip, pipeline, temp_dir):
        """Test the _zip_results_folders method."""
        # Create test directories
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create test subdirectories
        os.makedirs(os.path.join(results_dir, "dir1"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "dir2"), exist_ok=True)
        
        # Call the method
        pipeline._zip_results_folders(results_dir)
        
        # Verify the calls
        assert mock_zip.call_count == 2  # Should be called for each directory
    
    @patch("geneinsight.pipeline.reports_pipeline.run_pipeline")
    def test_generate_report(self, mock_run_pipeline, pipeline, mock_files):
        """Test the _generate_report method."""
        # Setup the mock
        mock_run_pipeline.return_value = (True, "/path/to/index.html")
        
        output_path = os.path.join(pipeline.dirs["final"], "test_run")
        os.makedirs(output_path, exist_ok=True)
        
        # Call the method
        report_path = pipeline._generate_report(
            output_path=output_path,
            query_gene_set=mock_files["query"],
            report_title="Test Report"
        )
        
        # Verify the call
        mock_run_pipeline.assert_called_once()
        assert report_path == "/path/to/index.html"


# Test for full pipeline run
class TestPipelineRun:
    @patch.multiple("geneinsight.pipeline.Pipeline",
        _get_stringdb_enrichment=MagicMock(return_value=(pd.DataFrame(), pd.DataFrame())),
        _run_topic_modeling=MagicMock(return_value=pd.DataFrame()),
        _generate_prompts=MagicMock(return_value=pd.DataFrame()),
        _process_api_calls=MagicMock(return_value=pd.DataFrame()),
        _create_summary=MagicMock(return_value=pd.DataFrame()),
        _perform_hypergeometric_enrichment=MagicMock(return_value=pd.DataFrame()),
        _run_topic_modeling_on_filtered_sets=MagicMock(return_value=pd.DataFrame()),
        _get_key_topics=MagicMock(return_value=pd.DataFrame()),
        _filter_topics=MagicMock(return_value=pd.DataFrame()),
        _run_clustering=MagicMock(return_value=pd.DataFrame()),
        _perform_ontology_enrichment=MagicMock(return_value=pd.DataFrame()),
        _finalize_outputs=MagicMock(return_value="/path/to/output"),
        _generate_report=MagicMock(return_value="/path/to/report"),
        _reorganize_output_directory=MagicMock()
    )
    def test_full_pipeline_run(self, pipeline, mock_files):
        """Test the full pipeline run method with mocked component methods."""
        # Call the run method
        result = pipeline.run(
            query_gene_set=mock_files["query"],
            background_gene_list=mock_files["background"],
            generate_report=True,
            report_title="Test Report"
        )
        
        # Verify all methods were called
        pipeline._get_stringdb_enrichment.assert_called_once()
        pipeline._run_topic_modeling.assert_called_once()
        pipeline._generate_prompts.assert_called_once()
        pipeline._process_api_calls.assert_called_once()
        pipeline._create_summary.assert_called_once()
        pipeline._perform_hypergeometric_enrichment.assert_called_once()
        pipeline._run_topic_modeling_on_filtered_sets.assert_called_once()
        pipeline._get_key_topics.assert_called_once()
        pipeline._filter_topics.assert_called_once()
        pipeline._run_clustering.assert_called_once()
        pipeline._perform_ontology_enrichment.assert_called_once()
        pipeline._finalize_outputs.assert_called_once()
        pipeline._generate_report.assert_called_once()
        pipeline._reorganize_output_directory.assert_called_once()
        
        # Check result
        assert result == pipeline.dirs["final"]
    
    @patch.multiple("geneinsight.pipeline.Pipeline",
        _get_stringdb_enrichment=MagicMock(side_effect=Exception("Test error")),
        _cleanup_temp=MagicMock()
    )
    def test_pipeline_run_error_handling(self, pipeline, mock_files):
        """Test error handling in the pipeline run method."""
        # Call the run method with expectation of exception
        with pytest.raises(Exception, match="Test error"):
            pipeline.run(
                query_gene_set=mock_files["query"],
                background_gene_list=mock_files["background"]
            )
        
        # Verify cleanup was called
        pipeline._cleanup_temp.assert_called_once()


# Test command-line interface
class TestCommandLineInterface:
    @patch("geneinsight.pipeline.Pipeline")
    @patch("geneinsight.pipeline.argparse.ArgumentParser.parse_args")
    def test_command_line_interface(self, mock_parse_args, mock_pipeline_class, mock_files):
        """Test the command-line interface."""
        # Setup mock args
        args = MagicMock()
        args.query_gene_set = mock_files["query"]
        args.background_gene_list = mock_files["background"]
        args.output_dir = "./output"
        args.no_report = False
        args.n_samples = 5
        args.num_topics = 10
        args.pvalue_threshold = 0.05
        args.api_service = "openai"
        args.api_model = "gpt-4"
        args.api_parallel_jobs = 4
        args.api_base_url = None
        args.target_filtered_topics = 25
        args.temp_dir = None
        args.report_title = "Test Report"
        
        mock_parse_args.return_value = args
        
        # Setup mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline_instance
        
        # Call the main function by executing the module
        from geneinsight.pipeline import __name__ as module_name
        
        # Mock __name__ == "__main__"
        with patch("geneinsight.pipeline.__name__", "__main__"):
            with patch("sys.argv", ["pipeline.py", mock_files["query"], mock_files["background"]]):
                # This would execute the if __name__ == "__main__" block
                # But we need to import it again to trigger it
                import importlib
                importlib.reload(__import__("geneinsight.pipeline"))
        
        # Verify Pipeline was initialized with correct args
        mock_pipeline_class.assert_called_once_with(
            output_dir="./output",
            temp_dir=None,
            n_samples=5,
            num_topics=10,
            pvalue_threshold=0.05,
            api_service="openai",
            api_model="gpt-4",
            api_parallel_jobs=4,
            api_base_url=None,
            target_filtered_topics=25
        )
        
        # Verify run was called with correct args
        mock_pipeline_instance.run.assert_called_once_with(
            query_gene_set=mock_files["query"],
            background_gene_list=mock_files["background"],
            generate_report=True,
            report_title="Test Report"
        )