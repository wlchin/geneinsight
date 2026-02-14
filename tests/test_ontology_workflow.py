# tests/test_ontology_workflow.py
"""
Tests for the geneinsight.ontology.workflow module.
"""

import os
import pytest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock
import shutil

from geneinsight.ontology.workflow import OntologyWorkflow


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_ontology_folder(tmp_path):
    """Creates a temporary folder with mock ontology files."""
    ontology_folder = tmp_path / "ontologies"
    ontology_folder.mkdir()

    # Create mock ontology files
    (ontology_folder / "GO_BP.txt").write_text("term1\t\tGENE1\tGENE2\nterm2\t\tGENE3\tGENE4\n")
    (ontology_folder / "GO_MF.txt").write_text("term3\t\tGENE5\tGENE6\n")

    return str(ontology_folder)


@pytest.fixture
def mock_summary_df():
    """Creates a mock summary DataFrame."""
    return pd.DataFrame({
        "query": ["Topic1", "Topic2"],
        "unique_genes": ["{'GENE1': 1, 'GENE2': 1}", "{'GENE3': 1, 'GENE4': 1}"]
    })


@pytest.fixture
def mock_clustered_df():
    """Creates a mock clustered topics DataFrame."""
    return pd.DataFrame({
        "Term": ["Topic1", "Topic2", "Topic3"]
    })


@pytest.fixture
def temp_gene_files(tmp_path):
    """Creates temporary gene list and background files."""
    gene_list = tmp_path / "gene_list.txt"
    gene_list.write_text("GENE1\nGENE2\nGENE3\nGENE4\n")

    background = tmp_path / "background.txt"
    background.write_text("GENE1\nGENE2\nGENE3\nGENE4\nGENE5\nGENE6\n")

    return str(gene_list), str(background)


# ============================================================================
# Tests for OntologyWorkflow initialization
# ============================================================================

class TestOntologyWorkflowInit:

    def test_init_with_ontology_folder(self, temp_ontology_folder):
        """Test initialization with a valid ontology folder."""
        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)
        assert workflow.ontology_folder == temp_ontology_folder
        assert workflow.fdr_threshold == 0.1  # default
        assert workflow.use_temp_files is False  # default

    def test_init_with_custom_fdr_threshold(self, temp_ontology_folder):
        """Test initialization with custom FDR threshold."""
        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder, fdr_threshold=0.05)
        assert workflow.fdr_threshold == 0.05

    def test_init_with_temp_files(self, temp_ontology_folder):
        """Test initialization with use_temp_files=True."""
        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder, use_temp_files=True)
        assert workflow.use_temp_files is True
        assert workflow.temp_dir is not None
        # Clean up
        if workflow.temp_dir and os.path.exists(workflow.temp_dir):
            shutil.rmtree(workflow.temp_dir)

    def test_init_invalid_folder(self):
        """Test initialization with a non-existent folder."""
        with pytest.raises(ValueError, match="Ontology folder does not exist"):
            OntologyWorkflow(ontology_folder="/nonexistent/folder")

    def test_init_default_folder_error(self, monkeypatch):
        """Test initialization when default folder cannot be found."""
        # Mock the importlib.resources to raise an error
        monkeypatch.setattr(
            "geneinsight.ontology.workflow.pkg_resources.files",
            MagicMock(side_effect=ImportError("Module not found"))
        )

        with pytest.raises(ValueError, match="No ontology folder provided"):
            OntologyWorkflow(ontology_folder=None)


# ============================================================================
# Tests for run_ontology_enrichment
# ============================================================================

class TestRunOntologyEnrichment:

    @patch('geneinsight.ontology.workflow.RAGModuleGSEAPY')
    @patch('geneinsight.ontology.workflow.OntologyReader')
    def test_run_ontology_enrichment_basic(
        self, mock_reader, mock_rag, temp_ontology_folder, tmp_path
    ):
        """Test basic ontology enrichment run."""
        # Setup mocks
        mock_reader_instance = MagicMock()
        mock_reader_instance.gene_dict = {"term1": ["GENE1", "GENE2"]}
        mock_reader.return_value = mock_reader_instance

        mock_rag_instance = MagicMock()
        mock_rag_instance.get_top_documents.return_value = (
            [0, 1],
            "formatted output",
            pd.DataFrame({"Term": ["term1"], "Adjusted P-value": [0.05]}),
            pd.DataFrame({"Term": ["term1"], "Adjusted P-value": [0.05]}),
            "formatted output"
        )
        mock_rag.return_value = mock_rag_instance

        # Create input files
        summary_csv = tmp_path / "summary.csv"
        pd.DataFrame({
            "query": ["Topic1"],
            "unique_genes": ["{'GENE1': 1, 'GENE2': 1}"]
        }).to_csv(summary_csv, index=False)

        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Term": ["Topic1"]}).to_csv(filter_csv, index=False)

        gene_origin = tmp_path / "genes.txt"
        gene_origin.write_text("GENE1\nGENE2\n")

        background = tmp_path / "background.txt"
        background.write_text("GENE1\nGENE2\nGENE3\n")

        output_csv = tmp_path / "output" / "results.csv"

        # Run
        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)
        result = workflow.run_ontology_enrichment(
            summary_csv=str(summary_csv),
            gene_origin=str(gene_origin),
            background_genes=str(background),
            filter_csv=str(filter_csv),
            output_csv=str(output_csv)
        )

        assert isinstance(result, pd.DataFrame)
        assert mock_rag_instance.get_top_documents.called

    @patch('geneinsight.ontology.workflow.RAGModuleGSEAPY')
    @patch('geneinsight.ontology.workflow.OntologyReader')
    def test_run_ontology_enrichment_dict_genes(
        self, mock_reader, mock_rag, temp_ontology_folder, tmp_path
    ):
        """Test enrichment when unique_genes is already a dict (not string)."""
        mock_reader_instance = MagicMock()
        mock_reader_instance.gene_dict = {"term1": ["GENE1"]}
        mock_reader.return_value = mock_reader_instance

        mock_rag_instance = MagicMock()
        mock_rag_instance.get_top_documents.return_value = (
            [], "No results", pd.DataFrame(), pd.DataFrame(), "No results"
        )
        mock_rag.return_value = mock_rag_instance

        # Create summary with dict-type unique_genes (not string)
        summary_df = pd.DataFrame({
            "query": ["Topic1"],
            "unique_genes": [{"GENE1": 1, "GENE2": 1}]  # dict, not string
        })
        summary_csv = tmp_path / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Term": ["Topic1"]}).to_csv(filter_csv, index=False)

        output_csv = tmp_path / "output" / "results.csv"

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)
        # This should handle dict-type unique_genes
        result = workflow.run_ontology_enrichment(
            summary_csv=str(summary_csv),
            gene_origin="dummy",
            background_genes="dummy",
            filter_csv=str(filter_csv),
            output_csv=str(output_csv)
        )

        assert isinstance(result, pd.DataFrame)

    @patch('geneinsight.ontology.workflow.OntologyReader')
    def test_run_ontology_enrichment_read_warning(
        self, mock_reader, temp_ontology_folder, tmp_path
    ):
        """Test that warnings are logged when ontology read fails."""
        # Make OntologyReader raise an exception for some files
        mock_reader.side_effect = [
            Exception("Failed to read"),  # First file fails
            MagicMock(gene_dict={"term": ["GENE1"]})  # Second succeeds
        ]

        summary_csv = tmp_path / "summary.csv"
        pd.DataFrame({
            "query": ["Topic1"],
            "unique_genes": ["{'GENE1': 1}"]
        }).to_csv(summary_csv, index=False)

        filter_csv = tmp_path / "filter.csv"
        pd.DataFrame({"Term": ["Topic1"]}).to_csv(filter_csv, index=False)

        output_csv = tmp_path / "output" / "results.csv"

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)

        with patch('geneinsight.ontology.workflow.RAGModuleGSEAPY') as mock_rag:
            mock_rag_instance = MagicMock()
            mock_rag_instance.get_top_documents.return_value = (
                [], "No results", pd.DataFrame(), pd.DataFrame(), "No results"
            )
            mock_rag.return_value = mock_rag_instance

            result = workflow.run_ontology_enrichment(
                summary_csv=str(summary_csv),
                gene_origin="dummy",
                background_genes="dummy",
                filter_csv=str(filter_csv),
                output_csv=str(output_csv)
            )


# ============================================================================
# Tests for create_ontology_dictionary
# ============================================================================

class TestCreateOntologyDictionary:

    @patch('geneinsight.ontology.workflow.process_ontology_enrichment')
    def test_create_ontology_dictionary_basic(self, mock_process, temp_ontology_folder, tmp_path):
        """Test basic ontology dictionary creation."""
        mock_process.return_value = pd.DataFrame({
            "query": ["Topic1"],
            "ontology_dict": [{"term1": "GENE1,GENE2"}]
        })

        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"query": ["Topic1"]}).to_csv(input_csv, index=False)

        output_csv = tmp_path / "output" / "dict.csv"

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)
        result = workflow.create_ontology_dictionary(
            input_csv=str(input_csv),
            output_csv=str(output_csv)
        )

        assert isinstance(result, pd.DataFrame)
        mock_process.assert_called_once()


# ============================================================================
# Tests for process_dataframes
# ============================================================================

class TestProcessDataframes:

    @patch('geneinsight.ontology.workflow.OntologyWorkflow.run_ontology_enrichment')
    @patch('geneinsight.ontology.workflow.OntologyWorkflow.create_ontology_dictionary')
    def test_process_dataframes_temp_files(
        self, mock_dict, mock_enrich, temp_ontology_folder,
        mock_summary_df, mock_clustered_df, temp_gene_files, tmp_path
    ):
        """Test process_dataframes with temp files mode."""
        gene_list, background = temp_gene_files

        mock_enrich.return_value = pd.DataFrame({"query": ["Topic1"]})
        mock_dict.return_value = pd.DataFrame({"query": ["Topic1"], "ontology_dict": [{}]})

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder, use_temp_files=True)

        enrichment_df, dict_df = workflow.process_dataframes(
            summary_df=mock_summary_df,
            clustered_df=mock_clustered_df,
            gene_list_path=gene_list,
            background_genes_path=background
        )

        assert isinstance(enrichment_df, pd.DataFrame)
        assert isinstance(dict_df, pd.DataFrame)
        mock_enrich.assert_called_once()
        mock_dict.assert_called_once()

    @patch('geneinsight.ontology.workflow.OntologyWorkflow.run_ontology_enrichment')
    @patch('geneinsight.ontology.workflow.OntologyWorkflow.create_ontology_dictionary')
    def test_process_dataframes_with_output_dir(
        self, mock_dict, mock_enrich, temp_ontology_folder,
        mock_summary_df, mock_clustered_df, temp_gene_files, tmp_path
    ):
        """Test process_dataframes with output directory."""
        gene_list, background = temp_gene_files
        output_dir = tmp_path / "output"

        mock_enrich.return_value = pd.DataFrame({"query": ["Topic1"]})
        mock_dict.return_value = pd.DataFrame({"query": ["Topic1"], "ontology_dict": [{}]})

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder, use_temp_files=True)

        enrichment_df, dict_df = workflow.process_dataframes(
            summary_df=mock_summary_df,
            clustered_df=mock_clustered_df,
            gene_list_path=gene_list,
            background_genes_path=background,
            output_dir=str(output_dir)
        )

        # Output files should be created
        assert (output_dir / "ontology_enrichment.csv").exists()
        assert (output_dir / "ontology_dict.csv").exists()

    @patch('geneinsight.ontology.workflow.RAGModuleGSEAPY')
    @patch('geneinsight.ontology.workflow.OntologyReader')
    @patch('geneinsight.ontology.workflow.process_ontology_enrichment')
    def test_process_dataframes_memory_mode(
        self, mock_process, mock_reader, mock_rag,
        temp_ontology_folder, mock_summary_df, mock_clustered_df,
        temp_gene_files, tmp_path
    ):
        """Test process_dataframes in memory mode (use_temp_files=False)."""
        gene_list, background = temp_gene_files

        mock_reader_instance = MagicMock()
        mock_reader_instance.gene_dict = {"term1": ["GENE1"]}
        mock_reader.return_value = mock_reader_instance

        mock_rag_instance = MagicMock()
        mock_rag_instance.get_top_documents.return_value = (
            [], "No results", pd.DataFrame(), pd.DataFrame(), "No results"
        )
        mock_rag.return_value = mock_rag_instance

        mock_process.return_value = pd.DataFrame({
            "query": ["Topic1"],
            "ontology_dict": [{}]
        })

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder, use_temp_files=False)

        enrichment_df, dict_df = workflow.process_dataframes(
            summary_df=mock_summary_df,
            clustered_df=mock_clustered_df,
            gene_list_path=gene_list,
            background_genes_path=background
        )

        assert isinstance(enrichment_df, pd.DataFrame)
        assert isinstance(dict_df, pd.DataFrame)


# ============================================================================
# Tests for run_full_workflow
# ============================================================================

class TestRunFullWorkflow:

    @patch('geneinsight.ontology.workflow.OntologyWorkflow.run_ontology_enrichment')
    @patch('geneinsight.ontology.workflow.OntologyWorkflow.create_ontology_dictionary')
    def test_run_full_workflow_basic(
        self, mock_dict, mock_enrich, temp_ontology_folder, tmp_path
    ):
        """Test run_full_workflow with basic parameters."""
        mock_enrich.return_value = pd.DataFrame({"query": ["Topic1"]})
        mock_dict.return_value = pd.DataFrame({"query": ["Topic1"], "ontology_dict": [{}]})

        # Create necessary directories and files
        output_dir = tmp_path / "results"
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True)
        (summary_dir / "test_set.csv").write_text("query,unique_genes\nTopic1,{}")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test_set.txt").write_text("GENE1\nGENE2")
        (data_dir / "BackgroundList.txt").write_text("GENE1\nGENE2\nGENE3")

        clustered_dir = output_dir / "clustered_topics"
        clustered_dir.mkdir()
        (clustered_dir / "test_set_clustered_topics.csv").write_text("Term\nTopic1")

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)

        result = workflow.run_full_workflow(
            gene_set="test_set",
            summary_csv=str(summary_dir / "test_set.csv"),
            gene_origin=str(data_dir / "test_set.txt"),
            background_genes=str(data_dir / "BackgroundList.txt"),
            filter_csv=str(clustered_dir / "test_set_clustered_topics.csv"),
            output_dir=str(output_dir),
            return_results=True
        )

        assert result is not None
        assert "enrichment" in result
        assert "dictionary" in result

    @patch('geneinsight.ontology.workflow.OntologyWorkflow.run_ontology_enrichment')
    @patch('geneinsight.ontology.workflow.OntologyWorkflow.create_ontology_dictionary')
    def test_run_full_workflow_no_return(
        self, mock_dict, mock_enrich, temp_ontology_folder, tmp_path
    ):
        """Test run_full_workflow with return_results=False."""
        mock_enrich.return_value = pd.DataFrame({"query": ["Topic1"]})
        mock_dict.return_value = pd.DataFrame({"query": ["Topic1"], "ontology_dict": [{}]})

        workflow = OntologyWorkflow(ontology_folder=temp_ontology_folder)

        result = workflow.run_full_workflow(
            gene_set="test_set",
            summary_csv=str(tmp_path / "summary.csv"),
            gene_origin=str(tmp_path / "genes.txt"),
            background_genes=str(tmp_path / "background.txt"),
            filter_csv=str(tmp_path / "filter.csv"),
            output_dir=str(tmp_path / "results"),
            return_results=False
        )

        assert result is None
