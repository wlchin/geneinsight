# test_ontology_workflow.py

import os
import pytest
import pandas as pd
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

from geneinsight.ontology.workflow import OntologyWorkflow

@pytest.fixture
def mock_ontology_folder(tmp_path):
    """
    Create a mock ontology folder with dummy files for testing.
    """
    folder = tmp_path / "ontology"
    folder.mkdir()
    
    # Create some fake ontology files
    (folder / "dummy_ontology1.txt").write_text("fake content 1")
    (folder / "dummy_ontology2.txt").write_text("fake content 2")
    
    return str(folder)


@pytest.fixture
def empty_ontology_folder(tmp_path):
    """
    Create an empty folder (or with invalid files) to test 'no valid ontology files' scenario.
    """
    folder = tmp_path / "empty_ontology"
    folder.mkdir()
    
    # Optionally, create hidden or invalid files to ensure the code
    # doesn't find any "valid" ontologies.
    (folder / ".hidden_file").write_text("hidden content")
    return str(folder)


@pytest.fixture
def mock_data_files(tmp_path):
    """
    Create mock CSV files for summary, filter, and gene files.
    Returns paths to these files.
    """
    summary_csv = tmp_path / "summary.csv"
    filter_csv = tmp_path / "filter.csv"
    background_txt = tmp_path / "background.txt"
    gene_origin_txt = tmp_path / "gene_origin.txt"

    # Create a small CSV with minimal columns used in the workflow
    # Specifically "query" and "unique_genes" columns
    summary_df = pd.DataFrame({
        "query": ["TermA", "TermB"],
        "unique_genes": [
            '{"GENE1": 1, "GENE2": 2}',  # string-dict representation
            '{"GENE3": 1, "GENE4": 2}'
        ],
    })
    summary_df.to_csv(summary_csv, index=False)

    # Create filter CSV with "Term" column
    filter_df = pd.DataFrame({
        "Term": ["TermA", "TermB"]
    })
    filter_df.to_csv(filter_csv, index=False)

    # Create background gene list
    background_txt.write_text("BG1\nBG2\nBG3\n")

    # Create gene origin file
    gene_origin_txt.write_text("GENE1\nGENE2\nGENE3\nGENE4\n")

    return {
        "summary_csv": str(summary_csv),
        "filter_csv": str(filter_csv),
        "background_txt": str(background_txt),
        "gene_origin_txt": str(gene_origin_txt),
    }


@pytest.fixture
def mock_ontology_workflow(mock_ontology_folder):
    """
    Instantiates an OntologyWorkflow with a mock ontology folder.
    """
    return OntologyWorkflow(
        ontology_folder=mock_ontology_folder,
        fdr_threshold=0.1,
        use_temp_files=False
    )


# -----------------
# Tests for __init__
# -----------------
def test_init_with_valid_folder(mock_ontology_folder):
    """
    Test that OntologyWorkflow initializes with a valid folder.
    """
    workflow = OntologyWorkflow(ontology_folder=mock_ontology_folder)
    assert workflow.ontology_folder == mock_ontology_folder
    assert workflow.fdr_threshold == 0.1
    assert workflow.use_temp_files is False


def test_init_with_invalid_folder(tmp_path):
    """
    Test that OntologyWorkflow raises an error when folder does not exist.
    """
    non_existent = str(tmp_path / "no_such_folder")
    with pytest.raises(ValueError, match="Ontology folder does not exist"):
        OntologyWorkflow(ontology_folder=non_existent)


def test_init_no_folder_and_no_default():
    """
    Test that if ontology_folder=None and the default package location is missing,
    a ValueError is raised.
    We mock out import to simulate a missing package resource.
    """
    with patch("geneinsight.ontology.workflow.pkg_resources.files", side_effect=AttributeError("No default folder")):
        with pytest.raises(ValueError, match="No ontology folder provided and default not found"):
            OntologyWorkflow(ontology_folder=None)


# -----------------------------------------------------
# Test no valid ontology files found (empty folder case)
# -----------------------------------------------------
def test_no_valid_ontology_files(empty_ontology_folder):
    """
    If no valid (non-hidden) files are found in the ontology folder,
    the constructor runs fine, but run_ontology_enrichment should fail.
    """
    workflow = OntologyWorkflow(ontology_folder=empty_ontology_folder, use_temp_files=False)
    
    # Attempt to run the enrichment with an empty folder
    with pytest.raises(ValueError, match="No valid ontology files found"):
        workflow.run_ontology_enrichment(
            summary_csv="dummy_summary.csv",
            gene_origin="dummy_gene_origin.txt",
            background_genes="dummy_background.txt",
            filter_csv="dummy_filter.csv",
            output_csv="dummy_output.csv"
        )


# ----------------------------------
# Tests for run_ontology_enrichment
# ----------------------------------
@patch("geneinsight.ontology.workflow.OntologyReader")
@patch("geneinsight.ontology.workflow.RAGModuleGSEAPY")
def test_run_ontology_enrichment(
    mock_rag_class,
    mock_ontology_reader_class,
    mock_ontology_workflow,
    mock_data_files
):
    """
    Test run_ontology_enrichment with minimal mocking.
    """
    # Mock the behavior of OntologyReader so that it doesn't fail reading dummy files
    mock_reader_instance = MagicMock()
    mock_ontology_reader_class.return_value = mock_reader_instance
    
    # Mock the RAGModuleGSEAPY instance and its get_top_documents method
    mock_rag_instance = MagicMock()
    mock_rag_class.return_value = mock_rag_instance
    mock_rag_instance.get_top_documents.return_value = (
        [1, 2, 3],  # top_results_indices
        {"extract_key": "extract_value"},  # extracted_items
        pd.DataFrame({"col": [1, 2, 3]}),  # enrichr_results
        pd.DataFrame({"col_filter": [4, 5]}),  # enrichr_df_filtered
        "some_formatted_output",
    )

    output_csv = os.path.join(str(Path(mock_data_files["summary_csv"]).parent), "results.csv")

    df_result = mock_ontology_workflow.run_ontology_enrichment(
        summary_csv=mock_data_files["summary_csv"],
        gene_origin=mock_data_files["gene_origin_txt"],
        background_genes=mock_data_files["background_txt"],
        filter_csv=mock_data_files["filter_csv"],
        output_csv=output_csv
    )

    # Check if the output CSV was created
    assert os.path.exists(output_csv), "Output CSV should be created by run_ontology_enrichment."

    # Validate the resulting DataFrame structure
    assert "query" in df_result.columns, "DataFrame should contain 'query' column."
    assert "top_results_indices" in df_result.columns
    assert "extracted_items" in df_result.columns
    assert "enrichr_results" in df_result.columns
    assert "enrichr_df_filtered" in df_result.columns
    assert "formatted_output" in df_result.columns

    # Check that the RAGModuleGSEAPY was called
    assert mock_rag_class.called, "RAGModuleGSEAPY should have been instantiated."
    assert mock_rag_instance.get_top_documents.called, "get_top_documents should have been called on the RAG instance."


def test_run_ontology_enrichment_no_background_file(mock_ontology_workflow, mock_data_files):
    """
    If background_genes is not a valid file path, it should simply skip loading the background genes.
    """
    non_existent_bg = "this_file_does_not_exist.txt"
    output_csv = os.path.join(str(Path(mock_data_files["summary_csv"]).parent), "results_no_bg.csv")

    # We don't mock the entire RAGModule because we're specifically testing the absent background file logic.
    # But we do need to ensure the ontology reading step won't fail. We'll patch the reading method.
    with patch("geneinsight.ontology.workflow.OntologyReader", MagicMock()), \
         patch("geneinsight.ontology.workflow.RAGModuleGSEAPY.get_top_documents", 
               return_value=([], {}, pd.DataFrame(), pd.DataFrame(), "")):
        df_result = mock_ontology_workflow.run_ontology_enrichment(
            summary_csv=mock_data_files["summary_csv"],
            gene_origin=mock_data_files["gene_origin_txt"],
            background_genes=non_existent_bg,
            filter_csv=mock_data_files["filter_csv"],
            output_csv=output_csv
        )

    assert os.path.exists(output_csv), "Output CSV should be created even with missing background file."
    assert isinstance(df_result, pd.DataFrame)


# ----------------------------------
# Tests for create_ontology_dictionary
# ----------------------------------
@patch("geneinsight.ontology.workflow.process_ontology_enrichment")
def test_create_ontology_dictionary(
    mock_process_ontology_enrichment,
    mock_ontology_workflow,
    mock_data_files
):
    """
    Test the create_ontology_dictionary method to ensure it calls
    process_ontology_enrichment and writes a CSV.
    """
    # Mock process_ontology_enrichment to return a small DataFrame
    mock_result_df = pd.DataFrame({"key": ["value1"], "another_key": ["value2"]})
    mock_process_ontology_enrichment.return_value = mock_result_df

    input_csv = mock_data_files["summary_csv"]
    output_csv = os.path.join(str(Path(input_csv).parent), "dictionary.csv")

    df_result = mock_ontology_workflow.create_ontology_dictionary(
        input_csv=input_csv,
        output_csv=output_csv
    )

    assert mock_process_ontology_enrichment.called, "process_ontology_enrichment should have been called."
    assert df_result.equals(mock_result_df), "Returned DataFrame should match the mocked DataFrame."

    # Check if the output CSV was created
    assert os.path.exists(output_csv), "Output CSV should be created by create_ontology_dictionary."


# -----------------------------
# Tests for process_dataframes
# -----------------------------
@patch("geneinsight.ontology.workflow.OntologyReader")
@patch("geneinsight.ontology.workflow.RAGModuleGSEAPY")
def test_process_dataframes(
    mock_rag_class,
    mock_ontology_reader_class,
    mock_ontology_workflow,
    tmp_path
):
    """
    Test process_dataframes with minimal data in memory.
    """
    # Mock the ontology reading
    mock_ontology_reader_class.return_value = MagicMock()

    # Mock the RAG module
    mock_rag_instance = MagicMock()
    mock_rag_class.return_value = mock_rag_instance
    mock_rag_instance.get_top_documents.return_value = (
        [1, 2, 3],
        {"extracted": "items"},
        pd.DataFrame({"results": [1, 2, 3]}),
        pd.DataFrame({"filtered": [4, 5, 6]}),
        "formatted_output"
    )

    # Prepare minimal DataFrames in memory
    summary_df = pd.DataFrame({
        "query": ["TermX", "TermY", "TermZ"],
        "unique_genes": [
            '{"GENE1":1}',  # string representation
            '{"GENE2":1}',
            '{"GENE3":1}'
        ]
    })
    clustered_df = pd.DataFrame({
        "Term": ["TermX", "TermY"]
    })

    # Create dummy gene list and background gene list
    gene_list = tmp_path / "gene_list.txt"
    gene_list.write_text("GENE1\nGENE2\nGENE3\n")

    background_genes = tmp_path / "background_list.txt"
    background_genes.write_text("BG1\nBG2\nBG3\n")

    output_dir = tmp_path / "results"

    enrichment_df, ontology_dict_df = mock_ontology_workflow.process_dataframes(
        summary_df=summary_df,
        clustered_df=clustered_df,
        gene_list_path=str(gene_list),
        background_genes_path=str(background_genes),
        output_dir=str(output_dir)
    )

    # Check the shape of the returned DataFrames
    assert not enrichment_df.empty, "Enrichment DataFrame should not be empty."
    assert not ontology_dict_df.empty, "Ontology dictionary DataFrame should not be empty."

    # Check if files were written to output_dir
    enrichment_csv = output_dir / "ontology_enrichment.csv"
    dict_csv = output_dir / "ontology_dict.csv"
    assert enrichment_csv.exists(), "Enrichment CSV should exist in output_dir."
    assert dict_csv.exists(), "Ontology dictionary CSV should exist in output_dir."

    # Check that get_top_documents was called for the right terms
    # Only TermX and TermY are in both dataframes
    calls = mock_rag_instance.get_top_documents.call_args_list
    assert len(calls) == 2, "Should only call get_top_documents for matching terms (TermX, TermY)."
    called_queries = [call[1]["query"] for call in calls]
    assert set(called_queries) == {"TermX", "TermY"}, "Should be called for TermX and TermY only."


def test_process_dataframes_no_matches(mock_ontology_workflow, tmp_path):
    """
    Test process_dataframes when there are no matching terms in summary_df and clustered_df.
    Should result in empty enrichment DataFrame.
    """
    summary_df = pd.DataFrame({
        "query": ["TermA", "TermB"],
        "unique_genes": [
            '{"GENE1":1}',
            '{"GENE2":1}'
        ]
    })
    # No overlapping terms with summary_df
    clustered_df = pd.DataFrame({
        "Term": ["TermX", "TermY"]
    })

    gene_list = tmp_path / "gene_list.txt"
    gene_list.write_text("GENE1\nGENE2\n")

    background_genes = tmp_path / "background_list.txt"
    background_genes.write_text("BG1\nBG2\n")

    enrichment_df, ontology_dict_df = mock_ontology_workflow.process_dataframes(
        summary_df=summary_df,
        clustered_df=clustered_df,
        gene_list_path=str(gene_list),
        background_genes_path=str(background_genes)
    )

    assert enrichment_df.empty, "Enrichment DataFrame should be empty if no matching terms."
    assert ontology_dict_df.empty, "Ontology dictionary DataFrame should also be empty."


# ----------------------------------------------
# Test using temp files for intermediate results
# ----------------------------------------------
def test_process_dataframes_with_temp_files(mock_ontology_folder, tmp_path):
    """
    Test process_dataframes using the temp file option.
    Ensures files are written to a temporary directory and cleaned up.
    """
    workflow = OntologyWorkflow(
        ontology_folder=mock_ontology_folder,
        fdr_threshold=0.1,
        use_temp_files=True
    )

    # Prepare minimal DataFrames in memory
    summary_df = pd.DataFrame({
        "query": ["TermAlpha"],
        "unique_genes": [
            '{"GENE9":1}'
        ]
    })
    clustered_df = pd.DataFrame({"Term": ["TermAlpha"]})

    # Create dummy gene list and background gene list
    gene_list = tmp_path / "gene_list.txt"
    gene_list.write_text("GENE9\n")

    background_genes = tmp_path / "background_list.txt"
    background_genes.write_text("BG1\n")

    # Patch out the RAGModule so we don't fail on actual ontology reading
    with patch("geneinsight.ontology.workflow.OntologyReader", MagicMock()), \
         patch("geneinsight.ontology.workflow.RAGModuleGSEAPY.get_top_documents",
               return_value=([], {}, pd.DataFrame(), pd.DataFrame(), "")), \
         patch("shutil.rmtree") as mock_rmtree:

        enrichment_df, ontology_dict_df = workflow.process_dataframes(
            summary_df=summary_df,
            clustered_df=clustered_df,
            gene_list_path=str(gene_list),
            background_genes_path=str(background_genes)
        )

        # The code should attempt to remove the temp dir
        assert mock_rmtree.called, "Temporary directory should be cleaned up."
        assert isinstance(enrichment_df, pd.DataFrame), "Should still return DataFrame objects."
        assert isinstance(ontology_dict_df, pd.DataFrame), "Should still return DataFrame objects."


# ---------------------------
# Tests for run_full_workflow
# ---------------------------
@patch("geneinsight.ontology.workflow.OntologyWorkflow.run_ontology_enrichment")
@patch("geneinsight.ontology.workflow.OntologyWorkflow.create_ontology_dictionary")
def test_run_full_workflow(
    mock_create_dictionary,
    mock_run_enrichment,
    mock_ontology_workflow,
    mock_data_files
):
    """
    Test the end-to-end workflow.
    """
    # Mock the return values of the two main methods
    enrichment_df = pd.DataFrame({"test": [1, 2, 3]})
    dictionary_df = pd.DataFrame({"dict_test": [10, 20, 30]})
    
    mock_run_enrichment.return_value = enrichment_df
    mock_create_dictionary.return_value = dictionary_df

    # Now call run_full_workflow
    gene_set_name = "test_geneset"
    result = mock_ontology_workflow.run_full_workflow(
        gene_set=gene_set_name,
        summary_csv=mock_data_files["summary_csv"],
        gene_origin=mock_data_files["gene_origin_txt"],
        background_genes=mock_data_files["background_txt"],
        filter_csv=mock_data_files["filter_csv"],
        output_dir="my_results",
        return_results=True
    )

    # Ensure the main steps were called
    mock_run_enrichment.assert_called_once()
    mock_create_dictionary.assert_called_once()
    
    # Check that the returned dictionary is correct
    assert isinstance(result, dict)
    assert "enrichment" in result
    assert "dictionary" in result
    assert result["enrichment"].equals(enrichment_df)
    assert result["dictionary"].equals(dictionary_df)


def test_run_full_workflow_defaults(mock_ontology_workflow):
    """
    If no paths are provided to run_full_workflow, check that it uses 
    default file paths based on the gene_set name and output_dir.
    We'll just ensure it doesn't raise an error when defaults are used.
    """
    # Patch out the actual calls to avoid needing real files
    with patch.object(mock_ontology_workflow, "run_ontology_enrichment", return_value=pd.DataFrame()), \
         patch.object(mock_ontology_workflow, "create_ontology_dictionary", return_value=pd.DataFrame()):
        try:
            mock_ontology_workflow.run_full_workflow(gene_set="DefaultTest")
        except Exception as e:
            pytest.fail(f"run_full_workflow raised an exception with defaults: {e}")
