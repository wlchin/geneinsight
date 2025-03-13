# test_stringdb.py

import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from io import StringIO

# Import your module functions
from geneinsight.enrichment.stringdb import (
    read_gene_list,
    get_string_output,
    query_string_db_individual_genes,
    process_gene_enrichment
)

@pytest.fixture
def gene_list_file(tmp_path):
    """
    Creates a temporary CSV file containing a list of genes.
    Returns the path to that file.
    """
    data = ["GeneA", "GeneB", "GeneC"]
    file_path = tmp_path / "genes.csv"
    file_path.write_text("\n".join(data))
    return file_path

@pytest.fixture
def empty_gene_list_file(tmp_path):
    """
    Creates a temporary CSV file that is empty (no genes).
    """
    file_path = tmp_path / "empty_genes.csv"
    file_path.write_text("")
    return file_path

@pytest.fixture
def output_dir(tmp_path):
    """
    Creates a temporary directory for storing outputs.
    """
    d = tmp_path / "output"
    d.mkdir()
    return d

#
# Tests for read_gene_list
#
def test_read_gene_list(gene_list_file):
    """
    Test that read_gene_list correctly reads in a list of genes
    from a file with no header.
    """
    genes = read_gene_list(str(gene_list_file))
    assert genes == ["GeneA", "GeneB", "GeneC"], "Should correctly read list of genes."


def test_read_gene_list_empty_file(empty_gene_list_file):
    """
    Test that read_gene_list returns an empty list if file is empty.
    """
    genes = read_gene_list(str(empty_gene_list_file))
    assert genes == [], "Should return empty list when file is empty."


#
# Tests for get_string_output
#
@patch("geneinsight.enrichment.stringdb.stringdb.get_string_ids")
@patch("geneinsight.enrichment.stringdb.stringdb.get_enrichment")
def test_get_string_output(mock_get_enrichment, mock_get_string_ids):
    """
    Test that get_string_output calls stringdb API functions and
    returns the expected DataFrame and documents list.
    """
    # Mock responses
    mock_get_string_ids.return_value = pd.DataFrame({
        "queryItem": ["ID1", "ID2", "ID3"]
    })
    mock_get_enrichment.return_value = pd.DataFrame({
        "description": ["Term1", "Term2", "Term1", "Term3"],
        "some_other_column": [1, 2, 3, 4]
    })

    test_genes = ["GeneA", "GeneB", "GeneC"]
    df, documents = get_string_output(test_genes)

    # Assertions
    assert mock_get_string_ids.called, "Should call get_string_ids at least once"
    assert mock_get_enrichment.called, "Should call get_enrichment at least once"
    assert not df.empty, "Returned DataFrame should not be empty"
    assert "description" in df.columns, "DataFrame should have description column"
    assert documents == ["Term1", "Term2", "Term3"], "Should capture unique document descriptions"


#
# Tests for query_string_db_individual_genes
#
@patch("geneinsight.enrichment.stringdb.stringdb.get_string_ids")
@patch("geneinsight.enrichment.stringdb.stringdb.get_enrichment")
@patch("geneinsight.enrichment.stringdb.time.sleep", return_value=None)
def test_query_string_db_individual_genes_success(
    mock_sleep,
    mock_get_enrichment,
    mock_get_string_ids,
    tmp_path
):
    """
    Test query_string_db_individual_genes with all successful requests.
    Ensures a proper DataFrame and documents list are returned.
    """
    mock_get_string_ids.side_effect = [
        pd.DataFrame({"queryItem": ["ID-GeneA"]}),
        pd.DataFrame({"queryItem": ["ID-GeneB"]}),
        pd.DataFrame({"queryItem": ["ID-GeneC"]}),
    ]
    mock_get_enrichment.side_effect = [
        pd.DataFrame({"description": ["TermA"]}),
        pd.DataFrame({"description": ["TermB"]}),
        pd.DataFrame({"description": ["TermC"]}),
    ]

    genes = ["GeneA", "GeneB", "GeneC"]
    log_file = str(tmp_path / "bad_requests.log")
    df, docs = query_string_db_individual_genes(genes, log_file)

    assert len(df) == 3, "Should have one row per gene in the returned DataFrame"
    assert "gene_queried" in df.columns, "Returned DataFrame should contain 'gene_queried' column"
    assert docs == ["TermA", "TermB", "TermC"], "Should capture the unique descriptions"
    assert not os.path.exists(log_file), "Log file should not be created if no errors occurred"


@patch("geneinsight.enrichment.stringdb.stringdb.get_string_ids")
@patch("geneinsight.enrichment.stringdb.stringdb.get_enrichment")
@patch("geneinsight.enrichment.stringdb.time.sleep", return_value=None)
def test_query_string_db_individual_genes_error_handling(
    mock_sleep,
    mock_get_enrichment,
    mock_get_string_ids,
    tmp_path
):
    """
    Test query_string_db_individual_genes with an exception thrown for one gene.
    Ensures the bad gene is logged, partial DataFrame is returned, and coverage for error branch.
    """
    # Suppose "GeneB" triggers an exception
    def side_effect_get_string_ids(gene_list, species=9606):
        # Check if this is the problematic gene
        if gene_list == ["GeneB"]:
            raise ValueError("Test Exception for GeneB")
        return pd.DataFrame({"queryItem": [f"ID-{gene_list[0]}"]})

    mock_get_string_ids.side_effect = side_effect_get_string_ids
    mock_get_enrichment.return_value = pd.DataFrame({"description": ["TermX"]})

    genes = ["GeneA", "GeneB", "GeneC"]
    log_file = str(tmp_path / "bad_requests.log")
    df, docs = query_string_db_individual_genes(genes, log_file)

    # We should have 2 successful calls and 1 failure
    assert len(df) == 2, "Should contain data for the successful genes only"
    assert "GeneB" not in df["gene_queried"].values, "GeneB should not appear in the successful DataFrame"
    assert docs == ["TermX"], "Unique documents from the successful queries only"

    # Check the log file for the bad gene
    assert os.path.exists(log_file), "Log file should be created because there was a bad gene"
    with open(log_file, "r") as f:
        content = f.read().strip()
    assert "GeneB" in content, "Bad gene should be logged"


def test_query_string_db_individual_genes_no_genes(tmp_path):
    """
    Test query_string_db_individual_genes when no genes are passed in.
    Should return an empty DataFrame and empty documents list.
    """
    df, docs = query_string_db_individual_genes([], str(tmp_path / "bad_requests.log"))
    assert df.empty, "Returned DataFrame should be empty if no genes are queried"
    assert docs == [], "Returned documents list should be empty if no genes are queried"


#
# Tests for process_gene_enrichment
#
@patch("geneinsight.enrichment.stringdb.get_string_output")
def test_process_gene_enrichment_list_mode(mock_get_string_output, gene_list_file, output_dir):
    """
    Test process_gene_enrichment with mode='list'.
    It should call get_string_output and create two CSV files.
    """
    # Mock the return value from get_string_output
    mock_get_string_output.return_value = (
        pd.DataFrame({"description": ["Term1", "Term2"]}),
        ["Term1", "Term2"]
    )

    df, docs = process_gene_enrichment(
        input_file=str(gene_list_file),
        output_dir=str(output_dir),
        mode="list"
    )

    # Check calls
    mock_get_string_output.assert_called_once()

    # Check returned values
    assert not df.empty, "Should return a non-empty DataFrame in list mode"
    assert docs == ["Term1", "Term2"], "Should return correct documents"

    # Check output CSV files
    basename = os.path.splitext(gene_list_file.name)[0]
    enrichment_csv = output_dir / f"{basename}__enrichment.csv"
    documents_csv = output_dir / f"{basename}__documents.csv"

    assert enrichment_csv.exists(), "Enrichment CSV file should be created"
    assert documents_csv.exists(), "Documents CSV file should be created"

    # Optional: Read them back to ensure correctness
    enrichment_df = pd.read_csv(enrichment_csv)
    assert list(enrichment_df.columns) == ["description"], "Enrichment CSV columns should match mock"
    documents_df = pd.read_csv(documents_csv)
    assert list(documents_df["description"]) == ["Term1", "Term2"], "Documents CSV should match the docs list"


@patch("geneinsight.enrichment.stringdb.query_string_db_individual_genes")
def test_process_gene_enrichment_single_mode(
    mock_query_string_db_individual_genes,
    gene_list_file,
    output_dir
):
    """
    Test process_gene_enrichment with mode='single'.
    It should call query_string_db_individual_genes and create two CSV files.
    """
    mock_query_string_db_individual_genes.return_value = (
        pd.DataFrame({"description": ["TermA"], "gene_queried": ["GeneA"]}),
        ["TermA"]
    )

    df, docs = process_gene_enrichment(
        input_file=str(gene_list_file),
        output_dir=str(output_dir),
        mode="single"
    )

    # Check calls
    mock_query_string_db_individual_genes.assert_called_once()

    # Check returned values
    assert not df.empty, "Should return a non-empty DataFrame in single mode"
    assert docs == ["TermA"], "Should return correct documents"

    # Check output CSV files
    basename = os.path.splitext(gene_list_file.name)[0]
    enrichment_csv = output_dir / f"{basename}__enrichment.csv"
    documents_csv = output_dir / f"{basename}__documents.csv"

    assert enrichment_csv.exists(), "Enrichment CSV file should be created"
    assert documents_csv.exists(), "Documents CSV file should be created"

    # Optional: Read them back to ensure correctness
    enrichment_df = pd.read_csv(enrichment_csv)
    assert "description" in enrichment_df.columns, "Enrichment CSV should contain description column"
    assert "gene_queried" in enrichment_df.columns, "Enrichment CSV should contain gene_queried column"
    documents_df = pd.read_csv(documents_csv)
    assert list(documents_df["description"]) == ["TermA"], "Documents CSV should match the docs list"


def test_process_gene_enrichment_empty_genes(empty_gene_list_file, output_dir):
    """
    Test process_gene_enrichment with an empty gene list file. Should create empty CSVs.
    """
    df, docs = process_gene_enrichment(
        input_file=str(empty_gene_list_file),
        output_dir=str(output_dir),
        mode="list"  # or 'single', doesn't matter for an empty file
    )

    assert df.empty, "Returned DataFrame should be empty for empty input"
    assert docs == [], "Returned documents should be empty for empty input"

    # Check output CSV files
    basename = os.path.splitext(empty_gene_list_file.name)[0]
    enrichment_csv = output_dir / f"{basename}__enrichment.csv"
    documents_csv = output_dir / f"{basename}__documents.csv"

    assert enrichment_csv.exists(), "Should still create the enrichment CSV"
    assert documents_csv.exists(), "Should still create the documents CSV"

    enrichment_df = pd.read_csv(enrichment_csv)
    assert enrichment_df.empty, "Enrichment CSV should be empty"
    documents_df = pd.read_csv(documents_csv)
    assert documents_df.empty, "Documents CSV should be empty"
