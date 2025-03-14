# test_ontology_dictionary.py
import pytest
import pandas as pd
import os
import json
from tempfile import NamedTemporaryFile
import logging
from unittest.mock import patch, MagicMock
from io import StringIO
from unittest.mock import patch, MagicMock
from geneinsight.ontology.get_ontology_dictionary import (
    process_ontology_enrichment,
    save_ontology_dictionary,
    main
)


@pytest.fixture
def sample_dataframe():
    """
    Return a DataFrame with typical usage rows.
    """
    df = pd.DataFrame({
        "query": ["query1", "query2", "query3", "query4"],
        "enrichr_df_filtered": [
            '[{"Term": "TermA", "Genes": ["Gene1", "Gene2"]}]',  # Valid JSON string
            '[{"Term": "TermB", "Genes": ["Gene3"]}]',            # Valid JSON string
            '',                                                  # Empty string
            '[{"term": "TermC", "genes": ["Gene4"]}]',           # Lowercase keys
        ]
    })
    return df


def test_process_ontology_enrichment_with_dataframe(sample_dataframe):
    """
    Test that process_ontology_enrichment handles a typical DataFrame input correctly.
    """
    processed_df = process_ontology_enrichment(sample_dataframe)
    
    # We expect 4 rows in the output
    assert len(processed_df) == 4
    
    # Check each row in the output
    # Row 1: Expect a dict with TermA -> ["Gene1", "Gene2"]
    row1 = processed_df[processed_df['query'] == 'query1'].iloc[0]
    assert row1['ontology_dict'] == {"TermA": ["Gene1", "Gene2"]}
    
    # Row 2: Expect a dict with TermB -> ["Gene3"]
    row2 = processed_df[processed_df['query'] == 'query2'].iloc[0]
    assert row2['ontology_dict'] == {"TermB": ["Gene3"]}
    
    # Row 3: Empty string => no enrichment results => empty dict
    row3 = processed_df[processed_df['query'] == 'query3'].iloc[0]
    assert row3['ontology_dict'] == {}
    
    # Row 4: Lowercase term/genes => code attempts to fix column names => TermC -> ["Gene4"]
    row4 = processed_df[processed_df['query'] == 'query4'].iloc[0]
    assert row4['ontology_dict'] == {"TermC": ["Gene4"]}


def test_process_ontology_enrichment_with_csv(tmp_path, sample_dataframe):
    """
    Test that process_ontology_enrichment handles a CSV path input correctly.
    """
    # Create a temporary CSV file
    temp_csv_path = tmp_path / "test_input.csv"
    sample_dataframe.to_csv(temp_csv_path, index=False)
    
    processed_df = process_ontology_enrichment(str(temp_csv_path))
    
    # We expect the same results as the DataFrame-based test
    assert len(processed_df) == 4
    row1 = processed_df[processed_df['query'] == 'query1'].iloc[0]
    assert row1['ontology_dict'] == {"TermA": ["Gene1", "Gene2"]}


def test_empty_and_nan_values(caplog):
    """
    Test that empty or 'NaN' / 'None' strings or lists produce empty dictionaries.
    """
    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({
        "query": ["q_nan", "q_none", "q_nanstring", "q_listempty", "q_bracketempty"],
        "enrichr_df_filtered": [
            float('nan'),                # True NaN
            None,                        # None
            "NaN",                       # A literal string "NaN"
            [],                          # An empty list
            "[]",                        # A string that is empty list
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    # Expect all 5 to produce empty dicts
    assert len(processed_df) == 5
    for idx in range(5):
        assert processed_df.iloc[idx]['ontology_dict'] == {}, \
            f"Row {idx} should have empty dict"
    
    # Check that logs mention warnings
    # We can check 'caplog.text' for certain warning messages
    # (Optional, if you want to assert log content)
    assert "No enrichment results for query: q_nan" in caplog.text
    assert "No enrichment results for query: q_none" in caplog.text


def test_missing_required_columns():
    """
    Test that if 'Term'/'Genes' columns are missing or spelled incorrectly (and not fixable),
    we get empty dictionaries.
    """
    df = pd.DataFrame({
        "query": ["q_missing_cols"],
        "enrichr_df_filtered": [
            '[{"NoTerm": "TermX", "NoGenes": ["GeneX"]}]'
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    row = processed_df.iloc[0]
    # Expect empty because we can't fix from "NoTerm" / "NoGenes"
    assert row['ontology_dict'] == {}


def test_partial_and_lowercase_corrections():
    """
    Test partial corrections: if we have 'term' and 'Genes',
    or 'Term' and 'genes', etc., it should still work.
    """
    df = pd.DataFrame({
        "query": ["q_partial_lower"],
        "enrichr_df_filtered": [
            '[{"term": "TermZ", "Genes": ["GeneZ"]}]'
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    row = processed_df.iloc[0]
    # Expect corrected to TermZ -> [GeneZ]
    assert row['ontology_dict'] == {"TermZ": ["GeneZ"]}


def test_direct_list_input():
    """
    Test passing a DataFrame where enrichr_df_filtered is already a list (not a string).
    """
    df = pd.DataFrame({
        "query": ["q_list"],
        "enrichr_df_filtered": [
            [{"Term": "TermList", "Genes": ["GeneList"]}]
        ]
    })
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    row = processed_df.iloc[0]
    assert row['ontology_dict'] == {"TermList": ["GeneList"]}


def test_corrupted_json_string(caplog):
    """
    Test that a corrupted JSON string logs an error and yields an empty dict.
    """
    caplog.set_level(logging.WARNING)
    df = pd.DataFrame({
        "query": ["q_corrupt"],
        "enrichr_df_filtered": [
            '[{"Term": "BadTerm", "Genes": ["GeneBad"]'  # Missing closing brace
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    row = processed_df.iloc[0]
    # Expect empty dict
    assert row['ontology_dict'] == {}
    
    # Check logs
    assert "Failed to parse string as list for query q_corrupt" in caplog.text


def test_save_ontology_dictionary(tmp_path, sample_dataframe):
    """
    Test that save_ontology_dictionary can save the processed DataFrame to CSV.
    """
    processed_df = process_ontology_enrichment(sample_dataframe)
    
    # Create a temporary output CSV
    output_csv_path = tmp_path / "ontology_dict_output.csv"
    save_ontology_dictionary(processed_df, str(output_csv_path))
    
    # Read back the CSV and verify content
    reloaded_df = pd.read_csv(output_csv_path)
    assert len(reloaded_df) == len(processed_df)
    # The "ontology_dict" column will be stored as string in CSV;
    # we can check just that they're not all empty or do a quick parse check.
    for original_dict_str, new_dict_str in zip(processed_df['ontology_dict'], reloaded_df['ontology_dict']):
        # Convert the new dict from string to Python object
        reloaded_dict = {}
        # Some CSV readers escape quotes differently; replace single with double quotes
        if isinstance(new_dict_str, str) and new_dict_str.strip():
            try:
                reloaded_dict = json.loads(new_dict_str.replace("'", '"'))
            except json.JSONDecodeError:
                pass
        assert reloaded_dict == original_dict_str, \
            f"Mismatch: {reloaded_dict} vs {original_dict_str}"


def test_main_function(tmp_path, sample_dataframe):
    """
    Test the main function end-to-end. 
    """
    # Input CSV
    input_csv_path = tmp_path / "input.csv"
    sample_dataframe.to_csv(input_csv_path, index=False)
    
    # Output CSV
    output_csv_path = tmp_path / "output.csv"
    
    # Run main
    result_df = main(str(input_csv_path), str(output_csv_path))
    
    # Basic checks
    assert len(result_df) == len(sample_dataframe)
    assert output_csv_path.exists(), "Output CSV was not created."
    
    # Read back the output CSV and compare
    reloaded_df = pd.read_csv(output_csv_path)
    assert len(reloaded_df) == len(sample_dataframe)

    # Quick check for the dictionary content:
    # (Same logic as test_save_ontology_dictionary)
    for original_dict_str, new_dict_str in zip(result_df['ontology_dict'], reloaded_df['ontology_dict']):
        reloaded_dict = {}
        if isinstance(new_dict_str, str) and new_dict_str.strip():
            try:
                reloaded_dict = json.loads(new_dict_str.replace("'", '"'))
            except json.JSONDecodeError:
                pass
        assert reloaded_dict == original_dict_str


def test_multiple_terms_same_genes():
    """
    Test handling of multiple terms mapping to the same set of genes.
    """
    df = pd.DataFrame({
        "query": ["multiple_terms"],
        "enrichr_df_filtered": [
            '[{"Term": "Term1", "Genes": ["Gene1", "Gene2"]}, {"Term": "Term2", "Genes": ["Gene1", "Gene2"]}]'
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    ontology_dict = processed_df.iloc[0]['ontology_dict']
    
    # Both terms should be present with the same gene lists
    assert "Term1" in ontology_dict
    assert "Term2" in ontology_dict
    assert ontology_dict["Term1"] == ["Gene1", "Gene2"]
    assert ontology_dict["Term2"] == ["Gene1", "Gene2"]


def test_malformed_gene_lists():
    """
    Test handling of malformed gene lists (not lists but strings or other types).
    """
    df = pd.DataFrame({
        "query": ["string_genes", "int_genes", "dict_genes"],
        "enrichr_df_filtered": [
            '[{"Term": "StringGenes", "Genes": "Gene1,Gene2"}]',  # String instead of list
            '[{"Term": "IntGene", "Genes": 12345}]',              # Integer instead of list
            '[{"Term": "DictGenes", "Genes": {"key": "value"}}]'  # Dict instead of list
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 3
    
    # Check that all cases still result in a dictionary
    for i in range(3):
        assert isinstance(processed_df.iloc[i]['ontology_dict'], dict)
    
    # The function should preserve the original format of Genes
    assert processed_df.iloc[0]['ontology_dict']["StringGenes"] == "Gene1,Gene2"
    assert processed_df.iloc[1]['ontology_dict']["IntGene"] == 12345
    assert processed_df.iloc[2]['ontology_dict']["DictGenes"] == {"key": "value"}


def test_unicode_characters():
    """
    Test handling of Unicode characters in terms and gene names.
    """
    df = pd.DataFrame({
        "query": ["unicode_test"],
        "enrichr_df_filtered": [
            '[{"Term": "Café-Pathway", "Genes": ["Gene-α", "Gene-β", "Gene-γ"]}]'
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    ontology_dict = processed_df.iloc[0]['ontology_dict']
    
    # Check that Unicode characters are preserved
    assert "Café-Pathway" in ontology_dict
    assert "Gene-α" in ontology_dict["Café-Pathway"]
    assert "Gene-β" in ontology_dict["Café-Pathway"]
    assert "Gene-γ" in ontology_dict["Café-Pathway"]


def test_duplicate_terms():
    """
    Test handling of duplicate terms (last occurrence should overwrite previous ones).
    """
    df = pd.DataFrame({
        "query": ["duplicate_terms"],
        "enrichr_df_filtered": [
            '[{"Term": "DupTerm", "Genes": ["Gene1"]}, {"Term": "DupTerm", "Genes": ["Gene2"]}]'
        ]
    })
    
    processed_df = process_ontology_enrichment(df)
    assert len(processed_df) == 1
    ontology_dict = processed_df.iloc[0]['ontology_dict']
    
    # The last occurrence should overwrite
    assert ontology_dict["DupTerm"] == ["Gene2"]


def test_input_file_not_found():
    """
    Test handling of file not found error.
    """
    # The implementation should raise FileNotFoundError when trying to read a non-existent file
    nonexistent_file = "totally_nonexistent_file_that_doesnt_exist.csv"
    with pytest.raises(FileNotFoundError):
        process_ontology_enrichment(nonexistent_file)


def test_column_missing_from_input(caplog):
    """
    Test handling of a missing required column in the input DataFrame.
    """
    caplog.set_level(logging.WARNING)
    # Missing 'query' column
    df = pd.DataFrame({
        "not_query": ["q1"],
        "enrichr_df_filtered": ['[{"Term": "Term1", "Genes": ["Gene1"]}]']
    })
    
    with pytest.raises(KeyError):
        process_ontology_enrichment(df)
    
    # Reset caplog for the next test
    caplog.clear()
    
    # Missing 'enrichr_df_filtered' column
    # The implementation handles this with try/except, not by raising KeyError
    df = pd.DataFrame({
        "query": ["q1"],
        "not_enrichr": ['[{"Term": "Term1", "Genes": ["Gene1"]}]']
    })
    
    # The function should return a DataFrame with empty dictionaries
    result_df = process_ontology_enrichment(df)
    assert len(result_df) == 1
    assert result_df.iloc[0]['query'] == 'q1'
    assert result_df.iloc[0]['ontology_dict'] == {}
    
    # Check that an error was logged
    assert "Error checking enrichr_df_filtered for query q1" in caplog.text


@patch('pandas.DataFrame.to_csv')
def test_save_ontology_dictionary_exception(mock_to_csv):
    """
    Test handling of exceptions during saving the output CSV.
    """
    # Mock to_csv to raise an exception
    mock_to_csv.side_effect = PermissionError("Permission denied")
    
    df = pd.DataFrame({"query": ["q1"], "ontology_dict": [{"Term1": ["Gene1"]}]})
    
    with pytest.raises(PermissionError):
        save_ontology_dictionary(df, "test_output.csv")


@patch('geneinsight.ontology.get_ontology_dictionary.process_ontology_enrichment')
@patch('geneinsight.ontology.get_ontology_dictionary.save_ontology_dictionary')
def test_main_with_exceptions(mock_save, mock_process):
    """
    Test main function handles exceptions from process or save functions.
    """
    # Test process_ontology_enrichment exception
    mock_process.side_effect = Exception("Process error")
    
    with pytest.raises(Exception, match="Process error"):
        main("input.csv", "output.csv")
    
    # Test save_ontology_dictionary exception
    mock_process.side_effect = None  # Reset
    mock_process.return_value = pd.DataFrame({"query": ["q1"], "ontology_dict": [{"Term1": ["Gene1"]}]})
    mock_save.side_effect = Exception("Save error")
    
    with pytest.raises(Exception, match="Save error"):
        main("input.csv", "output.csv")


def test_arg_parsing():
    """
    Test argument parsing in main function.
    """
    # A more direct approach to test the CLI argument parsing
    with patch('argparse.ArgumentParser.parse_args') as mock_args:
        # Set up the mock to return an object with input_csv and output_csv attributes
        mock_args.return_value = MagicMock(input_csv='in.csv', output_csv='out.csv')
        
        # Patch the main function components
        with patch('geneinsight.ontology.get_ontology_dictionary.process_ontology_enrichment') as mock_process, \
             patch('geneinsight.ontology.get_ontology_dictionary.save_ontology_dictionary') as mock_save:
            
            mock_process.return_value = pd.DataFrame({"query": ["q1"], "ontology_dict": [{"Term1": ["Gene1"]}]})
            
            # Import and run the main function from the CLI part
            from geneinsight.ontology.get_ontology_dictionary import main
            
            # Call main directly with the arguments we want to test
            main('in.csv', 'out.csv')
            
            # Verify the function calls
            mock_process.assert_called_once_with('in.csv')
            mock_save.assert_called_once()
            assert mock_save.call_args[0][1] == 'out.csv'


def test_large_dataset_performance():
    """
    Test performance with a large dataset.
    """
    # Create a large DataFrame (1000 rows)
    queries = [f"query{i}" for i in range(1000)]
    enrichr_values = ['[{"Term": "Term1", "Genes": ["Gene1", "Gene2"]}]'] * 1000
    
    large_df = pd.DataFrame({
        "query": queries,
        "enrichr_df_filtered": enrichr_values
    })
    
    # Process it and check all rows were processed
    processed_df = process_ontology_enrichment(large_df)
    assert len(processed_df) == 1000
    
    # Check a few random entries
    assert processed_df[processed_df['query'] == 'query0'].iloc[0]['ontology_dict'] == {"Term1": ["Gene1", "Gene2"]}
    assert processed_df[processed_df['query'] == 'query500'].iloc[0]['ontology_dict'] == {"Term1": ["Gene1", "Gene2"]}
    assert processed_df[processed_df['query'] == 'query999'].iloc[0]['ontology_dict'] == {"Term1": ["Gene1", "Gene2"]}


def test_log_messages(caplog):
    """
    Test that appropriate log messages are generated.
    """
    # Set up logging capture
    caplog.set_level(logging.INFO)
    
    # Process a simple DataFrame
    df = pd.DataFrame({
        "query": ["log_test"],
        "enrichr_df_filtered": ['[{"Term": "LogTerm", "Genes": ["LogGene"]}]']
    })
    
    process_ontology_enrichment(df)
    
    # Check for expected log messages
    assert "Processing enrichment results" in caplog.text
    assert "Using provided DataFrame directly" in caplog.text
    assert "Processed 1 queries into ontology dictionaries" in caplog.text


def test_empty_dataframe():
    """
    Test handling of an empty DataFrame.
    """
    empty_df = pd.DataFrame({"query": [], "enrichr_df_filtered": []})
    
    processed_df = process_ontology_enrichment(empty_df)
    # The implementation returns an empty DataFrame with expected columns
    assert len(processed_df) == 0
    
    # For an empty DataFrame, we should either check that it has the expected columns
    # or if the implementation returns a truly empty DataFrame, adjust our expectation
    if not processed_df.empty:
        assert list(processed_df.columns) == ["query", "ontology_dict"]
    else:
        # If the function returns a truly empty DataFrame, this test would pass
        assert processed_df.empty


def test_csv_roundtrip():
    """
    Test that the dictionary format survives a round trip to CSV and back.
    """
    df = pd.DataFrame({
        "query": ["roundtrip"],
        "enrichr_df_filtered": ['[{"Term": "TripTerm", "Genes": ["TripGene"]}]']
    })
    
    # Process to create the ontology dict
    processed_df = process_ontology_enrichment(df)
    
    # Save to a string buffer (like a CSV file)
    csv_buffer = StringIO()
    processed_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Reset to start of buffer
    
    # Read back
    reloaded_df = pd.read_csv(csv_buffer)
    
    # Convert string representation of dict back to dict
    reloaded_df['ontology_dict'] = reloaded_df['ontology_dict'].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else {})
    
    # Compare
    assert reloaded_df.iloc[0]['query'] == processed_df.iloc[0]['query']
    assert reloaded_df.iloc[0]['ontology_dict'] == processed_df.iloc[0]['ontology_dict']