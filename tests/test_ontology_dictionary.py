# test_ontology_dictionary.py
import pytest
import pandas as pd
import os
import json
from tempfile import NamedTemporaryFile
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
