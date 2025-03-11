import pytest
import pandas as pd
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock

# Import the module to test
import geneinsight.analysis.counter as tc

@pytest.fixture
def sample_data():
    """Fixture for test data"""
    return pd.DataFrame({
        'Document': ['apple', 'banana', 'apple', 'orange', 'apple', 'banana'],
        'Representative_document': [True, True, True, False, True, False],
        'Probability': [0.8, 0.7, 0.9, 0.4, 0.6, 0.3]
    })

@pytest.fixture
def temp_files():
    """Fixture for temporary files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, 'test_input.csv')
        output_file = os.path.join(temp_dir, 'test_output.csv')
        yield {'input': input_file, 'output': output_file, 'dir': temp_dir}

def test_count_strings_most_common():
    """Test count_strings_most_common function"""
    # Test with normal list
    test_list = ['apple', 'banana', 'apple', 'orange', 'apple', 'banana']
    result = tc.count_strings_most_common(test_list)
    
    assert len(result) == 3
    assert result[0] == ('apple', 3)
    assert result[1] == ('banana', 2)
    assert result[2] == ('orange', 1)
    
    # Test with empty list
    with patch('logging.Logger.warning') as mock_warning:
        result = tc.count_strings_most_common([])
        mock_warning.assert_called_once()
        assert result == []
    
    # Test with top_n
    result = tc.count_strings_most_common(test_list, top_n=2)
    assert len(result) == 2
    assert result[0] == ('apple', 3)
    assert result[1] == ('banana', 2)

def test_count_top_terms_standard_columns(sample_data, temp_files):
    """Test count_top_terms with standard column names"""
    # Save sample data to input file
    sample_data.to_csv(temp_files['input'], index=False)
    
    # Call count_top_terms
    result_df = tc.count_top_terms(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Check if output file was created
    assert os.path.exists(temp_files['output'])
    
    # Check the results
    assert len(result_df) == 2
    assert result_df.iloc[0]['Term'] == 'apple'
    assert result_df.iloc[0]['Count'] == 3
    assert result_df.iloc[1]['Term'] == 'banana'
    assert result_df.iloc[1]['Count'] == 1
    
    # Check with top_n
    result_df = tc.count_top_terms(
        input_file=temp_files['input'],
        output_file=temp_files['output'],
        top_n=1
    )
    
    assert len(result_df) == 1
    assert result_df.iloc[0]['Term'] == 'apple'


def test_count_top_terms_description_column(temp_files):
    """Test count_top_terms with description column"""
    # Create data with description column
    desc_data = pd.DataFrame({
        'description': ['apple', 'banana', 'apple', 'orange', 'apple', 'banana'],
        'Probability': [0.8, 0.7, 0.9, 0.4, 0.6, 0.3]
    })
    desc_data.to_csv(temp_files['input'], index=False)
    
    # Call count_top_terms
    result_df = tc.count_top_terms(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Check the results
    assert len(result_df) > 0

def test_count_top_terms_no_probability(temp_files):
    """Test count_top_terms with no Probability column"""
    # Create data with no Probability column
    no_prob_data = pd.DataFrame({
        'Term': ['apple', 'banana', 'apple', 'orange', 'apple', 'banana'],
    })
    no_prob_data.to_csv(temp_files['input'], index=False)
    
    # Call count_top_terms
    result_df = tc.count_top_terms(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Should default to considering all documents representative
    assert len(result_df) > 0
    assert result_df.iloc[0]['Term'] == 'apple'
    assert result_df.iloc[0]['Count'] == 3

def test_count_top_terms_no_suitable_column(temp_files):
    """Test count_top_terms with no suitable document column"""
    # Create data with no suitable document column
    bad_data = pd.DataFrame({
        'Irrelevant': ['apple', 'banana', 'apple'],
        'Other': [1, 2, 3]
    })
    bad_data.to_csv(temp_files['input'], index=False)
    
    # Call count_top_terms
    with patch('logging.Logger.error') as mock_error:
        result_df = tc.count_top_terms(
            input_file=temp_files['input'],
            output_file=temp_files['output']
        )
    
    # Should log an error about missing document column
    mock_error.assert_called_once()
    assert "Could not find a suitable document column" in mock_error.call_args[0][0]
    
    # Should return an empty DataFrame
    assert result_df.empty

def test_count_top_terms_file_not_found():
    """Test count_top_terms with non-existent input file"""
    with patch('logging.Logger.error') as mock_error:
        result_df = tc.count_top_terms(
            input_file='non_existent_file.csv',
            output_file='output.csv'
        )
    
    # Should log an error
    mock_error.assert_called_once()
    assert "Error counting top terms" in mock_error.call_args[0][0]
    
    # Should return an empty DataFrame
    assert result_df.empty

def test_count_top_terms_nested_output_dir(sample_data):
    """Test if nested output directory is created when it doesn't exist"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, 'test_input.csv')
        # Create a nested output path that doesn't exist
        nested_dir = os.path.join(temp_dir, 'level1', 'level2')
        output_file = os.path.join(nested_dir, 'test_output.csv')
        
        sample_data.to_csv(input_file, index=False)
        
        # Should create directories and not raise an error
        tc.count_top_terms(
            input_file=input_file,
            output_file=output_file
        )
        
        assert os.path.exists(output_file)

def test_count_top_terms_exception_handling():
    """Test exception handling in count_top_terms"""
    with patch('pandas.read_csv') as mock_read_csv:
        # Make read_csv raise an exception
        mock_read_csv.side_effect = Exception("Test exception")
        
        with patch('logging.Logger.error') as mock_error:
            with patch('traceback.print_exc') as mock_traceback:
                result_df = tc.count_top_terms(
                    input_file='any.csv',
                    output_file='output.csv'
                )
                
                # Should log the error
                mock_error.assert_called_once()
                assert "Test exception" in mock_error.call_args[0][0]
                
                # Should print the traceback
                mock_traceback.assert_called_once()
                
                # Should return an empty DataFrame
                assert result_df.empty