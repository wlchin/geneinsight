import pytest
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, Mock

# Import the module to test
# Assuming the module is named topic_prompt_generator.py
import geneinsight.workflows.prompt_generation as tpg

@pytest.fixture
def sample_data():
    """Fixture for test data"""
    return pd.DataFrame({
        'Topic': [0, 0, 0, 1, 1, 2],
        'Representative_document': [True, False, True, True, True, True],
        'Document': ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6'],
        'Top_n_words': ['word1, word2', 'word1, word2', 'word1, word2', 
                       'word3, word4', 'word3, word4', 'word5, word6'],
        'seed': [1, 1, 1, 1, 1, 2]
    })

@pytest.fixture
def temp_files():
    """Fixture for temporary files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, 'test_input.csv')
        output_file = os.path.join(temp_dir, 'test_output.csv')
        yield {'input': input_file, 'output': output_file, 'dir': temp_dir}

def test_create_transcript_for_topic(sample_data):
    """Test create_transcript_for_topic function"""
    # Test for topic 0
    transcript = tpg.create_transcript_for_topic(sample_data, 0)
    
    # Check if transcript contains expected documents and keywords
    expected_docs = json.dumps(['doc1', 'doc3'])
    expected_keywords = json.dumps(['word1, word2'])
    
    assert expected_docs in transcript
    assert expected_keywords in transcript
    
    # Test for topic 1
    transcript = tpg.create_transcript_for_topic(sample_data, 1)
    expected_docs = json.dumps(['doc4', 'doc5'])
    expected_keywords = json.dumps(['word3, word4'])
    
    assert expected_docs in transcript
    assert expected_keywords in transcript

def test_create_transcript_without_representative_flag(sample_data):
    """Test create_transcript_for_topic without Representative_document column"""
    # Remove Representative_document column
    df_no_rep = sample_data.drop('Representative_document', axis=1)
    
    transcript = tpg.create_transcript_for_topic(df_no_rep, 0)
    expected_docs = json.dumps(['doc1', 'doc2', 'doc3'])
    
    assert expected_docs in transcript

def test_create_transcript_missing_columns(sample_data):
    """Test create_transcript_for_topic with missing columns"""
    # Test with missing Document column
    df_no_doc = sample_data.drop('Document', axis=1)
    transcript = tpg.create_transcript_for_topic(df_no_doc, 0)
    assert "Here are documents: []" in transcript
    
    # Test with missing Top_n_words column
    df_no_words = sample_data.drop('Top_n_words', axis=1)
    transcript = tpg.create_transcript_for_topic(df_no_words, 0)
    assert "Here are keywords: []" in transcript

def test_generate_prompts(sample_data, temp_files):
    """Test generate_prompts function with sample data"""
    # Save sample data to input file
    sample_data.to_csv(temp_files['input'], index=False)
    
    # Call generate_prompts
    result_df = tpg.generate_prompts(
        input_file=temp_files['input'],
        num_subtopics=3,
        max_words=5,
        output_file=temp_files['output']
    )
    
    # Check if output file was created
    assert os.path.exists(temp_files['output'])
    
    # Check if resulting DataFrame has expected shape
    # We have 3 unique topics (0, 1, 2) and 3 subtopics per topic
    # But topic 2 only appears in seed 2, so we have 2 topics in seed 1 and 1 topic in seed 2
    # That's (2*3) + (1*3) = 9 rows
    assert len(result_df) == 9
    
    # Check column names
    expected_columns = [
        'prompt_type', 'seed', 'topic_label', 'subtopic_label',
        'major_transcript', 'subtopic_transcript', 'num_subtopics',
        'max_words', 'max_retries'
    ]
    for col in expected_columns:
        assert col in result_df.columns
    
    # Check subtopics
    assert set(result_df['subtopic_label'].unique()) == {0, 1, 2}
    
    # Check if prompts have different transcripts for different topics
    topic0_prompts = result_df[result_df['topic_label'] == 0]
    topic1_prompts = result_df[result_df['topic_label'] == 1]
    
    assert topic0_prompts['major_transcript'].iloc[0] != topic1_prompts['major_transcript'].iloc[0]

def test_generate_prompts_no_seed_column(sample_data, temp_files):
    """Test generate_prompts with data that has no seed column"""
    # Remove seed column
    no_seed_data = sample_data.drop('seed', axis=1)
    no_seed_data.to_csv(temp_files['input'], index=False)
    
    # Call generate_prompts
    result_df = tpg.generate_prompts(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Check if resulting DataFrame has expected shape
    # With no seed column, default seed is 0
    # We have 3 unique topics (0, 1, 2) and 3 subtopics for each = 9 rows
    assert len(result_df) == 9
    
    # Check default seed
    assert set(result_df['seed'].unique()) == {0}

def test_generate_prompts_skip_noise_topic(sample_data, temp_files):
    """Test generate_prompts skips noise topic (-1)"""
    # Add a noise topic
    noise_data = sample_data.copy()
    noise_row = pd.DataFrame({
        'Topic': [-1],
        'Representative_document': [True],
        'Document': ['noise_doc'],
        'Top_n_words': ['noise, words'],
        'seed': [1]
    })
    noise_data = pd.concat([noise_data, noise_row])
    noise_data.to_csv(temp_files['input'], index=False)
    
    # Call generate_prompts
    result_df = tpg.generate_prompts(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Check that noise topic is skipped
    assert -1 not in result_df['topic_label'].unique()

@patch('logging.Logger.info')
def test_logging(mock_logger_info, sample_data, temp_files):
    """Test if logging messages are generated correctly"""
    sample_data.to_csv(temp_files['input'], index=False)
    
    tpg.generate_prompts(
        input_file=temp_files['input'],
        output_file=temp_files['output']
    )
    
    # Check if logging info was called with expected messages
    mock_logger_info.assert_any_call(f"Reading topic modeling results from {temp_files['input']}")
    mock_logger_info.assert_any_call(f"Processing topic model with seed 1")
    mock_logger_info.assert_any_call(f"Processing topic model with seed 2")

def test_output_directory_creation(sample_data):
    """Test if output directory is created when it doesn't exist"""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, 'test_input.csv')
        # Create a nested output path that doesn't exist
        nonexistent_dir = os.path.join(temp_dir, 'new_dir', 'another_dir')
        output_file = os.path.join(nonexistent_dir, 'test_output.csv')
        
        sample_data.to_csv(input_file, index=False)
        
        # Should create directories and not raise an error
        tpg.generate_prompts(
            input_file=input_file,
            output_file=output_file
        )
        
        assert os.path.exists(output_file)