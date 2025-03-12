#!/usr/bin/env python3
"""
Pytest configuration file for similarity tests
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile

# Add the parent directory to sys.path to ensure the module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_dir():
    """Get the directory where tests are located"""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        yield tmp.name
        # Cleanup after the test
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

# Define fixtures that can be used across multiple test files

@pytest.fixture
def sample_gene_list():
    return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

@pytest.fixture
def sample_background_list():
    return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", 
            "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]

@pytest.fixture
def sample_ontology_content():
    return (
        "Term1\t\tGENE1\tGENE2\tGENE3\n"
        "Term2\t\tGENE4\tGENE5\n"
        "Term3\t\tGENE1\tGENE6\tGENE7\n"
        "Term4\t\tGENE8\tGENE9\tGENE10"
    )

@pytest.fixture
def mock_ontology_file(sample_ontology_content, tmp_path):
    """Creates a temporary ontology file for testing"""
    file_path = tmp_path / "test_ontology.txt"
    with open(file_path, 'w') as f:
        f.write(sample_ontology_content)
    return file_path

@pytest.fixture
def sample_enrichr_results():
    """Returns a sample enrichment results DataFrame"""
    return pd.DataFrame({
        'Term': ['Term1', 'Term2', 'Term3', 'Term4'],
        'P-value': [0.001, 0.01, 0.05, 0.1],
        'Adjusted P-value': [0.005, 0.05, 0.1, 0.2],
        'Genes': ['GENE1,GENE2,GENE3', 'GENE4,GENE5', 'GENE1,GENE6,GENE7', 'GENE8,GENE9,GENE10'],
        'Gene_set': ['ontology1', 'ontology1', 'ontology2', 'ontology2'],
    })

@pytest.fixture
def mock_sentence_transformer():
    """Returns a mocked SentenceTransformer class"""
    with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
        # Configure the mock to return a tensor for encode
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance
        yield mock_transformer

@pytest.fixture
def mock_gseapy_enrich():
    """Returns a mocked gseapy.enrich function"""
    with patch('gseapy.enrich') as mock_enrich:
        # Configure mock results
        mock_result = MagicMock()
        mock_result.res2d = pd.DataFrame({
            'Term': ['Term1', 'Term2'],
            'P-value': [0.01, 0.05],
            'Adjusted P-value': [0.02, 0.1],
            'Genes': ['GENE1,GENE2', 'GENE3,GENE4'],
            'Overlap': ['2/5', '2/7']
        })
        mock_enrich.return_value = mock_result
        yield mock_enrich

@pytest.fixture
def sample_summary_csv(tmp_path):
    """Creates a sample summary CSV file for testing the main function"""
    data = pd.DataFrame({
        'query': ['query1', 'query2'],
        'unique_genes': ["{'GENE1': 1, 'GENE2': 1}", "{'GENE3': 1, 'GENE4': 1}"]
    })
    file_path = tmp_path / "summary.csv"
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def sample_filter_csv(tmp_path):
    """Creates a sample filter CSV file for testing the main function"""
    data = pd.DataFrame({
        'Term': ['query1', 'query2', 'query3']
    })
    file_path = tmp_path / "filter.csv"
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def mock_ontology_folder(tmp_path):
    """Creates a mock ontology folder with sample ontology files"""
    folder_path = tmp_path / "ontologies"
    os.makedirs(folder_path, exist_ok=True)
    
    # Create a couple of ontology files
    with open(folder_path / "ontology1.txt", 'w') as f:
        f.write("Term1\t\tGENE1\tGENE2\nTerm2\t\tGENE3\tGENE4\n")
    
    with open(folder_path / "ontology2.txt", 'w') as f:
        f.write("Term3\t\tGENE5\tGENE6\nTerm4\t\tGENE7\tGENE8\n")
    
    return folder_path

@pytest.fixture
def realistic_embeddings():
    """Generate realistic embeddings for testing"""
    # Create more complex embeddings with higher dimensionality
    np.random.seed(42)  # For reproducibility
    
    # Create 10 embeddings with dimension 20
    base_embeddings = np.random.random((10, 20))
    
    # Normalize the embeddings
    norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    normalized = base_embeddings / norms
    
    # Create some similar embeddings
    similar_pairs = [
        (0, 1, 0.9),  # Items 0 and 1 are very similar
        (2, 3, 0.8),  # Items 2 and 3 are quite similar
        (4, 5, 0.7),  # Items 4 and 5 are somewhat similar
    ]
    
    # Adjust embeddings to enforce desired similarities
    for idx1, idx2, target_sim in similar_pairs:
        # Linear interpolation to create similar embeddings
        normalized[idx2] = target_sim * normalized[idx1] + (1 - target_sim) * normalized[idx2]
        # Re-normalize
        normalized[idx2] = normalized[idx2] / np.linalg.norm(normalized[idx2])
    
    return normalized


@pytest.fixture
def large_sample_dataset():
    """Create a larger dataset for more realistic testing"""
    # Create terms with varying similarities
    terms = [
        "machine learning", "deep learning", "artificial intelligence",
        "natural language processing", "text processing", "nlp",
        "computer vision", "image recognition", "object detection",
        "data science", "statistics", "data analysis",
        "neural networks", "convolutional networks", "recurrent networks",
        "gradient descent", "backpropagation", "optimization",
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "clustering", "classification", "regression",
        "feature extraction", "dimensionality reduction", "pca"
    ]
    
    # Create frequencies with different values
    counts = [
        100, 85, 90,
        70, 50, 65,
        80, 75, 60,
        95, 85, 90,
        80, 60, 55,
        40, 35, 45,
        75, 70, 65,
        60, 85, 80,
        50, 45, 40
    ]
    
    return pd.DataFrame({"Term": terms, "Count": counts})

