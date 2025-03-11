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