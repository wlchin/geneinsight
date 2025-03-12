#!/usr/bin/env python3
"""
Test suite for helpers.py module.
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the module to test - assuming it's in a package structure
from geneinsight.utils import helpers

class TestHelpers:
    
    def test_ensure_directory(self):
        """Test ensure_directory creates directories and returns absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test_dir")
            result = helpers.ensure_directory(test_dir)
            
            assert os.path.exists(test_dir)
            assert os.path.isabs(result)
            assert result == os.path.abspath(test_dir)

    def test_generate_timestamp(self):
        """Test generate_timestamp produces a properly formatted string."""
        timestamp = helpers.generate_timestamp()
        # Check format: YYYYMMDD_HHMMSS
        assert len(timestamp) == 15
        assert "_" in timestamp
        assert timestamp.replace("_", "").isdigit()
    
    def test_generate_run_id(self):
        """Test generate_run_id creates the expected format."""
        with patch('geneinsight.utils.helpers.generate_timestamp', return_value='20240311_123456'):
            run_id = helpers.generate_run_id("TestGeneSet")
            assert run_id == "TestGeneSet_20240311_123456"
    
    def test_is_valid_gene_list_file_csv(self):
        """Test is_valid_gene_list_file with a CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as tmp:
            try:
                tmp.write("GENE1\nGENE2\nGENE3")
                tmp.flush()
                
                assert helpers.is_valid_gene_list_file(tmp.name) is True
                
                # Test empty file
                with open(tmp.name, 'w') as f:
                    f.write("")
                assert helpers.is_valid_gene_list_file(tmp.name) is False
            finally:
                os.unlink(tmp.name)
    
    def test_is_valid_gene_list_file_text(self):
        """Test is_valid_gene_list_file with a text file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w+', delete=False) as tmp:
            try:
                tmp.write("GENE1\nGENE2\nGENE3")
                tmp.flush()
                
                assert helpers.is_valid_gene_list_file(tmp.name) is True
            finally:
                os.unlink(tmp.name)
    
    def test_is_valid_gene_list_file_nonexistent(self):
        """Test is_valid_gene_list_file with a non-existent file."""
        assert helpers.is_valid_gene_list_file("nonexistent_file.txt") is False
    
    def test_safe_json_serialize(self):
        """Test safe_json_serialize handles various types correctly."""
        # Test datetime
        dt = datetime(2024, 3, 11, 12, 34, 56)
        assert helpers.safe_json_serialize(dt) == dt.isoformat()
        
        # Test pandas DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        expected = [{"A": 1, "B": 3}, {"A": 2, "B": 4}]
        assert helpers.safe_json_serialize(df) == expected
        
        # Test pandas Series
        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        expected = {"a": 1, "b": 2, "c": 3}
        assert helpers.safe_json_serialize(series) == expected
        
        # Test set
        test_set = {1, 2, 3}
        assert sorted(helpers.safe_json_serialize(test_set)) == [1, 2, 3]
        
        # Test custom object
        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = "test"
        
        test_obj = TestObj()
        expected = {"a": 1, "b": "test"}
        assert helpers.safe_json_serialize(test_obj) == expected
        
        # Test plain object
        plain_obj = object()
        assert isinstance(helpers.safe_json_serialize(plain_obj), str)
    
    def test_save_metadata_csv(self):
        """Test save_metadata with CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metadata.csv")
            metadata = {"name": "test", "value": 123}
            
            result = helpers.save_metadata(metadata, output_path, format="csv")
            
            assert os.path.exists(output_path)
            assert result == output_path
            
            # Verify content
            df = pd.read_csv(output_path)
            assert df.iloc[0]["name"] == "test"
            assert df.iloc[0]["value"] == 123
    
    def test_save_metadata_json(self):
        """Test save_metadata with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metadata.json")
            metadata = {"name": "test", "value": 123}
            
            result = helpers.save_metadata(metadata, output_path, format="json")
            
            assert os.path.exists(output_path)
            assert result == output_path
            
            # Verify content
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert data["name"] == "test"
            assert data["value"] == 123
    
    def test_save_metadata_yaml(self):
        """Test save_metadata with YAML format."""
        # Mock yaml import since we don't want to require it for tests
        with patch.dict('sys.modules', {'yaml': MagicMock()}):
            mock_yaml = sys.modules['yaml']
            mock_yaml.dump = MagicMock()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "metadata.yaml")
                metadata = {"name": "test", "value": 123}
                
                result = helpers.save_metadata(metadata, output_path, format="yaml")
                
                # Just check if yaml.dump was called
                mock_yaml.dump.assert_called_once()
    
    def test_save_metadata_invalid_format(self):
        """Test save_metadata with an invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "metadata.xyz")
            metadata = {"name": "test", "value": 123}
            
            # The function logs an error but doesn't raise a ValueError
            result = helpers.save_metadata(metadata, output_path, format="xyz")
            assert result == ""
    
    def test_get_file_extension(self):
        """Test get_file_extension extracts the correct extension."""
        assert helpers.get_file_extension("file.txt") == "txt"
        assert helpers.get_file_extension("path/to/file.CSV") == "csv"
        assert helpers.get_file_extension("/absolute/path/file.JSON") == "json"
        assert helpers.get_file_extension("file") == ""
        assert helpers.get_file_extension("file.") == ""
    
    def test_read_gene_list_csv(self):
        """Test read_gene_list with CSV format."""
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as tmp:
            try:
                tmp.write("GENE1\nGENE2\nGENE3")
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == ["GENE1", "GENE2", "GENE3"]
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_txt(self):
        """Test read_gene_list with text format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w+', delete=False) as tmp:
            try:
                tmp.write("GENE1\nGENE2\nGENE3")
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == ["GENE1", "GENE2", "GENE3"]
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_json_list(self):
        """Test read_gene_list with JSON list format."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tmp:
            try:
                json.dump(["GENE1", "GENE2", "GENE3"], tmp)
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == ["GENE1", "GENE2", "GENE3"]
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_json_dict(self):
        """Test read_gene_list with JSON dict format."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tmp:
            try:
                json.dump({"genes": ["GENE1", "GENE2", "GENE3"]}, tmp)
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == ["GENE1", "GENE2", "GENE3"]
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_json_invalid(self):
        """Test read_gene_list with invalid JSON format."""
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=False) as tmp:
            try:
                json.dump({"not_genes": ["GENE1", "GENE2", "GENE3"]}, tmp)
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == []
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_unsupported(self):
        """Test read_gene_list with unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', mode='w+', delete=False) as tmp:
            try:
                tmp.write("GENE1\nGENE2\nGENE3")
                tmp.flush()
                
                result = helpers.read_gene_list(tmp.name)
                assert result == []
            finally:
                os.unlink(tmp.name)
    
    def test_read_gene_list_nonexistent(self):
        """Test read_gene_list with a non-existent file."""
        result = helpers.read_gene_list("nonexistent_file.txt")
        assert result == []

if __name__ == "__main__":
    pytest.main()