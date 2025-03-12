import os
import json
import yaml
import pytest
from unittest.mock import patch, mock_open

# Import the module directly - assuming the file structure is:
# geneinsight/
#   __init__.py
#   config.py
#   tests/
#     test_config.py
from geneinsight.config import (
    load_config, validate_config, load_from_env, 
    save_config, get_config, DEFAULT_CONFIG
)


@pytest.fixture
def sample_config():
    return {
        "n_samples": 10,
        "num_topics": 15,
        "pvalue_threshold": 0.01,
        "api_service": "anthropic",
        "api_model": "claude-3",
        "api_parallel_jobs": 8,
        "api_base_url": "https://api.example.com",
        "target_filtered_topics": 30,
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create temporary config files for testing."""
    # JSON config
    json_file = tmp_path / "config.json"
    with open(json_file, 'w') as f:
        json.dump(sample_config, f)
    
    # YAML config
    yaml_file = tmp_path / "config.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(sample_config, f)
    
    return {"json": json_file, "yaml": yaml_file}


def test_load_config_json(temp_config_file, sample_config):
    """Test loading JSON configuration."""
    config = load_config(str(temp_config_file["json"]))
    assert config == sample_config


def test_load_config_yaml(temp_config_file, sample_config):
    """Test loading YAML configuration."""
    config = load_config(str(temp_config_file["yaml"]))
    assert config == sample_config


def test_load_config_file_not_found():
    """Test behavior when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.json")


def test_load_config_unsupported_format():
    """Test behavior with unsupported file format."""
    with patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config("config.txt")


def test_load_config_invalid_content():
    """Test behavior with invalid JSON/YAML content."""
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="invalid: : json")):
            with pytest.raises(ValueError, match="Error loading configuration"):
                load_config("config.json")


def test_validate_config_valid_values(sample_config):
    """Test validation with valid values."""
    validated = validate_config(sample_config)
    assert validated == sample_config


def test_validate_config_type_conversion():
    """Test validation with type conversion."""
    config = {"n_samples": "20", "num_topics": "5"}
    validated = validate_config(config)
    assert validated["n_samples"] == 20
    assert validated["num_topics"] == 5
    assert isinstance(validated["n_samples"], int)
    assert isinstance(validated["num_topics"], int)


def test_validate_config_invalid_values():
    """Test validation with invalid values."""
    config = {"n_samples": 200, "num_topics": "not_a_number"}
    validated = validate_config(config)
    # n_samples should be rejected (out of range)
    assert validated["n_samples"] == DEFAULT_CONFIG["n_samples"]
    # num_topics should be rejected (invalid type)
    assert validated["num_topics"] == DEFAULT_CONFIG["num_topics"]


def test_validate_config_unknown_parameters():
    """Test validation with unknown parameters."""
    config = {"unknown_param": "value"}
    validated = validate_config(config)
    assert "unknown_param" not in validated
    assert validated == DEFAULT_CONFIG


def test_load_from_env():
    """Test loading from environment variables."""
    env_vars = {
        "TOPICGENES_N_SAMPLES": "15",
        "TOPICGENES_NUM_TOPICS": "20",
        "TOPICGENES_PVALUE_THRESHOLD": "0.02",
        "TOPICGENES_API_SERVICE": "anthropic",
        "TOPICGENES_API_MODEL": "claude-3",
        "TOPICGENES_API_PARALLEL_JOBS": "6",
        "TOPICGENES_API_BASE_URL": "https://api.example.com",
        "TOPICGENES_TARGET_FILTERED_TOPICS": "35",
    }
    
    with patch.dict(os.environ, env_vars):
        config = load_from_env()
        
        assert config["n_samples"] == 15
        assert config["num_topics"] == 20
        assert config["pvalue_threshold"] == 0.02
        assert config["api_service"] == "anthropic"
        assert config["api_model"] == "claude-3"
        assert config["api_parallel_jobs"] == 6
        assert config["api_base_url"] == "https://api.example.com"
        assert config["target_filtered_topics"] == 35


def test_load_from_env_invalid_values():
    """Test loading from environment variables with invalid values."""
    env_vars = {
        "TOPICGENES_N_SAMPLES": "not_a_number",
        "TOPICGENES_PVALUE_THRESHOLD": "not_a_float",
    }
    
    with patch.dict(os.environ, env_vars):
        config = load_from_env()
        assert "n_samples" not in config
        assert "pvalue_threshold" not in config


def test_load_from_env_custom_prefix():
    """Test loading from environment variables with custom prefix."""
    env_vars = {
        "CUSTOM_N_SAMPLES": "25",
        "CUSTOM_NUM_TOPICS": "30",
    }
    
    with patch.dict(os.environ, env_vars):
        config = load_from_env(prefix="CUSTOM_")
        assert config["n_samples"] == 25
        assert config["num_topics"] == 30


def test_save_config_json(tmp_path, sample_config):
    """Test saving configuration to JSON file."""
    config_path = tmp_path / "output" / "config.json"
    save_config(sample_config, str(config_path))
    
    assert config_path.exists()
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == sample_config


def test_save_config_yaml(tmp_path, sample_config):
    """Test saving configuration to YAML file."""
    config_path = tmp_path / "output" / "config.yaml"
    save_config(sample_config, str(config_path))
    
    assert config_path.exists()
    with open(config_path, 'r') as f:
        saved_config = yaml.safe_load(f)
    assert saved_config == sample_config


def test_save_config_unsupported_format(sample_config, caplog):
    """Test saving to unsupported file format."""
    # Based on the error log, the function logs an error instead of raising an exception
    save_config(sample_config, "config.txt")
    assert "Error saving configuration: Unsupported configuration file format: .txt" in caplog.text


def test_get_config_default():
    """Test getting default configuration."""
    config = get_config()
    assert config == DEFAULT_CONFIG


def test_get_config_with_env_vars():
    """Test getting configuration with environment variables."""
    env_vars = {
        "TOPICGENES_N_SAMPLES": "15",
        "TOPICGENES_API_SERVICE": "anthropic",
    }
    
    with patch.dict(os.environ, env_vars):
        config = get_config()
        
        assert config["n_samples"] == 15
        assert config["api_service"] == "anthropic"
        # Other values should be default
        assert config["num_topics"] == DEFAULT_CONFIG["num_topics"]


def test_get_config_with_file(temp_config_file, sample_config):
    """Test getting configuration with config file."""
    config = get_config(str(temp_config_file["json"]))
    assert config == sample_config


def test_get_config_precedence():
    """Test configuration precedence (file > env > default)."""
    file_config = {
        "n_samples": 25,
        "num_topics": 30,
    }
    
    env_vars = {
        "TOPICGENES_N_SAMPLES": "15",
        "TOPICGENES_API_SERVICE": "anthropic",
    }
    
    with patch("geneinsight.config.load_config", return_value=file_config):
        with patch.dict(os.environ, env_vars):
            config = get_config("dummy_path.json")
            
            # File should override env var
            assert config["n_samples"] == 25
            # File should override default
            assert config["num_topics"] == 30
            # Env var should override default
            assert config["api_service"] == "anthropic"
            # Untouched default value
            assert config["pvalue_threshold"] == DEFAULT_CONFIG["pvalue_threshold"]


def test_get_config_file_error():
    """Test behavior when config file has an error."""
    env_vars = {"TOPICGENES_N_SAMPLES": "15"}
    
    with patch("geneinsight.config.load_config", side_effect=ValueError("Test error")):
        with patch.dict(os.environ, env_vars):
            # Should fall back to env + default without failing
            config = get_config("error_file.json")
            assert config["n_samples"] == 15
            assert config["num_topics"] == DEFAULT_CONFIG["num_topics"]