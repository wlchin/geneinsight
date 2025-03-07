#!/usr/bin/env python3
"""
Configuration handling for the TopicGenes package.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "n_samples": 5,
    "num_topics": 10,
    "pvalue_threshold": 0.01,
    "api_service": "openai",
    "api_model": "gpt-4o-mini",
    "api_parallel_jobs": 4,
    "api_base_url": None,
    "target_filtered_topics": 25,
}

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported or invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")
    
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration values and set defaults for missing values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
    """
    validated = DEFAULT_CONFIG.copy()
    
    # Update with provided values
    for key, value in config.items():
        if key in validated:
            # Type checking
            if isinstance(validated[key], int) and not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid type for {key}, expected int. Using default: {validated[key]}")
                    continue
            
            # Range checking for specific parameters
            if key == "n_samples" and (value < 1 or value > 100):
                logger.warning(f"Invalid value for {key}: {value}, must be between 1 and 100. Using default: {validated[key]}")
                continue
                
            # Add similar validation for other parameters...
                
            validated[key] = value
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return validated

def load_from_env(prefix: str = "TOPICGENES_") -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Args:
        prefix: Prefix for environment variables
        
    Returns:
        Dictionary with configuration from environment variables
    """
    config = {}
    
    # Map environment variable names to config keys
    env_mappings = {
        f"{prefix}N_SAMPLES": "n_samples",
        f"{prefix}NUM_TOPICS": "num_topics",
        f"{prefix}PVALUE_THRESHOLD": "pvalue_threshold",
        f"{prefix}API_SERVICE": "api_service",
        f"{prefix}API_MODEL": "api_model",
        f"{prefix}API_PARALLEL_JOBS": "api_parallel_jobs",
        f"{prefix}API_BASE_URL": "api_base_url",
        f"{prefix}TARGET_FILTERED_TOPICS": "target_filtered_topics",
    }
    
    # Get values from environment
    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            
            # Try to convert numeric values
            try:
                if config_key in ["n_samples", "num_topics", "api_parallel_jobs", "target_filtered_topics"]:
                    value = int(value)
                elif config_key == "pvalue_threshold":
                    value = float(value)
            except ValueError:
                logger.warning(f"Invalid value for {env_var}: {value}")
                continue
                
            config[config_key] = value
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration from various sources, with precedence:
    1. Configuration file (if provided)
    2. Environment variables
    3. Default configuration
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Complete configuration dictionary
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Update with environment variables
    config.update(load_from_env())
    
    # Update with configuration file if provided
    if config_path:
        try:
            file_config = load_config(config_path)
            config.update(file_config)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Error loading configuration file: {e}")
    
    # Validate the final configuration
    config = validate_config(config)
    
    return config