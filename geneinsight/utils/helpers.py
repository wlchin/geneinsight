#!/usr/bin/env python3
"""
Helper utilities for the TopicGenes package.
"""

import os
import logging
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

def ensure_directory(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(directory_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_run_id(gene_set_name: str) -> str:
    """
    Generate a unique run ID for a pipeline run.
    
    Args:
        gene_set_name: Name of the gene set
        
    Returns:
        Unique run ID
    """
    timestamp = generate_timestamp()
    return f"{gene_set_name}_{timestamp}"

def is_valid_gene_list_file(file_path: str) -> bool:
    """
    Check if a file is a valid gene list file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a valid gene list file, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        df = pd.read_csv(file_path, header=None)
        return len(df) > 0
    except Exception:
        try:
            # Try reading as a text file with one gene per line
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            return len(lines) > 0
        except Exception:
            return False

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize objects to JSON, handling non-serializable types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def save_metadata(
    metadata: Dict[str, Any],
    output_path: str,
    format: str = 'csv'
) -> str:
    """
    Save metadata to a file.
    
    Args:
        metadata: Dictionary of metadata
        output_path: Path to save the metadata
        format: Format to save (csv, json, yaml)
        
    Returns:
        Path to the saved metadata file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    format = format.lower()
    
    try:
        if format == 'csv':
            pd.DataFrame([metadata]).to_csv(output_path, index=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(metadata, f, default=safe_json_serialize, indent=2)
        elif format in ('yaml', 'yml'):
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return ""

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (lowercase, without the dot)
    """
    return os.path.splitext(file_path)[1].lower().lstrip('.')

def read_gene_list(file_path: str) -> List[str]:
    """
    Read a gene list from a file, handling different formats.
    
    Args:
        file_path: Path to the gene list file
        
    Returns:
        List of genes
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    ext = get_file_extension(file_path)
    
    try:
        if ext in ('csv', 'txt'):
            try:
                # Try reading as CSV
                df = pd.read_csv(file_path, header=None)
                return df.iloc[:, 0].tolist()
            except Exception:
                # Try reading as a text file with one gene per line
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        elif ext == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'genes' in data:
                    return data['genes']
                else:
                    logger.error(f"Unexpected JSON format in {file_path}")
                    return []
        else:
            logger.error(f"Unsupported file format: {ext}")
            return []
    except Exception as e:
        logger.error(f"Error reading gene list from {file_path}: {e}")
        return []