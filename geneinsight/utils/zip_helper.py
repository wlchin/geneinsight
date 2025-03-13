"""
Utility functions for zipping directories.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def zip_directory(directory_path: str, output_path: str, ignore_patterns: Optional[List[str]] = None) -> str:
    """
    Zip the contents of a directory.
    
    Args:
        directory_path: Path to the directory to zip
        output_path: Path to the output zip file
        ignore_patterns: List of additional patterns to ignore (uses startswith)
        
    Returns:
        Path to the created zip file
    """
    directory_path = os.path.abspath(directory_path)
    output_path = os.path.abspath(output_path)
    
    default_ignore = ['.DS_Store', '__MACOSX', 'Thumbs.db', '.git']
    if ignore_patterns is None:
        ignore_patterns = default_ignore
    else:
        # Always include the default ignore patterns.
        ignore_patterns = default_ignore + ignore_patterns
    
    logger.info(f"Zipping directory {directory_path} to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dir_name = os.path.basename(directory_path)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            # Filter out directories that match any ignore pattern
            dirs[:] = [d for d in dirs if not any(d.startswith(p) for p in ignore_patterns)]
            
            rel_path = os.path.relpath(root, os.path.dirname(directory_path))
            
            for file in files:
                if any(file.startswith(p) for p in ignore_patterns):
                    continue
                file_path = os.path.join(root, file)
                archive_path = os.path.join(rel_path, file)
                zipf.write(file_path, archive_path)
    
    logger.info(f"Successfully created zip file at {output_path}")
    return output_path
