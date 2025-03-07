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
        ignore_patterns: List of patterns to ignore (uses startswith)
        
    Returns:
        Path to the created zip file
    """
    directory_path = os.path.abspath(directory_path)
    output_path = os.path.abspath(output_path)
    
    if ignore_patterns is None:
        ignore_patterns = ['.DS_Store', '__MACOSX', 'Thumbs.db', '.git']
    
    logger.info(f"Zipping directory {directory_path} to {output_path}")
    
    # Create parent directory for zip file if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get the name of the directory to be zipped
    dir_name = os.path.basename(directory_path)
    
    # Create the zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            # Remove directories in ignore_patterns
            dirs[:] = [d for d in dirs if not any(d.startswith(p) for p in ignore_patterns)]
            
            # Calculate the relative path from the directory_path
            rel_path = os.path.relpath(root, os.path.dirname(directory_path))
            
            for file in files:
                # Skip files in ignore_patterns
                if any(file.startswith(p) for p in ignore_patterns):
                    continue
                
                # Get the full file path
                file_path = os.path.join(root, file)
                
                # Calculate the archive path
                archive_path = os.path.join(rel_path, file)
                
                # Add the file to the zip
                zipf.write(file_path, archive_path)
    
    logger.info(f"Successfully created zip file at {output_path}")
    return output_path