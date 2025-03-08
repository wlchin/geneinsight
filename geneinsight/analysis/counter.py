"""
Module for counting top terms in topic modeling results.
"""

import os
import logging
import sys
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def count_strings_most_common(terms_list: List[str], top_n: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Count the occurrences of each string in the list and return the most common.
    
    Args:
        terms_list: List of strings to count
        top_n: Number of top terms to return (None for all)
        
    Returns:
        List of (term, count) tuples sorted by count descending
    """
    if not terms_list:
        logger.warning("Empty list provided for counting")
        return []
    
    # Count the occurrences of each string in the list
    counter = Counter(terms_list)
    
    if top_n is not None:
        return counter.most_common(top_n)
    
    return counter.most_common()  # Return all counts if top_n is not specified

def count_top_terms(input_file: str, output_file: str, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Count the top terms in a topic modeling result CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        top_n: Number of top terms to output (None for all)
        
    Returns:
        DataFrame containing the top terms and their counts
    """
    try:
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Check for required columns
        if "Representative_document" not in df.columns or "Document" not in df.columns:
            logger.warning("Input CSV missing required columns. Looking for alternative columns...")
            
            # Try to find suitable columns
            doc_col = None
            rep_col = None
            
            # Look for document column
            for col in ["Document", "Term", "description"]:
                if col in df.columns:
                    doc_col = col
                    break
            
            # Look for representative document column or create one
            if "Representative_document" in df.columns:
                rep_col = "Representative_document"
            else:
                if "Probability" in df.columns:
                    # Create representative column based on probability
                    df["Representative_document"] = df["Probability"] > df["Probability"].median()
                    rep_col = "Representative_document"
                else:
                    # Default to considering all documents as representative
                    df["Representative_document"] = True
                    rep_col = "Representative_document"
            
            if not doc_col:
                logger.error("Could not find a suitable document column in the input CSV")
                return pd.DataFrame()
            
            logger.info(f"Using columns: Document='{doc_col}', Representative='{rep_col}'")
            
            # Extract the terms from the DataFrame
            terms_list = df[df[rep_col] == True][doc_col].tolist()
        else:
            # Use standard column names
            terms_list = df[df["Representative_document"] == True]["Document"].tolist()
        
        logger.info(f"Found {len(terms_list)} representative terms to count")
        
        # Count the occurrences of each term and get the top N terms
        top_terms = count_strings_most_common(terms_list, top_n)
        
        # Convert the top terms to a DataFrame
        top_terms_df = pd.DataFrame(top_terms, columns=["Term", "Count"])
        
        # Save the top terms DataFrame to a CSV file
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            top_terms_df.to_csv(output_file, index=False)
            logger.info(f"Saved top terms to {output_file}")
        
        return top_terms_df
    
    except Exception as e:
        logger.error(f"Error counting top terms: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()