"""
Module for performing hypergeometric Gene Set Enrichment Analysis (GSEA).
"""

import os
import ast
import logging
import pandas as pd
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Try to import gseapy
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

class HypergeometricGSEA:
    """
    Class for performing hypergeometric Gene Set Enrichment Analysis (GSEA).
    """
    
    def __init__(self, genelist: List[str], background_list: Optional[List[str]] = None):
        """
        Initialize the analysis.
        
        Args:
            genelist: List of genes to test
            background_list: Background gene population (optional)
        """
        self.genelist = genelist
        self.background_list = background_list
    
    def perform_hypergeometric_gsea(self, geneset_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Perform hypergeometric GSEA using the provided gene sets.
        
        Args:
            geneset_dict: Dictionary where keys are set names and values are gene lists
            
        Returns:
            DataFrame with enrichment results
        """
        if not GSEAPY_AVAILABLE:
            logger.error("gseapy package not available. Please install it.")
            return pd.DataFrame()
        
        # Return empty DataFrame if geneset_dict is empty
        if not geneset_dict:
            logger.warning("No gene sets provided for GSEA analysis.")
            return pd.DataFrame()
        
        logger.info(f"Performing hypergeometric GSEA with {len(geneset_dict)} gene sets")
        
        try:
            enr = gp.enrich(
                gene_list=self.genelist,
                gene_sets=geneset_dict,
                background=self.background_list,
                outdir=None,
                verbose=True
            )
            # Ensure we return a DataFrame even if res2d is None
            if hasattr(enr, 'res2d') and enr.res2d is not None:
                return enr.res2d
            else:
                logger.warning("No results returned from GSEA analysis.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error in hypergeometric GSEA: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

def hypergeometric_enrichment(
    df_path: str,
    gene_origin_path: str,
    background_genes_path: str,
    output_csv: str,
    pvalue_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Perform hypergeometric enrichment analysis on a summary DataFrame.
    
    Args:
        df_path: Path to summary CSV file
        gene_origin_path: Path to gene origin file
        background_genes_path: Path to background genes file
        output_csv: Path to output CSV file
        pvalue_threshold: P-value threshold for filtering results
        
    Returns:
        DataFrame with filtered enrichment results
    """
    if not GSEAPY_AVAILABLE:
        logger.error("gseapy package not available. Please install it.")
        return pd.DataFrame()
    
    logger.info("Starting hypergeometric enrichment analysis")
    
    # Read input files
    try:
        df = pd.read_csv(df_path)
        logger.info(f"Read summary data with {len(df)} rows")
        
        gene_origin = pd.read_csv(gene_origin_path, header=None)
        logger.info(f"Read {len(gene_origin)} genes from origin file")
        
        background_genes = pd.read_csv(background_genes_path, header=None)
        logger.info(f"Read {len(background_genes)} background genes")
        
        background_genes_list = background_genes[0].tolist()
    except Exception as e:
        logger.error(f"Error reading input files: {e}")
        return pd.DataFrame()
    
    # Extract the unique genes from each topic
    try:
        # Extract minor topics with unique genes
        if "query" not in df.columns or "unique_genes" not in df.columns:
            logger.error("Required columns 'query' and 'unique_genes' not found in summary CSV")
            return pd.DataFrame()
        
        minor_topics = df[["query", "unique_genes"]].drop_duplicates()
        
        # Keep only the first query for each unique set of genes
        minor_topics = minor_topics.drop_duplicates(subset=["unique_genes"], keep="first")
        logger.info(f"Found {len(minor_topics)} unique gene sets")
        
        # Create gene query dictionary
        genequery = {}
        for _, row in minor_topics.iterrows():
            query_name = row["query"]
            
            try:
                # Try to parse the unique_genes string as a dictionary and extract keys
                geneset = list(ast.literal_eval(row["unique_genes"]).keys())
                if geneset:  # Only add if geneset is not empty
                    genequery[query_name] = geneset
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Error parsing unique_genes for query {query_name}: {e}")
                continue
        
        if not genequery:
            logger.warning("No valid gene sets extracted for GSEA analysis.")
            # Create empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"])
            
            # Save to CSV if requested
            if output_csv:
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                empty_df.to_csv(output_csv, index=False)
                logger.info(f"Saved empty results to {output_csv}")
                
            return empty_df
            
        logger.info(f"Created query dictionary with {len(genequery)} entries")
    except Exception as e:
        logger.error(f"Error processing gene sets: {e}")
        return pd.DataFrame()
    
    # Perform GSEA
    gsea = HypergeometricGSEA(gene_origin[0].tolist(), background_list=background_genes_list)
    result = gsea.perform_hypergeometric_gsea(genequery)
    
    # Always ensure result is a DataFrame
    if result is None:
        logger.warning("GSEA returned None instead of results.")
        result = pd.DataFrame()
    
    if result.empty:
        logger.warning("GSEA returned no results")
        # Create empty DataFrame with expected columns if needed
        if len(result.columns) == 0:
            result = pd.DataFrame(columns=["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"])
    else:
        # Filter for significant results
        filtered_result = result[result["Adjusted P-value"] < pvalue_threshold]
        logger.info(f"Filtered to {len(filtered_result)} significant results (p < {pvalue_threshold})")
        result = filtered_result
    
    # Save to CSV
    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            result.to_csv(output_csv, index=False)
            logger.info(f"Saved results to {output_csv}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    return result