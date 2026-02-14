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
        if background_list is not None and genelist:
            first_gene = genelist[0]
            if first_gene.isupper():
                background_list = [g.upper() for g in background_list]
            elif first_gene.islower():
                background_list = [g.lower() for g in background_list]
            elif first_gene.istitle():
                background_list = [g.title() for g in background_list]

            intersect = set(genelist).intersection(set(background_list))
            if not intersect:
                logger.warning("No intersection found between genelist and background_list.")

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
        
        if self.genelist:
            first_gene = self.genelist[0]
            if first_gene.isupper():
                for k, v in geneset_dict.items():
                    geneset_dict[k] = [g.upper() for g in v]
            elif first_gene.islower():
                for k, v in geneset_dict.items():
                    geneset_dict[k] = [g.lower() for g in v]
            elif first_gene.istitle():
                for k, v in geneset_dict.items():
                    geneset_dict[k] = [g.title() for g in v]

            all_genes = set()
            for genes in geneset_dict.values():
                all_genes.update(genes)
            if not set(self.genelist).intersection(all_genes):
                logger.warning("No intersection found between genelist and geneset_dict.")

        logger.info(f"Performing hypergeometric GSEA with {len(geneset_dict)} gene sets")
        # print out geneset_dict
        logger.debug(f"Gene set dictionary: {geneset_dict}")
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
    pvalue_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Perform hypergeometric enrichment analysis on a summary DataFrame.

    Note: The gene_origin_path parameter is kept for backward compatibility
    but the genelist is now extracted from the summary DataFrame to ensure
    consistent gene ID formats (symbols) regardless of input format (e.g.,
    Ensembl IDs are converted to symbols by StringDB before reaching here).

    Args:
        df_path: Path to summary CSV file
        gene_origin_path: Path to gene origin file (kept for backward compatibility)
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
    except Exception as e:
        logger.error(f"Error reading summary file: {e}")
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

        # Create gene query dictionary AND extract all unique genes for genelist
        genequery = {}
        all_unique_genes = set()

        for _, row in minor_topics.iterrows():
            query_name = row["query"]

            try:
                # Try to parse the unique_genes string as a dictionary and extract keys
                genes_dict = ast.literal_eval(row["unique_genes"])
                geneset = list(genes_dict.keys())
                if geneset:  # Only add if geneset is not empty
                    genequery[query_name] = geneset
                    all_unique_genes.update(geneset)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Error parsing unique_genes for query {query_name}: {e}")
                continue

        # Extract unique genes from ALL rows (not just minor_topics) for complete genelist
        for _, row in df.iterrows():
            try:
                genes_dict = ast.literal_eval(row["unique_genes"])
                all_unique_genes.update(genes_dict.keys())
            except (SyntaxError, ValueError):
                continue

        genelist = list(all_unique_genes)
        logger.info(f"Extracted {len(genelist)} unique genes from summary for GSEA")

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

    # Read background genes (optional - may not match format)
    background_genes_list = None
    try:
        background_genes = pd.read_csv(background_genes_path, header=None)
        background_genes_list = background_genes[0].tolist()
        logger.info(f"Read {len(background_genes_list)} background genes")

        # Check if background genes match format of genelist
        if genelist and background_genes_list:
            sample_query = genelist[0]
            sample_bg = background_genes_list[0]
            # Detect format mismatch (Ensembl IDs start with ENS)
            if sample_bg.startswith("ENS") and not sample_query.startswith("ENS"):
                logger.warning(
                    "Background genes appear to be Ensembl IDs but query genes are symbols. "
                    "Using default background (all genes in gene sets)."
                )
                background_genes_list = None
    except Exception as e:
        logger.warning(f"Could not read background genes: {e}. Using default background.")
        background_genes_list = None

    # Perform GSEA using extracted genelist (gene symbols from summary)
    gsea = HypergeometricGSEA(genelist, background_list=background_genes_list)
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


def filter_by_overlap_ratio(df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """
    Filter enriched terms by overlap ratio.

    The overlap ratio measures how many genes in a term overlap with the query gene set.
    Terms with low overlap ratios tend to be generic/non-specific.

    Args:
        df: DataFrame with 'Overlap' column (format: "n/m")
        threshold: Minimum overlap ratio to keep (default: 0.25)

    Returns:
        Filtered DataFrame with only terms meeting threshold
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to filter_by_overlap_ratio")
        return df

    if 'Overlap' not in df.columns:
        logger.warning("'Overlap' column not found in DataFrame, skipping overlap ratio filter")
        return df

    def parse_overlap(overlap_str):
        if pd.isna(overlap_str):
            return None
        try:
            num, denom = str(overlap_str).split("/")
            return int(num) / int(denom) if int(denom) > 0 else None
        except (ValueError, ZeroDivisionError):
            return None

    df = df.copy()
    df['overlap_ratio'] = df['Overlap'].apply(parse_overlap)

    # Count terms that will be filtered
    pre_filter_count = len(df)
    filtered = df[df['overlap_ratio'] >= threshold].drop(columns=['overlap_ratio'])
    post_filter_count = len(filtered)

    logger.info(f"Overlap filter: {pre_filter_count} → {post_filter_count} terms (threshold={threshold})")

    return filtered