import os
import logging
import pandas as pd
import argparse
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any, Union, Tuple
import importlib.resources as pkg_resources

from .calculate_ontology_enrichment import RAGModuleGSEAPY, OntologyReader, HypergeometricGSEA
from .get_ontology_dictionary import process_ontology_enrichment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class OntologyWorkflow:
    """
    A class that orchestrates the ontology enrichment analysis workflow.
    This mimics the functionality of the Snakemake workflow rules.
    """
    def __init__(
        self, 
        ontology_folder: Optional[str] = None,
        fdr_threshold: float = 0.1,
        use_temp_files: bool = False
    ):
        """
        Initialize the workflow with optional parameters.
        
        Parameters
        ----------
        ontology_folder : str, optional
            Path to the folder containing ontology files. If None, uses the default package location.
        fdr_threshold : float, optional
            The FDR threshold for filtering enrichment results.
        """
        self.fdr_threshold = fdr_threshold
        self.use_temp_files = use_temp_files
        self.temp_dir = None
        
        if self.use_temp_files:
            self.temp_dir = tempfile.mkdtemp(prefix="geneinsight_ontology_")
            
        # If no ontology folder is provided, use the package default
        if ontology_folder is None:
            try:
                # Get the path to the package's ontology_folders
                import geneinsight.ontology.ontology_folders as ontology_pkg
                self.ontology_folder = str(pkg_resources.files(ontology_pkg))
                logger.info(f"Using default ontology folder: {self.ontology_folder}")
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to locate default ontology folder: {e}")
                raise ValueError("No ontology folder provided and default not found")
        else:
            self.ontology_folder = ontology_folder
            
        # Validate the ontology folder exists
        if not os.path.isdir(self.ontology_folder):
            raise ValueError(f"Ontology folder does not exist: {self.ontology_folder}")

    def run_ontology_enrichment(
        self,
        summary_csv: str,
        gene_origin: str,
        background_genes: str,
        filter_csv: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        Run the ontology enrichment analysis step.
        
        Parameters
        ----------
        summary_csv : str
            Path to the summary CSV file.
        gene_origin : str
            Path to the gene origin file.
        background_genes : str
            Path to the background genes file.
        filter_csv : str
            Path to the CSV file used to filter queries.
        output_csv : str
            Path to the output CSV file.
            
        Returns
        -------
        pd.DataFrame
            The results dataframe.
        """
        logger.info("Starting ontology enrichment analysis")
        
        # Load ontologies from the specified folder
        ontologies = []
        try:
            logger.info(f"Reading ontology files from folder: {self.ontology_folder}")
            for filename in os.listdir(self.ontology_folder):
                file_path = os.path.join(self.ontology_folder, filename)
                # Skip directories and hidden files
                if not os.path.isfile(file_path) or filename.startswith('.'):
                    continue

                base_name, _ = os.path.splitext(filename)
                try:
                    ontology_obj = OntologyReader(file_path, base_name)
                    ontologies.append(ontology_obj)
                    logger.debug(f"Loaded ontology: {base_name}")
                except Exception as read_err:
                    logger.warning(f"Failed to read {file_path}. Skipping. Error: {read_err}")
        except Exception as e:
            logger.error(f"Error reading ontology folder '{self.ontology_folder}': {e}")
            raise

        if not ontologies:
            logger.error(f"No valid ontology files found in {self.ontology_folder}.")
            raise ValueError("No valid ontology files found")

        # Create the RAG module with the loaded ontologies
        rag_module = RAGModuleGSEAPY(ontologies)
        
        # Process the input data
        logger.info(f"Reading summary CSV: {summary_csv}")
        filter_df = pd.read_csv(filter_csv)
        valid_terms = set(filter_df["Term"].unique())
        df = pd.read_csv(summary_csv)
        df = df[df["query"].isin(valid_terms)]
        df = df.drop_duplicates(subset=["query"])
        
        # Read background genes if provided as a file path
        background_gene_list = None
        if isinstance(background_genes, str) and os.path.isfile(background_genes):
            with open(background_genes, 'r') as bg_file:
                background_gene_list = [line.strip() for line in bg_file if line.strip()]
            logger.info(f"Loaded {len(background_gene_list)} background genes")
        
        results = []
        import ast  # For parsing string representations of Python objects

        logger.info(f"Processing {df.shape[0]} rows from the summary CSV")
        for index, row in df.iterrows():
            query = row["query"]
            # Handle the case where unique_genes might be a string representation of a dict
            if isinstance(row["unique_genes"], str):
                unique_genes = list(ast.literal_eval(row["unique_genes"]).keys())
            else:
                unique_genes = list(row["unique_genes"].keys())

            (
                top_results_indices, 
                extracted_items, 
                enrichr_results, 
                enrichr_df_filtered, 
                formatted_output
            ) = rag_module.get_top_documents(
                query=query,
                gene_list=unique_genes,
                background_list=background_gene_list,
                N=5,
                fdr_threshold=self.fdr_threshold
            )

            results.append({
                "query": query,
                "top_results_indices": top_results_indices,
                "extracted_items": extracted_items,
                "enrichr_results": (
                    enrichr_results.to_dict(orient="records") if not enrichr_results.empty else []
                ),
                "enrichr_df_filtered": (
                    enrichr_df_filtered.to_dict(orient="records") if not enrichr_df_filtered.empty else []
                ),
                "formatted_output": formatted_output
            })

        # Create final dataframe and save results
        final_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Wrote enrichment results to {output_csv}")
        
        return final_df

    def create_ontology_dictionary(
        self,
        input_csv: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        Convert the ontology enrichment results into a dictionary format.
        
        Parameters
        ----------
        input_csv : str
            Path to the input CSV file with enrichment results.
        output_csv : str
            Path to the output CSV file for the dictionary.
            
        Returns
        -------
        pd.DataFrame
            The ontology dictionary dataframe.
        """
        logger.info(f"Processing ontology enrichment results: {input_csv}")
        
        # Process the enrichment results
        results_df = process_ontology_enrichment(input_csv)
        
        # Save to output file
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Wrote ontology dictionary to {output_csv}")
        
        return results_df
        
    def process_dataframes(
        self,
        summary_df: pd.DataFrame,
        clustered_df: pd.DataFrame,
        gene_list_path: str,
        background_genes_path: str,
        output_dir: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process dataframes directly without writing to intermediate files.
        This is designed to be used within a pipeline that already has dataframes in memory.
        
        Parameters
        ----------
        summary_df : pd.DataFrame
            The summary dataframe containing gene information.
        clustered_df : pd.DataFrame
            The dataframe containing clustered topics.
        gene_list_path : str
            Path to the query gene list file.
        background_genes_path : str
            Path to the background genes file.
        output_dir : str, optional
            Directory to save the output files if desired.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The enrichment results and ontology dictionary dataframes.
        """
        logger.info("Processing dataframes for ontology enrichment")
        
        # Create temporary files if needed
        if self.use_temp_files:
            temp_dir = self.temp_dir or tempfile.mkdtemp(prefix="geneinsight_ontology_temp_")
            summary_path = os.path.join(temp_dir, "summary.csv")
            filter_path = os.path.join(temp_dir, "clustered_topics.csv")
            enrichment_path = os.path.join(temp_dir, "enrichment_results.csv")
            dict_path = os.path.join(temp_dir, "ontology_dict.csv")
            
            # Write dataframes to temporary files
            summary_df.to_csv(summary_path, index=False)
            clustered_df.to_csv(filter_path, index=False)
            
            # Run the enrichment analysis
            enrichment_df = self.run_ontology_enrichment(
                summary_csv=summary_path,
                gene_origin=gene_list_path,
                background_genes=background_genes_path,
                filter_csv=filter_path,
                output_csv=enrichment_path
            )
            
            # Create the ontology dictionary
            ontology_dict_df = self.create_ontology_dictionary(
                input_csv=enrichment_path,
                output_csv=dict_path
            )
            
            # If output_dir is provided, save the files there too
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                final_enrichment_path = os.path.join(output_dir, "ontology_enrichment.csv")
                final_dict_path = os.path.join(output_dir, "ontology_dict.csv")
                
                enrichment_df.to_csv(final_enrichment_path, index=False)
                ontology_dict_df.to_csv(final_dict_path, index=False)
                logger.info(f"Saved final enrichment to {final_enrichment_path}")
                logger.info(f"Saved final ontology dictionary to {final_dict_path}")
                
            # Clean up temporary files if we created them
            try:
                if self.temp_dir and os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary directory: {e}")
                
        else:
            # Process the dataframes in memory - more complex but avoids disk I/O
            logger.info("Processing ontology enrichment in memory")
            
            # Load ontologies
            ontologies = []
            try:
                logger.info(f"Reading ontology files from folder: {self.ontology_folder}")
                for filename in os.listdir(self.ontology_folder):
                    file_path = os.path.join(self.ontology_folder, filename)
                    if not os.path.isfile(file_path) or filename.startswith('.'):
                        continue
                    base_name, _ = os.path.splitext(filename)
                    try:
                        ontology_obj = OntologyReader(file_path, base_name)
                        ontologies.append(ontology_obj)
                    except Exception as read_err:
                        logger.warning(f"Failed to read {file_path}. Skipping. Error: {read_err}")
            except Exception as e:
                logger.error(f"Error reading ontology folder '{self.ontology_folder}': {e}")
                raise

            if not ontologies:
                logger.error(f"No valid ontology files found in {self.ontology_folder}.")
                raise ValueError("No valid ontology files found")
                
            # Create the RAG module with the loaded ontologies
            rag_module = RAGModuleGSEAPY(ontologies)
            
            # Filter the summary_df using terms from clustered_df
            valid_terms = set(clustered_df["Term"].unique())
            filtered_summary_df = summary_df[summary_df["query"].isin(valid_terms)]
            filtered_summary_df = filtered_summary_df.drop_duplicates(subset=["query"])
            
            # Read background genes if provided as a file path
            background_gene_list = None
            if isinstance(background_genes_path, str) and os.path.isfile(background_genes_path):
                with open(background_genes_path, 'r') as bg_file:
                    background_gene_list = [line.strip() for line in bg_file if line.strip()]
                logger.info(f"Loaded {len(background_gene_list)} background genes")
            
            # Process each row in the filtered summary dataframe
            results = []
            import ast
            
            logger.info(f"Processing {filtered_summary_df.shape[0]} rows from the summary dataframe")
            for index, row in filtered_summary_df.iterrows():
                query = row["query"]
                # Handle the case where unique_genes might be a string representation of a dict
                if isinstance(row["unique_genes"], str):
                    unique_genes = list(ast.literal_eval(row["unique_genes"]).keys())
                else:
                    unique_genes = list(row["unique_genes"].keys())

                (
                    top_results_indices, 
                    extracted_items, 
                    enrichr_results, 
                    enrichr_df_filtered, 
                    formatted_output
                ) = rag_module.get_top_documents(
                    query=query,
                    gene_list=unique_genes,
                    background_list=background_gene_list,
                    N=5,
                    fdr_threshold=self.fdr_threshold
                )

                results.append({
                    "query": query,
                    "top_results_indices": top_results_indices,
                    "extracted_items": extracted_items,
                    "enrichr_results": (
                        enrichr_results.to_dict(orient="records") if not enrichr_results.empty else []
                    ),
                    "enrichr_df_filtered": (
                        enrichr_df_filtered.to_dict(orient="records") if not enrichr_df_filtered.empty else []
                    ),
                    "formatted_output": formatted_output
                })
            
            # Create the enrichment dataframe
            enrichment_df = pd.DataFrame(results)
            
            # Process the enrichment results to create the ontology dictionary
            ontology_dict_df = process_ontology_enrichment(enrichment_df)
            
            # If output_dir is provided, save the files there
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                final_enrichment_path = os.path.join(output_dir, "ontology_enrichment.csv")
                final_dict_path = os.path.join(output_dir, "ontology_dict.csv")
                
                enrichment_df.to_csv(final_enrichment_path, index=False)
                ontology_dict_df.to_csv(final_dict_path, index=False)
                logger.info(f"Saved final enrichment to {final_enrichment_path}")
                logger.info(f"Saved final ontology dictionary to {final_dict_path}")
        
        return enrichment_df, ontology_dict_df

    def run_full_workflow(
        self,
        gene_set: str,
        summary_csv: Optional[str] = None,
        gene_origin: Optional[str] = None,
        background_genes: Optional[str] = None,
        filter_csv: Optional[str] = None,
        output_dir: str = "results",
        return_results: bool = False
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Run the full ontology workflow, from enrichment to dictionary creation.
        
        Parameters
        ----------
        gene_set : str
            Name of the gene set being analyzed.
        summary_csv : str, optional
            Path to the summary CSV file. If None, uses default path based on gene_set.
        gene_origin : str, optional
            Path to the gene origin file. If None, uses default path based on gene_set.
        background_genes : str, optional
            Path to the background genes file. If None, uses default path.
        filter_csv : str, optional
            Path to the clustered topics CSV. If None, uses default path based on gene_set.
        output_dir : str, optional
            Directory for output files. Default is "results".
        return_results : bool, optional
            Whether to return the results dataframes.
            
        Returns
        -------
        dict or None
            If return_results is True, returns a dictionary with 'enrichment' and 'dictionary' keys.
        """
        # Set default paths if not provided
        if summary_csv is None:
            summary_csv = f"{output_dir}/summary/{gene_set}.csv"
        if gene_origin is None:
            gene_origin = f"data/{gene_set}.txt"
        if background_genes is None:
            background_genes = "data/BackgroundList.txt"
        if filter_csv is None:
            filter_csv = f"{output_dir}/clustered_topics/{gene_set}_clustered_topics.csv"
            
        # Create output paths
        enrichment_output = f"{output_dir}/ontology_results/{gene_set}_ontology_enrichment.csv"
        dictionary_output = f"{output_dir}/ontology_dict/{gene_set}_ontology_dict.csv"
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(enrichment_output), exist_ok=True)
        os.makedirs(os.path.dirname(dictionary_output), exist_ok=True)
        
        # Run the workflow steps
        enrichment_df = self.run_ontology_enrichment(
            summary_csv=summary_csv,
            gene_origin=gene_origin,
            background_genes=background_genes,
            filter_csv=filter_csv,
            output_csv=enrichment_output
        )
        
        dictionary_df = self.create_ontology_dictionary(
            input_csv=enrichment_output,
            output_csv=dictionary_output
        )
        
        logger.info(f"Completed ontology workflow for gene set: {gene_set}")
        
        if return_results:
            return {
                "enrichment": enrichment_df,
                "dictionary": dictionary_df
            }
        return None


def main():
    """Command line interface for the ontology workflow."""
    parser = argparse.ArgumentParser(description="Run ontology enrichment workflow")
    parser.add_argument("--gene_set", required=True, help="Name of the gene set to analyze")
    parser.add_argument("--summary_csv", help="Path to the summary CSV file")
    parser.add_argument("--gene_origin", help="Path to the gene origin file")
    parser.add_argument("--background_genes", help="Path to the background genes file")
    parser.add_argument("--filter_csv", help="Path to the CSV file with clustered topics")
    parser.add_argument("--ontology_folder", help="Path to the folder containing ontology files")
    parser.add_argument("--output_dir", default="results", help="Directory for output files")
    parser.add_argument("--fdr_threshold", type=float, default=0.1, help="FDR threshold for filtering results")
    parser.add_argument("--use_temp_files", action="store_true", help="Use temporary files for intermediate steps")
    args = parser.parse_args()
    
    # Initialize the workflow
    workflow = OntologyWorkflow(
        ontology_folder=args.ontology_folder,
        fdr_threshold=args.fdr_threshold,
        use_temp_files=args.use_temp_files
    )
    
    # Run the full workflow
    workflow.run_full_workflow(
        gene_set=args.gene_set,
        summary_csv=args.summary_csv,
        gene_origin=args.gene_origin,
        background_genes=args.background_genes,
        filter_csv=args.filter_csv,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
