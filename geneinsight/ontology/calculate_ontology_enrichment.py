import os
import sys
import logging
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import gseapy as gp
import argparse
import ast

# ------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class HypergeometricGSEA:
    """
    A class to perform hypergeometric Gene Set Enrichment Analysis (GSEA).
    """
    def __init__(self, genelist, background_list=None):
        """
        Parameters
        ----------
        genelist : list
            The list of genes to test.
        background_list : list
            The background gene population.
        """
        self.genelist = genelist
        self.background_list = background_list
        logger.debug(
            f"Initialized HypergeometricGSEA with {len(genelist)} genes and "
            f"{'no' if background_list is None else len(background_list)} background genes."
        )

    def perform_hypergeometric_gsea(self, geneset_dict):
        """
        Perform hypergeometric GSEA using gseapy.

        Parameters
        ----------
        geneset_dict : dict
            A dictionary where keys are set names and values are gene lists.

        Returns
        -------
        pd.DataFrame
            A DataFrame with GSEA results.
        """
        logger.debug("Starting hypergeometric GSEA using gseapy.enrich()")
        enr = gp.enrich(
            gene_list=self.genelist,
            gene_sets=geneset_dict,
            background=self.background_list,
            outdir=None,
            verbose=True
        )
        logger.debug("Hypergeometric GSEA completed.")
        return enr.res2d


class OntologyReader:
    """
    A class that reads an ontology file and constructs a dictionary mapping ontology terms 
    to their associated genes.
    """
    def __init__(self, file_path_ontology, ontology_name):
        self.file_path = file_path_ontology
        self.name = ontology_name
        logger.info(f"Initializing OntologyReader for {ontology_name} from {file_path_ontology}")
        self.gene_dict = self.read_ontology_file_to_dict(self.file_path)

    def read_ontology_file_to_dict(self, filepath):
        logger.debug(f"Reading ontology file: {filepath}")
        with open(filepath, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue

            # If there's a "\t\t", assume that is the correct format
            if "\t\t" in line:
                parts = line.split("\t\t", 1)
                name = parts[0].strip()
                genes_part = parts[1].strip() if len(parts) > 1 else ""
                genes = genes_part.split("\t") if genes_part else []
            else:
                # Fallback: attempt to split on the first tab
                fallback_parts = line.split("\t", 1)
                name = fallback_parts[0].strip()
                if len(fallback_parts) > 1:
                    genes = fallback_parts[1].strip().split()
                else:
                    # No second part => no genes
                    genes = []

            data.append([name, genes])

        df = pd.DataFrame(data, columns=['name', 'gene_list'])
        gene_dict = df.set_index('name')['gene_list'].to_dict()
        logger.debug(f"Read {len(gene_dict)} ontology terms from {os.path.basename(filepath)}.")
        return gene_dict



class RAGModuleGSEAPY:
    """
    A class that orchestrates GSEA and retrieves top documents based on a query embedding.
    GSEA is run when get_top_documents is called.
    """
    def __init__(self, ontology_object_list):
        """
        Parameters
        ----------
        ontology_object_list : list
            A list of OntologyReader objects.
        """
        self.ontologies = ontology_object_list
        logger.info(f"Initializing RAGModuleGSEAPY with {len(ontology_object_list)} ontologies.")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.hypergeometric_gsea_obj = None
        self.enrichr_results = None

    def query_gse_single_ontology(self, ontology_object):
        """
        Perform GSEA on a single ontology and return the results.

        Parameters
        ----------
        ontology_object : OntologyReader

        Returns
        -------
        pd.DataFrame
            The GSEA result DataFrame for that ontology.
        """
        logger.debug(f"Running hypergeometric GSEA for ontology: {ontology_object.name}")
        try:
            res_df = self.hypergeometric_gsea_obj.perform_hypergeometric_gsea(ontology_object.gene_dict)
            res_df["Gene_set"] = ontology_object.name
        except Exception as e:
            logger.warning(f"Failed to enrich for ontology {ontology_object.name}: {e}")
            res_df = pd.DataFrame()
        return res_df

    def get_enrichment(self, fdr_threshold=0.05):
        """
        Run GSEA on all provided ontologies and filter the results by the given FDR threshold.

        Parameters
        ----------
        fdr_threshold : float, optional (default=0.1)
            The FDR (Adjusted P-value) threshold to filter results.

        Returns
        -------
        pd.DataFrame
            A DataFrame of combined results from all ontologies.
        """
        logger.info(f"Running enrichment for {len(self.ontologies)} ontologies at FDR < {fdr_threshold}")
        results = []
        for ontology in self.ontologies:
            res_df = self.query_gse_single_ontology(ontology)
            if not res_df.empty:
                results.append(res_df)

        if not results:
            logger.info("No results found for any ontology before FDR filtering.")
            self.enrichr_results = pd.DataFrame()
            return self.enrichr_results

        combined_res_df = pd.concat(results, ignore_index=True)
        logger.debug(f"Combined results: {combined_res_df.shape[0]} rows total.")
        combined_res_df = combined_res_df[combined_res_df['Adjusted P-value'] < fdr_threshold]

        if combined_res_df.empty:
            logger.info("All results filtered out by FDR threshold. No significant terms remain.")
            self.enrichr_results = pd.DataFrame()
        else:
            logger.debug(f"Retained {combined_res_df.shape[0]} rows after FDR filtering.")
            self.enrichr_results = combined_res_df.drop_duplicates()

        return self.enrichr_results

    def get_top_documents(self, query, gene_list, background_list=None, N=5, fdr_threshold=0.05):
        """
        The main function that orchestrates the GSEA and document retrieval.
        Retrieve the top N documents (terms) related to the query based on cosine similarity.

        Parameters
        ----------
        query : str
            The query text.
        gene_list : list
            The list of genes for enrichment analysis.
        background_list : list, optional
            The background gene population.
        N : int, optional (default=5)
            Number of top documents to return.
        fdr_threshold : float, optional (default=0.1)
            The FDR threshold to filter enrichment results.

        Returns
        -------
        tuple
            (top_results_indices, extracted_items, enrichr_results, enrichr_df_filtered, formatted_output)
        """
        logger.info(f"Running GSEA + retrieval for query: '{query}' with top N={N}, FDR < {fdr_threshold}")
        self.hypergeometric_gsea_obj = HypergeometricGSEA(gene_list, background_list)
        self.get_enrichment(fdr_threshold=fdr_threshold)  # Updates self.enrichr_results

        # If no significant enrichment results, return early
        if self.enrichr_results is None or self.enrichr_results.empty:
            logger.info("No significant terms found. Returning empty results.")
            return (
                [],
                "No significant terms found at the specified FDR threshold.",
                pd.DataFrame(),
                pd.DataFrame(),
                "No significant terms found at the specified FDR threshold."
            )

        # Compute embeddings
        documents = self.enrichr_results["Term"].tolist()
        logger.debug(f"Computing embeddings for {len(documents)} documents.")
        document_embeddings = self.embedder.encode(documents, convert_to_tensor=True, show_progress_bar=False)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)

        # Cosine similarity
        cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

        # Determine top-k
        k_val = min(N, len(documents))
        if k_val == 0:
            logger.warning("No documents found or insufficient documents to compute topk.")
            return (
                [],
                "No significant terms found at the specified FDR threshold.",
                pd.DataFrame(),
                pd.DataFrame(),
                "No significant terms found at the specified FDR threshold."
            )

        try:
            top_results_indices = torch.topk(cosine_scores, k=k_val).indices.tolist()
        except RuntimeError as e:
            logger.warning(f"RuntimeError occurred in topk computation: {e}")
            top_results_indices = []
            err_msg = f"RuntimeError occurred in topk computation: {e}"
            return (
                top_results_indices,
                err_msg,
                self.enrichr_results,
                pd.DataFrame(),
                err_msg
            )

        logger.debug(f"Top {N} indices: {top_results_indices}")

        # Format the top documents
        extracted_items, enrichr_df_filtered = self.format_top_documents(top_results_indices)

        # Store cosine scores in the overall results
        self.enrichr_results["cosine_score"] = cosine_scores.tolist()

        # Return everything
        return top_results_indices, extracted_items, self.enrichr_results, enrichr_df_filtered, extracted_items

    def format_top_documents(self, top_results_indices):
        """
        Format the top documents into a readable list with their FDR and sets.
        Disallow negative indices to treat them as out-of-bounds.
        """
        # If we have no data or an empty index list, return early
        if (
            self.enrichr_results is None
            or self.enrichr_results.empty
            or not top_results_indices
        ):
            return "", pd.DataFrame()

        # Explicitly disallow negative indices
        if any(idx < 0 for idx in top_results_indices):
            logger.warning(
                "IndexError: Negative indices are disallowed. Returning empty results."
            )
            return "", pd.DataFrame()

        # Try to index the results
        try:
            df_filtered = self.enrichr_results.iloc[top_results_indices]
        except IndexError:
            logger.warning(
                "IndexError: Attempted to index out-of-bounds in enrichr_results. Returning empty results."
            )
            return "", pd.DataFrame()

        logger.debug(f"Formatting top {len(top_results_indices)} documents.")
        output = []
        for _, row in df_filtered.iterrows():
            output.append(
                f"* `{row['Gene_set']}: {row['Term']} - FDR: {row['Adjusted P-value']:.4f}`"
            )
        formatted_output = "\n".join(output)
        return formatted_output, df_filtered


def main():
    parser = argparse.ArgumentParser(description="Perform ontology enrichment analysis.")
    parser.add_argument("--summary_csv", required=True, help="Path to the summary CSV file.")
    parser.add_argument("--gene_origin", required=True, help="Path to the gene origin file.")
    parser.add_argument("--background_genes", required=True, help="Path to the background genes file.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    parser.add_argument("--fdr_threshold", type=float, default=0.1, help="FDR threshold for filtering results.")
    parser.add_argument("--ontology_folder", required=True, help="Path to the folder containing ontology files.")
    parser.add_argument("--filter_csv", required=True, help="Path to the CSV file used to filter queries.")
    args = parser.parse_args()

    logger.info("Starting main script.")

    # Load ontologies dynamically from the specified folder
    ontologies = []
    try:
        logger.info(f"Reading ontology files from folder: {args.ontology_folder}")
        for filename in os.listdir(args.ontology_folder):
            file_path = os.path.join(args.ontology_folder, filename)
            # Skip directories and hidden files, if any
            if not os.path.isfile(file_path) or filename.startswith('.'):
                logger.debug(f"Skipping {filename} - not a valid file.")
                continue

            base_name, _ = os.path.splitext(filename)
            try:
                ontology_obj = OntologyReader(file_path, base_name)
                ontologies.append(ontology_obj)
            except Exception as read_err:
                logger.warning(f"Failed to read {file_path}. Skipping. Error: {read_err}")

    except Exception as e:
        logger.error(f"Error reading ontology folder '{args.ontology_folder}': {e}")
        sys.exit(1)

    # If no ontology files were successfully loaded, exit.
    if not ontologies:
        logger.error(f"No valid ontology files found in {args.ontology_folder}. Exiting.")
        sys.exit(1)

    # Instantiate the GSEA module with all loaded ontologies
    rag_module = RAGModuleGSEAPY(ontologies)

    logger.info(f"Reading summary CSV: {args.summary_csv}")
    filter_df = pd.read_csv(args.filter_csv)
    valid_terms = set(filter_df["Term"].unique())
    df = pd.read_csv(args.summary_csv)
    df = df[df["query"].isin(valid_terms)]
    df = df.drop_duplicates(subset=["query"])
    results = []

    logger.info(f"Processing {df.shape[0]} rows from the summary CSV.")
    for index, row in df.iterrows():
        query = row["query"]
        unique_genes = list(ast.literal_eval(row["unique_genes"]).keys())

        (
            top_results_indices, 
            extracted_items, 
            enrichr_results, 
            enrichr_df_filtered, 
            formatted_output
        ) = rag_module.get_top_documents(
            query=query,
            gene_list=unique_genes,
            background_list=args.background_genes,
            N=5,
            fdr_threshold=args.fdr_threshold
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

    logger.info(f"Writing final results to {args.output_csv}")
    final_df = pd.DataFrame(results)
    final_df.to_csv(args.output_csv, index=False)
    logger.info("Script completed successfully.")


if __name__ == "__main__":
    main()
