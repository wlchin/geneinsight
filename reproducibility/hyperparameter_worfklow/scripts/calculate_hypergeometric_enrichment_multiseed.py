#!/usr/bin/env python3

import argparse
import ast
import os
import pandas as pd
import gseapy as gp

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
        enr = gp.enrich(
            gene_list=self.genelist,
            gene_sets=geneset_dict,
            background=self.background_list,
            outdir=None,
            verbose=True
        )
        return enr.res2d

def main():
    parser = argparse.ArgumentParser(
        description="Perform hypergeometric GSEA on a summary DataFrame."
    )
    parser.add_argument(
        "--df",
        required=True,
        help="Path to the CSV file containing the summary (e.g., topic_model_seed_10__summary.csv)."
    )
    parser.add_argument(
        "--gene_origin",
        required=True,
        help="Path to the gene origin file (e.g., Patient_Day24_DifferenceBetweenClones.txt)."
    )
    parser.add_argument(
        "--background_genes",
        required=True,
        help="Path to the background genes file (e.g., Patient_Day24-BackgroundList.txt)."
    )
    parser.add_argument(
        "--output_csv",
        default="testdf.csv",
        help="Path to the output CSV file. Defaults to 'testdf.csv'."
    )
    parser.add_argument(
        "--pvalue_threshold",
        type=float,
        default=0.01,
        help="Adjusted P-value threshold for filtering results. Defaults to 0.01."
    )

    args = parser.parse_args()

    # Read the input DataFrame
    df = pd.read_csv(args.df)
    
    # Extract the minor topics
    minor_topics = df[["query", "unique_genes"]].drop_duplicates()

    # Further filter to keep only the first query for each unique set of genes
    minor_topics = minor_topics.drop_duplicates(subset=["unique_genes"], keep="first")

    # Prepare the gene query dictionary
    genequery = {}
    for index, row in minor_topics.iterrows():
        query_name = row["query"]
        # Convert unique_genes string to dictionary and extract keys
        geneset = list(ast.literal_eval(row["unique_genes"]).keys())
        genequery[query_name] = geneset

    # Read the gene origin file
    gene_origin = pd.read_csv(args.gene_origin, header=None)

    # Read the background genes file
    background_genes = pd.read_csv(args.background_genes, header=None)
    background_genes = background_genes[0].tolist()

    # Perform GSEA
    x = HypergeometricGSEA(gene_origin[0].tolist(), background_list=background_genes)
    result = x.perform_hypergeometric_gsea(genequery)

    # Filter for significant results
    result = result[result["Adjusted P-value"] < args.pvalue_threshold]

    # Save to CSV
    result.to_csv(args.output_csv, index=False)
    print(f"Filtered results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()