"""
Collect Context Module for GeneInsight

This module handles collecting context for gene sets by analyzing clustered topics,
generating relevant headings and subheadings using OpenAI API calls, and leveraging
parallel processing.
"""

import os
import logging
import dotenv
import pandas as pd
import random
import re
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from joblib import Parallel, delayed
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()

# ------------------------------
# Global Prompt Templates
# ------------------------------
PROMPT_MAIN_HEADING = """\
You are an experienced bioinformatician interpreting software output. This output from a gene set enrichment analysis shows enriched pathways 
from literature and ontology databases for a user-submitted query gene set. 
Generate a topic heading of fewer than 10 words that broadly summarizes 
the main biological theme of the output.

ENRICHMENT DATA:
{transcript}
"""

PROMPT_MAIN_HEADING_TEXT = """\
You are an experienced bioinformatician interpreting software output. This is the output from a gene set enrichment software, showing enriched pathways (references) 
from the literature and ontology databases about a user-submitted query gene set. 
Summarize what these pathways suggest about the role of this gene set in **two sentences**:
1. In the first sentence, describe the association between the gene set and the pathways, focusing on enriched biological processes or pathways.
2. In the second sentence, hypothesize about the potential biological role or mechanisms of the genes in this gene set, referencing specific pathways or processes where possible.

Use precise, simple, and accessible language that a general scientist can easily understand.

ENRICHMENT DATA:
{transcript}
"""

PROMPT_SUBHEADING_TEXT = """\
You are an experienced bioinformatician interpreting software output. This is the output from a gene set enrichment software, showing enriched pathways (references) 
from the literature and ontology databases about a user-submitted query gene set. 
What is the biological insight suggested about this gene set in **two sentences**?
1. In the first sentence, describe the association between the gene set and the pathways, highlighting the enriched biological processes or systems.
2. In the second sentence, hypothesize about the potential role or mechanisms of the genes in this gene set, referencing specific pathways or processes where possible.

Use clear, precise, and accessible language that a general scientist can easily understand.

ENRICHMENT DATA:
{transcript}
"""

# ------------------------------
# Pydantic Models
# ------------------------------
class TopicHeading(BaseModel):
    topic: str = Field(..., description="A title for the topic")

class TopicText(BaseModel):
    topic: str = Field(..., description="A paragraph of text for the topic, which is 2 sentences long")

# ------------------------------
# API Caller (OpenAI only)
# ------------------------------
def call_api(
    messages: list,
    response_model: BaseModel,
    service: str,
    api_key: str,
    model: str,
    base_url: str = None,
    max_retries: int = 10
) -> str:
    """
    Generic API caller that supports both OpenAI and Ollama.
    """
    try:
        # Choose client based on the service parameter
        if service.lower() == "openai":
            openai_client = OpenAI(api_key=api_key)
            client = instructor.from_openai(openai_client, mode=instructor.Mode.TOOLS)
        elif service.lower() == "ollama":
            # For Ollama, if no API key is provided, fallback to "ollama"
            ollama_client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
            client = instructor.from_openai(ollama_client, mode=instructor.Mode.TOOLS)
        else:
            raise ValueError(f"Unsupported service: {service}")
        
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries
        )
        return resp.topic
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return ""


# ------------------------------
# Wrapper Functions for API Calls
# ------------------------------
def produce_main_heading(transcript: str, service: str, api_key: str, model: str, base_url: str = None) -> str:
    messages = [
        {"role": "user", "content": PROMPT_MAIN_HEADING.format(transcript=transcript)}
    ]
    return call_api(messages, TopicHeading, service, api_key, model, base_url)

def produce_main_heading_text(transcript: str, service: str, api_key: str, model: str, base_url: str = None) -> str:
    messages = [
        {"role": "user", "content": PROMPT_MAIN_HEADING_TEXT.format(transcript=transcript)}
    ]
    return call_api(messages, TopicText, service, api_key, model, base_url)

def produce_subheading_text(transcript: str, service: str, api_key: str, model: str, base_url: str = None) -> str:
    messages = [
        {"role": "user", "content": PROMPT_SUBHEADING_TEXT.format(transcript=transcript)}
    ]
    return call_api(messages, TopicText, service, api_key, model, base_url)

# ------------------------------
# Parallel Processing Helpers
# ------------------------------
def parallel_subheading_text(contexts, service, api_key, model, base_url, n_jobs):
    """
    Parallelize calls to produce_subheading_text() over unique contexts.
    Returns a dictionary mapping each context -> subheading text.
    """
    def _worker(ctx):
        #logging.info(f"Calling API for context: {ctx}")
        return ctx, produce_subheading_text(ctx, service, api_key, model, base_url)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(ctx) for ctx in tqdm(contexts, desc="Processing contexts")
    )
    return dict(results)

def parallel_main_heading(cluster_data, service, api_key, model, base_url, n_jobs):
    """
    Parallelize calls to produce_main_heading() for each cluster.
    cluster_data is a list of tuples (cluster_id, transcript).
    Returns list of dicts with keys: cluster, transcript, heading.
    """
    def _worker(cid, transcript):
        heading = produce_main_heading(transcript, service, api_key, model, base_url)
        #logging.info(f"Cluster {cid} heading: {heading}")
        return {"cluster": cid, "transcript": transcript, "heading": heading}
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(cid, tstr) for cid, tstr in tqdm(cluster_data, desc="Processing main headings")
    )
    return results

def parallel_main_heading_text(headings_df, subheading_df, service, api_key, model, base_url, n_jobs):
    """
    Parallelize calls to produce_main_heading_text() for each cluster.
    Gathers subheading texts for each cluster to create a combined transcript.
    """
    def _worker(row):
        cluster_id = row["cluster"]
        cluster_subs = subheading_df[subheading_df["Cluster"] == cluster_id]
        combined_list = [
            f"Subsection title: {subrow['query']}\nReferences: {subrow['subheading_text']}\n"
            for _, subrow in cluster_subs.iterrows()
        ]
        combined_text = "\n".join(combined_list)
        #logging.info(f"Calling API for main heading text for cluster {cluster_id}")
        return produce_main_heading_text(combined_text, service, api_key, model, base_url)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(row) for _, row in tqdm(headings_df.iterrows(), total=headings_df.shape[0],
                                                 desc="Processing main heading texts")
    )
    return results

# ------------------------------
# Main generate_context Function
# ------------------------------
def generate_context(summary_path, clustered_topics_path, output_headings_path, output_subheadings_path, 
                     service="openai", api_key=None, model="gpt-4o-mini", base_url=None, n_jobs=10):
    """
    Generate context for gene sets by analyzing clustered topics with OpenAI.
    
    This function integrates:
      - Reading the input CSVs (summary and clustered topics)
      - Merging data for subheading generation
      - Parallel API calls to generate subheadings, main headings, and main heading texts
      - Saving the output CSV files
    
    Args:
        summary_path (str): Path to the summary CSV file. Expected to contain columns: "query", "unique_genes", "context"
        clustered_topics_path (str): Path to the clustered topics CSV file. Expected to contain columns: "Term", "Cluster"
        output_headings_path (str): Path to save the generated headings CSV file.
        output_subheadings_path (str): Path to save the generated subheadings CSV file.
        service (str): Service to use for API calls (only "openai" is supported).
        api_key (str): API key for OpenAI.
        model (str): Model to use for generation.
        base_url (str, optional): Base URL if needed by the API.
        n_jobs (int): Number of parallel workers.
    
    Returns:
        headings_df, subheadings_df (as pandas DataFrames)
    """
    logging.info("Generating context from clustered topics using OpenAI")
    
    dotenv.load_dotenv()
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("API key not provided or found in environment for OpenAI.")
        raise ValueError("API key not provided")
    
    # Read input files
    try:
        summary_df = pd.read_csv(summary_path)
        clustered_df = pd.read_csv(clustered_topics_path)
    except Exception as e:
        logging.error(f"Error reading input files: {e}")
        raise

    # Check for required columns in summary_df
    for col in ["query", "unique_genes", "context"]:
        if col not in summary_df.columns:
            logging.error(f"Required column '{col}' not found in summary CSV.")
            raise ValueError(f"Required column '{col}' not found in summary CSV.")
    
    # Check for required columns in clustered_df
    for col in ["Term", "Cluster"]:
        if col not in clustered_df.columns:
            logging.error(f"Required column '{col}' not found in clustered topics CSV.")
            raise ValueError(f"Required column '{col}' not found in clustered topics CSV.")
    
    # Merge summary with clustered topics for subheading processing
    merged_df = summary_df.loc[summary_df["query"].isin(clustered_df["Term"])].merge(
        clustered_df, left_on="query", right_on="Term", how="left"
    )
    
    # Generate subheadings in parallel using the "context" column for additional background
    unique_contexts = merged_df["context"].unique()
    context_map = parallel_subheading_text(
        contexts=unique_contexts,
        service=service,
        api_key=api_key,
        model=model,
        base_url=base_url,
        n_jobs=n_jobs
    )
    
    # Create subheadings DataFrame including unique_genes and additional background
    subheading_df = merged_df[["query", "context", "unique_genes", "Cluster"]].drop_duplicates().copy()
    subheading_df["subheading_text"] = subheading_df["context"].map(context_map)
    
    # Add reference dictionary if available
    if {"reference_description", "reference_term", "reference_genes"}.issubset(merged_df.columns):
        ref_dict_map = (
            merged_df
            .groupby("query")[["reference_description", "reference_term", "reference_genes"]]
            .apply(lambda g: dict(
                zip(
                    g["reference_description"].apply(lambda x: re.sub(r'\(\d{4}\)', '', x)).astype(str) + 
                    " (" + g["reference_term"].astype(str) + ")",
                    g["reference_genes"]
                )
            ))
            .to_dict()
        )
        subheading_df["ref_dict"] = subheading_df["query"].map(ref_dict_map)
    else:
        subheading_df["ref_dict"] = None
    subheading_df = subheading_df[["query", "subheading_text", "context", "unique_genes", "ref_dict", "Cluster"]]
    
    # Process main headings in parallel
    cluster_data = []
    for cluster_id, grp in merged_df.groupby("Cluster"):
        queries = grp["query"].unique()
        transcript = "\n".join(queries)
        cluster_data.append((cluster_id, transcript))
    
    headings_list = parallel_main_heading(
        cluster_data=cluster_data,
        service=service,
        api_key=api_key,
        model=model,
        base_url=base_url,
        n_jobs=n_jobs
    )
    headings_df = pd.DataFrame(headings_list, columns=["cluster", "transcript", "heading"])
    
    # Generate main heading texts in parallel
    main_heading_texts = parallel_main_heading_text(
        headings_df=headings_df,
        subheading_df=subheading_df,
        service=service,
        api_key=api_key,
        model=model,
        base_url=base_url,
        n_jobs=n_jobs
    )
    headings_df["main_heading_text"] = main_heading_texts
    
    # Save output CSV files
    headings_df.to_csv(output_headings_path, index=False)
    logging.info(f"Saved headings to {output_headings_path}")
    
    subheading_df.to_csv(output_subheadings_path, index=False)
    logging.info(f"Saved subheadings to {output_subheadings_path}")
    
    return headings_df, subheading_df

# ------------------------------
# Optional CLI Interface
# ------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate context for gene sets using OpenAI API.")
    parser.add_argument("--input_summary", type=str, required=True, help="Path to the input summary CSV file.")
    parser.add_argument("--input_clustered", type=str, required=True, help="Path to the input clustered topics CSV file.")
    parser.add_argument("--output_headings", type=str, required=True, help="Path to the output headings CSV file.")
    parser.add_argument("--output_subheadings", type=str, required=True, help="Path to the output subheadings CSV file.")
    parser.add_argument("--service", type=str, default="openai", help="Service to use (only openai is supported).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for generation.")
    parser.add_argument("--base_url", type=str, default=None, help="Optional base URL for API calls.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers.")
    args = parser.parse_args()
    
    generate_context(
        summary_path=args.input_summary,
        clustered_topics_path=args.input_clustered,
        output_headings_path=args.output_headings,
        output_subheadings_path=args.output_subheadings,
        service=args.service,
        api_key=None,
        model=args.model,
        base_url=args.base_url,
        n_jobs=args.n_jobs
    )