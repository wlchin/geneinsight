import argparse
import pandas as pd
import os
from tqdm import tqdm
import logging
import json
import torch
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from dotenv import load_dotenv
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load the .env file
load_dotenv()

class RAGModule:
    def __init__(self, df, embeddings=None):
        logging.info("Initializing RAGModule...")
        self.documents = df["description"].to_list()
        self.df_metadata = df
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if embeddings is not None:
            logging.info("Using provided document embeddings.")
            self.document_embeddings = embeddings
        else:
            logging.info("Generating new embeddings for documents...")
            self.document_embeddings = self.embedder.encode(self.documents, convert_to_tensor=True)

        logging.info("RAGModule initialized successfully.")

    def get_top_documents(self, query, N=5):
        logging.info("Entering get_top_documents() with query='%s' and N=%d", query, N)
        query_embedding = self.embedder.encode(str(query), convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]
        top_results_indices = torch.topk(cosine_scores, k=N).indices.cpu()  # move to CPU before converting
        logging.info("Top documents indices: %s", top_results_indices.tolist())
        return top_results_indices

    def get_context(self, query, top_results_indices):
        logging.info("Entering get_context() with query='%s' and top_results_indices=%s", query, top_results_indices.tolist())
        context_documents = " ".join(
            [f"Document {i+1}: {self.documents[idx]}" for i, idx in enumerate(top_results_indices)]
        )
        context = (
            f"Title: {query}\n"
            f"Context: {context_documents}\n"
        )
        logging.info("Generated context for LLM.")
        return context

    def get_text_response(self, query, num_results=5):
        logging.info("Entering get_text_response() with query='%s'", query)
        top_results_indices = self.get_top_documents(query, N=num_results)
        context_documents = self.get_context(query, top_results_indices)
        response_text = ""
        response_dict = {
            "query": query,
            "context": context_documents,
            "text": response_text,
            "rouge": {}
        }
        logging.info("Returning from get_text_response() for query='%s'.", query)
        return response_dict, response_text

    def format_references_and_genes(self, indices):
        logging.info("Entering format_references_and_genes() with indices=%s", indices.tolist())
        df = self.df_metadata.iloc[indices].reset_index(drop=True)
        grouped = (
            df.groupby(["term", "description"])["inputGenes"]
            .apply(lambda x: ",".join(x))
            .reset_index()
        )
        
        references = []
        all_genes_list = []
        for _, row in grouped.iterrows():
            genes = [g.strip() for g in row["inputGenes"].split(",") if g.strip()]
            references.append({
                "term": row["term"],
                "description": row["description"],
                "genes": genes
            })
            all_genes_list.extend(genes)
        
        gene_counter = Counter(all_genes_list)
        unique_genes = dict(gene_counter)
        logging.info("References and unique genes formatted.")
        return references, unique_genes

    def get_summary_to_query(self, query, num_results=5):
        logging.info("Entering get_summary_to_query() with query='%s'", query)
        response_dict, _ = self.get_text_response(query, num_results=num_results)
        top_results_indices = self.get_top_documents(query, N=num_results)
        references, unique_genes = self.format_references_and_genes(top_results_indices)
        output = {
            "response": response_dict.get("text", ""),   
            "rouge": response_dict.get("rouge", {}),     
            "context": response_dict.get("context", ""), 
            "references": references,
            "unique_genes": unique_genes
        }
        logging.info("Exiting get_summary_to_query() with output keys=%s", list(output.keys()))
        return output

    def get_summary_to_query_df(self, query, num_results=5):
        logging.info("Entering get_summary_to_query_df() with query='%s'", query)
        output = self.get_summary_to_query(query, num_results=num_results)

        response_text = output.get("response", "")
        rouge = output.get("rouge", {})
        context_str = output.get("context", "")
        references = output.get("references", [])
        unique_genes = output.get("unique_genes", {})

        rouge1_f = rouge.get("rouge1", {}).get("fmeasure") if "rouge1" in rouge else None
        rouge2_f = rouge.get("rouge2", {}).get("fmeasure") if "rouge2" in rouge else None
        rougel_f = rouge.get("rougeL", {}).get("fmeasure") if "rougeL" in rouge else None

        unique_genes_str = json.dumps(unique_genes)

        rows = []
        for ref in references:
            rows.append({
                "query": query,
                "context": context_str,  
                "response": response_text,
                "rouge1_fmeasure": rouge1_f,
                "rouge2_fmeasure": rouge2_f,
                "rougeL_fmeasure": rougel_f,
                "reference_term": ref["term"],
                "reference_description": ref["description"],
                "reference_genes": ", ".join(ref["genes"]),
                "unique_genes": unique_genes_str
            })

        if not references:
            rows.append({
                "query": query,
                "context": context_str,
                "response": response_text,
                "rouge1_fmeasure": rouge1_f,
                "rouge2_fmeasure": rouge2_f,
                "rougeL_fmeasure": rougel_f,
                "reference_term": None,
                "reference_description": None,
                "reference_genes": None,
                "unique_genes": unique_genes_str
            })

        df = pd.DataFrame(rows)
        logging.info("Exiting get_summary_to_query_df() with DataFrame of shape=%s", df.shape)
        return df

    def save_output_to_json(self, output, filename):
        logging.info("Entering save_output_to_json() with filename='%s'", filename)
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
        logging.info("JSON output saved to '%s'.", filename)

    def to_markdown(self, output):
        logging.info("Entering to_markdown()...")
        md_lines = []
        md_lines.append("# Summary\n")
        md_lines.append("## Response\n")
        md_lines.append(output.get("response", "") + "\n")

        md_lines.append("## Rouge Scores\n")
        for metric, scores in output.get("rouge", {}).items():
            md_lines.append(f"**{metric}**:")
            md_lines.append(f"- Precision: {scores['precision']:.4f}")
            md_lines.append(f"- Recall: {scores['recall']:.4f}")
            md_lines.append(f"- F-measure: {scores['fmeasure']:.4f}\n")

        md_lines.append("## Context\n")
        md_lines.append(output.get("context", "") + "\n")

        md_lines.append("## References\n")
        for ref in output.get("references", []):
            md_lines.append(f"- **Term**: {ref['term']}")
            md_lines.append(f"  - Description: {ref['description']}")
            md_lines.append(f"  - Genes: {', '.join(ref['genes'])}\n")

        md_lines.append("## Unique Genes\n")
        for gene, count in output.get("unique_genes", {}).items():
            md_lines.append(f"- {gene}: {count}")

        logging.info("Exiting to_markdown().")
        return "\n".join(md_lines)

    def save_to_pickle(self, filename):
        logging.info("Entering save_to_pickle() with filename='%s'", filename)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        logging.info("RAGModule object saved to pickle file '%s'.", filename)

    @staticmethod
    def load_from_pickle(filename):
        logging.info("Entering load_from_pickle() with filename='%s'", filename)
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        logging.info("RAGModule object loaded from pickle file '%s'.", filename)
        return obj

def read_input_files(enrichment_csv: str, minor_topics_csv: str):
    df = pd.read_csv(enrichment_csv)
    topics = pd.read_csv(minor_topics_csv)
    return df, topics

def initialize_rag_module(df: pd.DataFrame) -> RAGModule:
    x_together = RAGModule(df)
    return x_together

def get_topics_of_interest(topics_df: pd.DataFrame) -> list:
    topics_df = topics_df[topics_df["prompt_type"]=="subtopic_BERT"]
    topics_of_interest = topics_df["generated_result"].tolist()
    return topics_of_interest

def generate_results(
    x_together: RAGModule,
    topics_of_interest: list,
    num_results: int = 5
) -> pd.DataFrame:
    results = []
    for topic in tqdm(topics_of_interest, desc="Processing topics"):
        results_df = x_together.get_summary_to_query_df(topic, num_results=num_results)
        results.append(results_df)
    final_df = pd.concat(results, ignore_index=True)
    return final_df

def save_results(final_df: pd.DataFrame, output_csv: str):
    final_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main(args):
    df, topics = read_input_files(args.enrichment_csv, args.minor_topics_csv)
    x_together = initialize_rag_module(df)
    topics_of_interest = get_topics_of_interest(topics)
    final_df = generate_results(
        x_together,
        topics_of_interest,
        num_results=args.num_results
    )
    save_results(final_df, args.output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summaries using RAGModule.")
    parser.add_argument(
        "--enrichment_csv",
        type=str,
        required=True,
        help="Path to the enrichment CSV file."
    )
    parser.add_argument(
        "--minor_topics_csv",
        type=str,
        required=True,
        help="Path to the minor topics CSV file."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output the final summary CSV file."
    )
    parser.add_argument(
        "--num_results",
        type=int,
        default=5,
        help="Number of results to retrieve for each query."
    )

    args = parser.parse_args()
    main(args)

# New: Define create_summary to generate a summary from API results and enrichment data.
def create_summary(api_results_df, enrichment_df, summary_output=None):
    """
    Create a summary DataFrame using API results and enrichment data.
    
    Args:
        api_results_df: pandas DataFrame containing API results, expected to have a 'query' column 
                        or a 'generated_result' column.
        enrichment_df: pandas DataFrame containing enrichment data for initializing RAGModule.
        summary_output: Optional path to output the summary CSV file.
    
    Returns:
        summary_df: pandas DataFrame containing summary information for each query.
    """
    # If 'query' column is missing, try renaming 'generated_result' to 'query'
    if 'query' not in api_results_df.columns:
        if 'generated_result' in api_results_df.columns:
            api_results_df = api_results_df.rename(columns={'generated_result': 'query'})
        else:
            raise ValueError("API results DataFrame must contain a 'query' column or a 'generated_result' column.")
    
    # Initialize RAGModule with the enrichment DataFrame
    rag_module = RAGModule(enrichment_df)
    
    summary_frames = []
    # Process each unique query in the API results
    for query in api_results_df["query"].unique():
        df_summary = rag_module.get_summary_to_query_df(query)
        summary_frames.append(df_summary)
    
    # Combine all individual query summaries into one DataFrame.
    summary_df = pd.concat(summary_frames, ignore_index=True)
    
    # Optionally save the summary to a CSV file if a path is provided.
    if summary_output:
        summary_df.to_csv(summary_output, index=False)
    
    return summary_df