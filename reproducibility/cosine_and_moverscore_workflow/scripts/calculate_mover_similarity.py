#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
import string
from collections import defaultdict, Counter
from itertools import chain
from math import log
from multiprocessing import Pool
import ot
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

###############################################################################
# Logging configuration
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

###############################################################################
# Global Initialization for Sentence Scoring (BERT + Moverscore)
###############################################################################
device = 'cuda'  # Use 'cpu' if CUDA is not available
model_name = os.environ.get('MOVERSCORE_MODEL', 'distilbert-base-uncased')
logger.info(f"Loading model {model_name} for sentence scoring")
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(
    model_name,
    output_hidden_states=True,
    output_attentions=True
)
model.eval()
model.to(device)

###############################################################################
# Helper Functions for Sentence Scoring
###############################################################################
def truncate(tokens):
    """Truncate tokens list to model maximum length minus 2 (for special tokens)."""
    if len(tokens) > tokenizer.model_max_length - 2:
        tokens = tokens[:tokenizer.model_max_length - 2]
    return tokens

def padding(arr, pad_token, dtype=torch.long):
    lengths = torch.LongTensor([len(seq) for seq in arr])
    max_len = lengths.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, seq in enumerate(arr):
        padded[i, :lengths[i]] = torch.tensor(seq, dtype=dtype)
        mask[i, :lengths[i]] = 1
    return padded, lengths, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        result = model(x, attention_mask=attention_mask)
    # For distilbert, hidden states are the second element
    if model_name == 'distilbert-base-uncased':
        return result[1]
    else:
        return result[2]

def collate_idf(arr, tokenize, numericalize, idf_dict, pad="[PAD]", device='cuda'):
    tokens = [["[CLS]"] + truncate(tokenize(text)) + ["[SEP]"] for text in arr]
    arr_ids = [numericalize(t) for t in tokens]
    idf_weights = [[idf_dict[i] for i in seq] for seq in arr_ids]
    pad_token = numericalize([pad])[0]
    padded, lengths, mask = padding(arr_ids, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lengths = lengths.to(device=device)
    return padded, padded_idf, lengths, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict, batch_size=-1, device='cuda'):
    padded_sens, padded_idf, lengths, mask, tokens = collate_idf(
        all_sens,
        tokenizer.tokenize,
        tokenizer.convert_tokens_to_ids,
        idf_dict,
        device=device
    )
    if batch_size == -1:
        batch_size = len(all_sens)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            embeddings.append(batch_embedding)
    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lengths, mask, padded_idf, tokens

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp,
                     stop_words=[], n_gram=1, remove_subwords=True,
                     batch_size=256, device='cuda'):
    preds = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start + batch_size]
        batch_hyps = hyps[batch_start:batch_start + batch_size]
        
        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(
            batch_refs, model, tokenizer, idf_dict_ref, device=device
        )
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(
            batch_hyps, model, tokenizer, idf_dict_hyp, device=device
        )
        
        # Use only the last layer embeddings.
        ref_embedding = ref_embedding[-1]
        hyp_embedding = hyp_embedding[-1]
        
        batch_size_actual = len(ref_tokens)
        for i in range(batch_size_actual):
            ref_ids = [k for k, w in enumerate(ref_tokens[i])
                       if w in stop_words or '##' in w or w in set(string.punctuation)]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i])
                       if w in stop_words or '##' in w or w in set(string.punctuation)]
            
            ref_embedding[i, ref_ids, :] = 0
            hyp_embedding[i, hyp_ids, :] = 0
            
            ref_idf[i, ref_ids] = 0
            hyp_idf[i, hyp_ids] = 0
            
        raw = torch.cat([ref_embedding, hyp_embedding], dim=1)
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30)
        
        distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()
                
        for i in range(batch_size_actual):
            c1 = np.zeros(raw.shape[1], dtype=float)
            c2 = np.zeros(raw.shape[1], dtype=float)
            c1[:len(ref_idf[i])] = ref_idf[i]
            c2[len(ref_idf[i]):] = hyp_idf[i]
            
            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            
            dst = distance_matrix[i]
            flow = ot.emd(c1, c2, dst)
            score = 1. / (1. + np.sum(flow * dst))
            preds.append(score)
    return preds

def sentence_score(hypothesis: str, references: list, trace=0):
    """
    Compute a sentence score given a hypothesis and its reference(s) using word mover's distance.
    """
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    hypothesis_list = [hypothesis] * len(references)
    scores = word_mover_score(
        refs=references,
        hyps=hypothesis_list,
        idf_dict_ref=idf_dict_ref,
        idf_dict_hyp=idf_dict_hyp,
        stop_words=[], n_gram=1, remove_subwords=False, device=device
    )
    sent_score = np.mean(scores)
    if trace > 0:
        print(hypothesis, references, sent_score)
    return sent_score

###############################################################################
# Functions analogous to SCRIPT ONE (but using sentence_score)
###############################################################################
def compute_sentence_score_matrix(texts1, texts2, batch_size_h=16):
    """
    Compute an M x N matrix where each element [i, j] is the sentence score
    computed between texts1[i] (as hypothesis) and texts2[j] (as reference),
    processing multiple hypotheses in a batch.
    
    Args:
        texts1 (list): List of hypothesis texts.
        texts2 (list): List of reference texts.
        batch_size_h (int): Number of hypotheses to process in one batch.
        
    Returns:
        np.ndarray: Score matrix of shape (len(texts1), len(texts2)).
    """
    M = len(texts1)
    N = len(texts2)
    logger.info(f"Computing sentence score matrix for {M} and {N} texts with hypothesis batch size {batch_size_h}...")
    matrix = np.zeros((M, N))
    
    for i in tqdm(range(0, M, batch_size_h), total=(M + batch_size_h - 1)//batch_size_h, desc="Batch Processing"):
        batch_texts1 = texts1[i:i+batch_size_h]
        # Construct the list of pairs for the batch.
        hyps = []
        refs = []
        for text in batch_texts1:
            hyps.extend([text] * N)
            refs.extend(texts2)
        scores = word_mover_score(
            refs=refs,
            hyps=hyps,
            idf_dict_ref=defaultdict(lambda: 1.),
            idf_dict_hyp=defaultdict(lambda: 1.),
            stop_words=[], n_gram=1, remove_subwords=False, device=device
        )
        scores = np.array(scores).reshape(len(batch_texts1), N)
        matrix[i:i+len(batch_texts1), :] = scores
    return matrix

def average_topk_sentence_by_columns_from_matrix(score_matrix, top_k=1):
    """
    Given a precomputed sentence score matrix, compute the average top-k scores
    for each column and then average across all columns.
    """
    topk_vals_for_each = []
    for j in range(score_matrix.shape[1]):
        col = score_matrix[:, j]
        top_k_indices = np.argpartition(col, -top_k)[-top_k:]
        topk_vals = col[top_k_indices]
        avg_topk = np.mean(topk_vals)
        topk_vals_for_each.append(avg_topk)
    return float(np.mean(topk_vals_for_each))

###############################################################################
# Main processing
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Compute and plot sentence similarity metrics using sentence score.")
    parser.add_argument("--sets_dir", type=str, default="1000geneset_benchmark/results/filtered_sets",
                        help="Directory containing gene set CSV files.")
    parser.add_argument("--topics_dir", type=str, default="1000geneset_benchmark/results/filtered_topics/samp_25",
                        help="Directory containing topic CSV files.")
    parser.add_argument("--scores_dir", type=str, default="scores_output_25",
                        help="Directory to save individual sentence score CSV files.")
    parser.add_argument("--results_csv", type=str, default="mover_score_results_25.csv",
                        help="CSV file to save aggregated results.")
    parser.add_argument("--plot_file", type=str, default="mover_score_scatter_25.png",
                        help="File name to save the scatter plot.")
    parser.add_argument("--num_gene_sets", type=int, default=10,
                        help="Number of gene sets to process from the intersection (default: 10)")
    args = parser.parse_args()

    sets_dir = args.sets_dir
    topics_dir = args.topics_dir
    scores_dir = args.scores_dir

    # File patterns
    sets_pattern   = os.path.join(sets_dir, "*_filtered_gene_sets.csv")
    topics_pattern = os.path.join(topics_dir, "*_filtered_topics.csv")

    # Collect filenames
    sets_files   = glob.glob(sets_pattern)
    topics_files = glob.glob(topics_pattern)

    # Build dictionaries mapping gene set names to file paths
    sets_dict = {}
    for f in sets_files:
        base = os.path.basename(f)
        gene_set_name = base.replace("_filtered_gene_sets.csv", "")
        sets_dict[gene_set_name] = f

    topics_dict = {}
    for f in topics_files:
        base = os.path.basename(f)
        gene_set_name = base.replace("_filtered_topics.csv", "")
        topics_dict[gene_set_name] = f

    # Collect results for aggregation
    results = []

    # Process only gene sets that exist in both directories
    all_gene_sets = sorted(set(sets_dict.keys()).intersection(topics_dict.keys()))
    if not all_gene_sets:
        logger.warning("No matching gene set files found. Exiting.")
        return

    # Prepare a directory to store individual metric CSVs
    os.makedirs(scores_dir, exist_ok=True)

    # Define topK values for evaluation
    topK_values = [2, 5, 10, 25, 50]

    for gene_set_name in all_gene_sets[:args.num_gene_sets]:
        logger.info(f"Processing gene set: {gene_set_name}")
        set_file = sets_dict[gene_set_name]
        topic_file = topics_dict[gene_set_name]
        df_sets = pd.read_csv(set_file)
        df_topics = pd.read_csv(topic_file)
        source_list = df_sets["Term"].tolist()  # full gene set (source)
        summary_list = df_topics["Term"].tolist()  # reduced set (topics)
        
        # Compute compression ratio
        compression_ratio = (
            len(source_list) / float(len(summary_list))
            if len(summary_list) > 0
            else 0.0
        )
        
        # Check if the score file already exists
        score_outfile = os.path.join(scores_dir, f"{gene_set_name}_sentence_scores.csv")
        if os.path.exists(score_outfile):
            logger.info(f"Score file exists for {gene_set_name}, loading it instead of recomputing.")
            # Load existing score matrix
            score_df = pd.read_csv(score_outfile, index_col=0)
            score_matrix = score_df.values
        else:
            # Compute sentence score matrix using batched processing
            logger.info(f"Calculating sentence scores for: {gene_set_name}")
            score_matrix = compute_sentence_score_matrix(source_list, summary_list, batch_size_h=16)
            score_df = pd.DataFrame(score_matrix, index=source_list, columns=summary_list)
            score_df.to_csv(score_outfile)
            logger.info(f"Saved sentence score matrix to: {score_outfile}")
        
        # Compute top-k recall using the matrix (either loaded or newly computed)
        gene_set_results = []
        for k in topK_values:
            recall_sentence = average_topk_sentence_by_columns_from_matrix(score_matrix, top_k=k)
            logger.info(f"[{gene_set_name}] topK={k} | sentence score recall={recall_sentence:.4f}")
            
            result_entry = {
                "gene_set": gene_set_name,
                "top_k": k,
                "compression_ratio": compression_ratio,
                "recall_sentence": recall_sentence,
                "source_length": len(source_list),
                "summary_length": len(summary_list)
            }
            results.append(result_entry)
            gene_set_results.append(result_entry)
            
        # Create and save a DataFrame with all top-k values for this gene set
        topk_df = pd.DataFrame(gene_set_results)
        topk_outfile = os.path.join(scores_dir, f"{gene_set_name}_topk.csv")
        topk_df.to_csv(topk_outfile, index=False)
        logger.info(f"Saved top-k results for {gene_set_name} to: {topk_outfile}")

    # Aggregate results into a DataFrame
    results_df = pd.DataFrame(results)
    logger.info("Aggregating results and preparing for plotting...")

    # Create bins for compression ratio
    bin_edges = [0, 50, 100, 150, 200, 350, 1e6]
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    results_df["ratio_bin"] = pd.cut(
        results_df["compression_ratio"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True
    )
    results_df.to_csv(args.results_csv, index=False)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=results_df,
        x="source_length",   # Source document size
        y="recall_sentence", # Moverscore (sentence score recall)
        style="top_k",
        s=100
    )

    plt.title("Top-k Moverscore vs. Source Document Size")
    plt.xlabel("Source Document Size (number of terms)")
    plt.ylabel("Top-k Moverscore")
    plt.legend(title="top_k")
    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {args.plot_file}")

if __name__ == "__main__":
    main()