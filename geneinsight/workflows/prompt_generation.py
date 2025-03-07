"""
Module for generating prompts from topic modeling results for API consumption.
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def create_transcript_for_topic(df: pd.DataFrame, topic: int) -> str:
    """
    Create a transcript for a topic by extracting relevant documents and keywords.
    
    Args:
        df: DataFrame containing topic modeling results
        topic: Topic ID to create transcript for
        
    Returns:
        A formatted transcript string with documents and keywords
    """
    # Ensure we filter for representative documents
    if "Representative_document" in df.columns:
        topic_df = df[(df["Topic"] == topic) & (df["Representative_document"] == True)]
    else:
        topic_df = df[df["Topic"] == topic]
    
    # Extract documents
    if "Document" in topic_df.columns:
        documents = topic_df["Document"].tolist()
    else:
        documents = []
    
    # Extract keywords
    if "Top_n_words" in topic_df.columns:
        keywords = topic_df["Top_n_words"].unique().tolist()
    else:
        keywords = []
    
    # Format as JSON strings to ensure proper escaping
    documents_str = json.dumps(documents)
    keywords_str = json.dumps(keywords)
    
    # Create the transcript
    transcript = f"Here are documents: {documents_str}. Here are keywords: {keywords_str}"
    return transcript

def generate_prompts(
    input_file: str,
    num_subtopics: int = 5,
    max_words: int = 10,
    output_file: Optional[str] = None,
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Generate prompts from topic modeling results.
    
    Args:
        input_file: Path to the CSV file with topic modeling results
        num_subtopics: Number of required subtopics per topic
        max_words: Maximum number of words per subtopic title
        output_file: Path to save the output CSV file
        max_retries: Maximum number of retries for generating subtopics
        
    Returns:
        DataFrame containing the generated prompts
    """
    logger.info(f"Reading topic modeling results from {input_file}")
    
    # Read the input file
    df_loaded_all_topics = pd.read_csv(input_file)
    seeds = df_loaded_all_topics["seed"].unique() if "seed" in df_loaded_all_topics.columns else [0]

    prompt_rows = []

    for seed in seeds:
        logger.info(f"Processing topic model with seed {seed}")
        
        # Filter by seed if present
        if "seed" in df_loaded_all_topics.columns:
            df_seed = df_loaded_all_topics[df_loaded_all_topics["seed"] == seed]
        else:
            df_seed = df_loaded_all_topics
            
        # Get unique topic labels
        topics_labels = df_seed["Topic"].unique() if "Topic" in df_seed.columns else [0]

        for topic_label in topics_labels:
            if topic_label == -1:
                logger.debug("Skipping noise topic -1")
                continue

            # Create transcript for the major topic
            major_transcript = create_transcript_for_topic(df_seed, topic_label)
            
            # Generate subtopics (using 0, 1, 2 as placeholders)
            subtopics = [0, 1, 2]
            for subtopic_label in subtopics:
                subtopic_transcript = ""  # Empty for now

                prompt_rows.append({
                    "prompt_type": "subtopic_BERT",  # Indicates what kind of processing to apply
                    "seed": seed,
                    "topic_label": topic_label,
                    "subtopic_label": subtopic_label,
                    "major_transcript": major_transcript,
                    "subtopic_transcript": subtopic_transcript,
                    "num_subtopics": num_subtopics,
                    "max_words": max_words,
                    "max_retries": max_retries,
                })
    
    # Create DataFrame from collected prompt rows
    df_prompts = pd.DataFrame(prompt_rows)
    
    # Save to CSV if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_prompts.to_csv(output_file, index=False)
        logger.info(f"Generated {len(df_prompts)} prompts, saved to {output_file}")
    
    return df_prompts