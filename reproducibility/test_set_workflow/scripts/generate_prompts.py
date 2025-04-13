import argparse
import pandas as pd
import logging
import os
import json

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def create_transcript_for_topic(df: pd.DataFrame, topic: int):
    documents = df[(df["Topic"] == topic) & (df["Representative_document"] == True)]["Document"].tolist()
    keywords = df[(df["Topic"] == topic) & (df["Representative_document"] == True)]["Top_n_words"].unique().tolist()
    documents_str = json.dumps(documents)
    keywords_str = json.dumps(keywords)
    transcript = f"Here are documents: {documents_str}. Here are keywords: {keywords_str}"
    return transcript

def main():
    parser = argparse.ArgumentParser(description="Generate prompts (no API calls).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with topics.")
    parser.add_argument("--num_subtopics", type=int, default=5, help="Number of required subtopics.")
    parser.add_argument("--max_words", type=int, default=15, help="Maximum number of words per subtopic title.")
    parser.add_argument("--output_prompts", type=str, required=True, help="Path to the output CSV file for prompts.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max number of retries for generating subtopics.")
    args = parser.parse_args()

    input_file = args.input
    number_of_required_subtopics = args.num_subtopics
    max_words = args.max_words
    output_prompts_file = args.output_prompts
    max_retries = args.max_retries

    logger.info(f"Reading input CSV from: {input_file}")
    df_loaded_all_topics = pd.read_csv(input_file)
    seeds = df_loaded_all_topics["seed"].unique()

    prompt_rows = []

    for seed in seeds:
        logger.info(f"Processing topic model with seed {seed}")
        df_seed = df_loaded_all_topics[df_loaded_all_topics["seed"] == seed]
        topics_labels = df_seed["Topic"].unique()

        for topic_label in topics_labels:
            if topic_label == -1:
                logger.debug("Skipping noise topic -1")
                continue

            major_transcript = create_transcript_for_topic(df_seed, topic_label)
            
            subtopics = [0,1,2]
            for subtopic_label in subtopics:
                subtopic_transcript = ""

                prompt_rows.append({
                    "prompt_type": "subtopic_BERT",  # Tells script2 to call generate_subtopic_titles_BERT
                    "seed": seed,
                    "topic_label": topic_label,
                    "subtopic_label": subtopic_label,
                    "major_transcript": major_transcript,
                    "subtopic_transcript": subtopic_transcript,
                    "num_subtopics": number_of_required_subtopics,
                    "max_words": max_words,
                    "max_retries": max_retries,
                })
    
    df_prompts = pd.DataFrame(prompt_rows)
    logger.info(f"Saving prompts to {output_prompts_file}")
    df_prompts.to_csv(output_prompts_file, index=False)
    logger.info("Prompts saved successfully.")

if __name__ == "__main__":
    main()
