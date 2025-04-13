import argparse
import os
import logging
import dotenv
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, List

# Import instructor + OpenAI
import instructor
from openai import OpenAI

# Pydantic typed model for validation
from pydantic import BaseModel, Field

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =====================
# 1) Pydantic Class
# =====================
class TopicHeading(BaseModel):
    """
    If you expect a single subtopic heading in the response, 
    you can store it in 'topic'.
    """
    topic: str = Field(..., description="A title for the subtopic")


# =====================
# 2) Generic typed API caller
# =====================
def fetch_subtopic_heading(
    user_prompt: str,
    system_prompt: str,
    *,
    service: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None
) -> str:
    """
    Generic function that uses instructor-based typed validation 
    to get a subtopic heading (TopicHeading.topic) from either:
        - Together AI (`service="together"`)
        - OpenAI (`service="openai"`)

    Args:
        user_prompt (str): The user portion of the prompt.
        system_prompt (str): The system portion of the prompt.
        service (str): "together" or "openai".
        api_key (str): The API key for the given service.
        model (str): The model name for the given service.
        base_url (Optional[str]): If needed (e.g., for Together).
    
    Returns:
        str: The validated 'topic' string or an empty string on error.
    """
    try:
        # Initialize the appropriate client
        if service.lower() == "together":
            # For Together
            together_client = OpenAI(api_key=api_key, base_url=base_url)
            client = instructor.from_openai(together_client, mode=instructor.Mode.TOOLS)
        elif service.lower() == "openai":
            # For OpenAI
            openai_client = OpenAI(api_key=api_key)
            client = instructor.from_openai(openai_client, mode=instructor.Mode.TOOLS)
        else:
            raise ValueError(f"Unsupported service: {service}")

        # Build your message array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Create the completion with typed validation
        response: TopicHeading = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=TopicHeading
        )
        return response.topic

    except Exception as e:
        logger.error(f"{service.capitalize()} API call failed: {e}")
        return ""


# =====================
# 3) Row Processing
# =====================
def process_subtopic_row(
    row: pd.Series,
    *,
    service: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> dict:
    """
    For a row with prompt_type='subtopic_BERT', build the system+user prompts
    and fetch the validated subtopic heading.
    """
    # Extract row data
    seed = row["seed"]
    #strategy = row["strategy"]
    topic_label = row["topic_label"]

    #subtopic_transcript = row["subtopic_transcript"]
    major_transcript = row["major_transcript"]
    max_words = int(row["max_words"])
    # if you need max_retries or other fields, extract them here

    # Construct the system prompt 
    long_prompt = (
    f"Using the provided information, including documents, keywords, and the major heading, "
    f"generate a single subtopic heading of at most {max_words} words. "
    f"The subtopic should provide more detail than the major heading while remaining closely related to it. "
    f"It should emulate the tone and structure of a gene ontology term, incorporating specific details from the information provided. "
    f"Respond only with the subtopic heading."
    )
    # Construct the user prompt
    user_prompt = long_prompt + f"PROVIDED INFORMATION:\n {major_transcript}"

    # Call the typed function
    generated_subtopic = fetch_subtopic_heading(
        user_prompt=user_prompt,
        system_prompt="You are a helpful AI assistant",
        service=service,
        api_key=api_key,
        model=model,
        base_url=base_url
    )

    return {
        "prompt_type": "subtopic_BERT",
        "seed": seed,
        "topic_label": topic_label,
        #"strategy": strategy,
        "generated_result": generated_subtopic
    }


# =====================
# 4) Parallel loop
# =====================
def parallel_subtopic_generation(
    df_subtopic: pd.DataFrame,
    service: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    n_jobs: int = 4
) -> List[dict]:
    """
    Parallel processing for subtopic_BERT rows using joblib.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subtopic_row)(
            row,
            service=service,
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        for _, row in tqdm(df_subtopic.iterrows(), total=len(df_subtopic), desc="subtopic_BERT")
    )
    return results


# =====================
# 5) Main script entry
# =====================
def main():
    parser = argparse.ArgumentParser(description="Typed, parallel subtopic_BERT generator.")
    parser.add_argument("--prompts_csv", type=str, required=True,
                        help="Path to CSV with the input prompts.")
    parser.add_argument("--output_api", type=str, required=True,
                        help="Output CSV for subtopic_BERT results.")
    parser.add_argument("--service", type=str, default="openai",
                        help="Which service: 'together' or 'openai'.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Which model to use (depends on service).")
    parser.add_argument("--base_url", type=str, default=None,
                        help="If using Together, specify the base_url. (Optional)")
    parser.add_argument("--n_jobs", type=int, default=2,
                        help="Number of parallel workers.")
    args = parser.parse_args()

    # Load CSV
    df_prompts = pd.read_csv(args.prompts_csv)
    logger.info(f"Loaded {len(df_prompts)} rows from {args.prompts_csv}")

    # Filter for subtopic_BERT
    df_subtopic = df_prompts[df_prompts["prompt_type"] == "subtopic_BERT"]
    logger.info(f"Found {len(df_subtopic)} rows with 'subtopic_BERT'")

    if df_subtopic.empty:
        logger.warning("No subtopic_BERT rows found. Exiting.")
        pd.DataFrame().to_csv(args.output_api, index=False)
        return

    # Get the API key from environment variables or wherever you store it
    api_key = os.getenv("TOGETHER_API_KEY") if args.service.lower() == "together" else os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("API key not found in environment. Please set it.")
        return

    # Parallel process
    results = parallel_subtopic_generation(
        df_subtopic=df_subtopic,
        service=args.service,
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        n_jobs=args.n_jobs
    )

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_api, index=False)
    logger.info(f"Wrote subtopic_BERT results to {args.output_api}")


if __name__ == "__main__":
    main()