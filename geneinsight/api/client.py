# geneinsight/api/client.py

import os
import logging
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Check for a local .env file and load it if available
env_path = Path('.env')
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        logger.info("Loaded environment variables from .env file.")
    except ImportError:
        logger.warning("python-dotenv not installed; cannot load .env file.")
else:
    logger.info("No .env file found; relying on system environment variables.")

# Import instructor + OpenAI if available
try:
    import instructor
    from openai import OpenAI
    APIS_AVAILABLE = True
except ImportError:
    APIS_AVAILABLE = False


class TopicHeading(BaseModel):
    """Pydantic model to represent a topic heading response."""
    topic: str = Field(..., description="A title for the subtopic")


def fetch_subtopic_heading(
    user_prompt: str,
    system_prompt: str,
    *,
    service: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None
) -> str:
    """
    Get a subtopic heading from the API.

    Args:
        user_prompt: The user portion of the prompt.
        system_prompt: The system portion of the prompt.
        service: API service to use ("openai", "together", "ollama").
        api_key: API key.
        model: Model to use.
        base_url: Base URL for the API service.

    Returns:
        The generated topic heading or an error string if something fails.
    """
    if not APIS_AVAILABLE:
        logger.warning("API modules (openai, instructor) not installed. Returning placeholder.")
        return "API_NOT_AVAILABLE"

    # Raise ValueError if the service is unsupported
    if service.lower() not in {"openai", "together", "ollama"}:
        raise ValueError(f"Unsupported service: {service}")

    try:
        # Initialize the appropriate client
        if service.lower() == "together":
            together_client = OpenAI(api_key=api_key, base_url=base_url)
            client = instructor.from_openai(together_client, mode=instructor.Mode.TOOLS)
        elif service.lower() == "openai":
            openai_client = OpenAI(api_key=api_key)
            client = instructor.from_openai(openai_client, mode=instructor.Mode.TOOLS)
        elif service.lower() == "ollama":
            ollama_client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
            client = instructor.from_openai(ollama_client, mode=instructor.Mode.TOOLS)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response: TopicHeading = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=TopicHeading
        )

        return response.topic if hasattr(response, 'topic') else str(response)

    except Exception as e:
        logger.error(f"{service.capitalize()} API call failed: {e}")
        return f"ERROR: {str(e)}"


def process_subtopic_row(
    row: pd.Series,
    *,
    service: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single row for subtopic generation.

    Args:
        row: Row from the prompts DataFrame.
        service: API service to use.
        api_key: API key.
        model: Model to use.
        base_url: Base URL for the API service.

    Returns:
        Dictionary with the row data and generated result.
    """
    seed = row["seed"]
    topic_label = row["topic_label"]
    major_transcript = row["major_transcript"]
    max_words = int(row["max_words"]) if "max_words" in row else 10

    long_prompt = (
        f"Using the provided information, including documents, keywords, and the major heading, "
        f"generate a single subtopic heading of at most {max_words} words. "
        f"The subtopic should provide more detail than the major heading while remaining closely related to it. "
        f"It should emulate the tone and structure of a gene ontology term, incorporating specific details from the information provided. "
        f"Respond only with the subtopic heading."
    )

    user_prompt = long_prompt + f"\n\nPROVIDED INFORMATION:\n{major_transcript}"

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
        "generated_result": generated_subtopic
    }


def batch_process_api_calls(
    prompts_csv: str,
    output_api: str,
    service: str = "openai",
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Batch process API calls for subtopic generation.

    Args:
        prompts_csv: Path to CSV with the input prompts.
        output_api: Path to output CSV for results.
        service: Which service to use.
        model: Which model to use.
        base_url: Base URL for the API service.
        n_jobs: Number of parallel workers.

    Returns:
        DataFrame with the results.
    """
    df_prompts = pd.read_csv(prompts_csv)
    logger.info(f"Loaded {len(df_prompts)} rows from {prompts_csv}")

    df_subtopic = df_prompts[df_prompts["prompt_type"] == "subtopic_BERT"]
    logger.info(f"Found {len(df_subtopic)} rows with 'subtopic_BERT'")

    if df_subtopic.empty:
        logger.warning("No subtopic_BERT rows found. Exiting.")
        pd.DataFrame().to_csv(output_api, index=False)
        return pd.DataFrame()

    # For openai/together, we read the environment variable
    # For ollama, we always set "ollama" if not provided
    api_key = None
    if service.lower() == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
    elif service.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif service.lower() == "ollama":
        api_key = "ollama"

    if not api_key and service.lower() != "ollama":
        raise ValueError(f"API key for {service} not found in .env file or environment variables.")

    logger.info(f"Processing {len(df_subtopic)} rows with {n_jobs} parallel jobs")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subtopic_row)(
            row,
            service=service,
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        for _, row in tqdm(df_subtopic.iterrows(), total=len(df_subtopic), desc="Processing subtopics")
    )

    df_results = pd.DataFrame(results)

    # Ensure the output path directory exists
    os.makedirs(os.path.dirname(output_api), exist_ok=True)
    df_results.to_csv(output_api, index=False)
    logger.info(f"Results saved to {output_api}")

    return df_results
