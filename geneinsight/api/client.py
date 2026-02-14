# geneinsight/api/client.py

import os
import time
import logging
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Optional, Dict, Any, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field
logging.getLogger("httpx").disabled = True

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import instrumentation utilities
try:
    from geneinsight.instrumentation import TokenUsage, TokenCounter
    INSTRUMENTATION_AVAILABLE = True
except ImportError:
    INSTRUMENTATION_AVAILABLE = False
    TokenUsage = None
    TokenCounter = None


@dataclass
class APICallResult:
    """Result of a single API call including metrics."""
    result: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

# Check for a local .env file and load it if available
env_path = Path('.env')
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        #logger.info("Loaded environment variables from .env file.")
    except ImportError:
        logger.warning("python-dotenv not installed; cannot load .env file.")
else:
    logger.warning("No .env file found; relying on system environment variables.")

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
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    return_metrics: bool = False
) -> str | APICallResult:
    """
    Get a subtopic heading from the API.

    Args:
        user_prompt: The user portion of the prompt.
        system_prompt: The system portion of the prompt.
        service: API service to use ("openai", "together", "ollama").
        api_key: API key.
        model: Model to use.
        base_url: Base URL for the API service.
        temperature: Sampling temperature for the model.
        return_metrics: If True, return APICallResult with token/latency metrics.

    Returns:
        The generated topic heading (str) or APICallResult if return_metrics=True.
    """
    if not APIS_AVAILABLE:
        logger.warning("API modules (openai, instructor) not installed. Returning placeholder.")
        if return_metrics:
            return APICallResult(result="API_NOT_AVAILABLE", success=False, error="API not available")
        return "API_NOT_AVAILABLE"

    # Raise ValueError if the service is unsupported
    if service.lower() not in {"openai", "together", "ollama"}:
        raise ValueError(f"Unsupported service: {service}")

    start_time = time.time()
    prompt_tokens = 0
    completion_tokens = 0

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

        # Check if service is openai and model contains "o3" - if so, don't use temperature
        # Use create_with_completion to get both the model and raw response with usage info
        if service.lower() == "openai" and "o3" in model:
            response, completion = client.chat.completions.create_with_completion(
                model=model,
                messages=messages,
                response_model=TopicHeading
            )
        else:
            response, completion = client.chat.completions.create_with_completion(
                model=model,
                messages=messages,
                response_model=TopicHeading,
                temperature=temperature
            )

        # Extract token usage from the completion object
        if hasattr(completion, 'usage') and completion.usage:
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
            completion_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

        result_text = response.topic if hasattr(response, 'topic') else str(response)
        latency_ms = (time.time() - start_time) * 1000

        if return_metrics:
            return APICallResult(
                result=result_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=True
            )
        return result_text

    except Exception as e:
        logger.error(f"{service.capitalize()} API call failed: {e}")
        latency_ms = (time.time() - start_time) * 1000
        error_msg = f"ERROR: {str(e)}"

        if return_metrics:
            return APICallResult(
                result=error_msg,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
        return error_msg


def process_subtopic_row(
    row: pd.Series,
    *,
    service: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    token_counter: Optional['TokenCounter'] = None
) -> Dict[str, Any]:
    """
    Process a single row for subtopic generation.

    Args:
        row: Row from the prompts DataFrame.
        service: API service to use.
        api_key: API key.
        model: Model to use.
        base_url: Base URL for the API service.
        temperature: Sampling temperature for the model.
        token_counter: Optional TokenCounter for aggregating token usage.

    Returns:
        Dictionary with the row data, generated result, and metrics.
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

    # Always get metrics now
    api_result = fetch_subtopic_heading(
        user_prompt=user_prompt,
        system_prompt="You are a helpful AI assistant",
        service=service,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        return_metrics=True
    )

    # Record in token counter if provided (thread-safe)
    if token_counter is not None and INSTRUMENTATION_AVAILABLE:
        token_counter.add(
            prompt_tokens=api_result.prompt_tokens,
            completion_tokens=api_result.completion_tokens,
            latency_ms=api_result.latency_ms
        )

    return {
        "prompt_type": "subtopic_BERT",
        "seed": seed,
        "topic_label": topic_label,
        "generated_result": api_result.result,
        # Include metrics in result for potential per-call analysis
        "_prompt_tokens": api_result.prompt_tokens,
        "_completion_tokens": api_result.completion_tokens,
        "_latency_ms": api_result.latency_ms
    }


@dataclass
class BatchAPIMetrics:
    """Aggregated metrics from batch API processing."""
    total_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latencies_ms: list = None
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def __post_init__(self):
        if self.latencies_ms is None:
            self.latencies_ms = []


def batch_process_api_calls(
    prompts_csv: str,
    output_api: str,
    service: str = "openai",
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    n_jobs: int = 4,
    temperature: float = 0.2
) -> Tuple[pd.DataFrame, Optional[BatchAPIMetrics]]:
    """
    Batch process API calls for subtopic generation.

    Args:
        prompts_csv: Path to CSV with the input prompts.
        output_api: Path to output CSV for results.
        service: Which service to use.
        model: Which model to use.
        base_url: Base URL for the API service.
        n_jobs: Number of parallel workers.
        temperature: Sampling temperature for the model.

    Returns:
        Tuple of (DataFrame with results, BatchAPIMetrics with token/latency data).
    """
    df_prompts = pd.read_csv(prompts_csv)
    logger.info(f"Loaded {len(df_prompts)} rows from {prompts_csv}")

    df_subtopic = df_prompts[df_prompts["prompt_type"] == "subtopic_BERT"]
    logger.info(f"Found {len(df_subtopic)} rows with 'subtopic_BERT'")

    if df_subtopic.empty:
        logger.warning("No subtopic_BERT rows found. Exiting.")
        pd.DataFrame().to_csv(output_api, index=False)
        return pd.DataFrame(), None

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

    # Create token counter for thread-safe aggregation
    token_counter = None
    if INSTRUMENTATION_AVAILABLE:
        token_counter = TokenCounter()

    logger.info(f"Processing {len(df_subtopic)} rows with {n_jobs} parallel jobs")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subtopic_row)(
            row,
            service=service,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            token_counter=token_counter
        )
        for _, row in tqdm(df_subtopic.iterrows(), total=len(df_subtopic), desc="Processing subtopics")
    )

    df_results = pd.DataFrame(results)

    # Build metrics from token counter
    batch_metrics = None
    if token_counter is not None:
        token_usage = token_counter.get_total()
        latencies = token_counter.get_latencies()

        # Calculate latency statistics
        mean_latency = sum(latencies) / len(latencies) if latencies else 0
        sorted_latencies = sorted(latencies) if latencies else []
        p95_idx = int(len(sorted_latencies) * 0.95) if sorted_latencies else 0

        batch_metrics = BatchAPIMetrics(
            total_calls=token_counter.get_call_count(),
            prompt_tokens=token_usage.prompt_tokens,
            completion_tokens=token_usage.completion_tokens,
            total_tokens=token_usage.total_tokens,
            latencies_ms=latencies,
            mean_latency_ms=mean_latency,
            p95_latency_ms=sorted_latencies[p95_idx] if sorted_latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0
        )

        logger.info(
            f"API metrics: {batch_metrics.total_calls} calls, "
            f"{batch_metrics.total_tokens:,} tokens "
            f"(prompt: {batch_metrics.prompt_tokens:,}, completion: {batch_metrics.completion_tokens:,}), "
            f"mean latency: {batch_metrics.mean_latency_ms:.1f}ms"
        )

    # Remove internal metric columns before saving (they start with _)
    output_columns = [col for col in df_results.columns if not col.startswith('_')]
    df_output = df_results[output_columns]

    # Ensure the output path directory exists
    os.makedirs(os.path.dirname(output_api), exist_ok=True)
    df_output.to_csv(output_api, index=False)
    logger.info(f"Results saved to {output_api}")

    return df_results, batch_metrics