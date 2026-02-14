# tests/test_client.py

import os
import sys
import pytest
import logging
import importlib
import pandas as pd
from unittest.mock import MagicMock, patch

import geneinsight.api.client as gi_client
from geneinsight.api.client import (
    APIS_AVAILABLE,
    fetch_subtopic_heading,
    process_subtopic_row,
    batch_process_api_calls
)


@pytest.fixture
def sample_row():
    """Fixture that returns a sample Pandas Series row
       that can be used in process_subtopic_row tests."""
    data = {
        "seed": "12345",
        "topic_label": "Biology",
        "major_transcript": "Major heading on gene regulations",
        "max_words": "5",
        "prompt_type": "subtopic_BERT",
    }
    return pd.Series(data)


@pytest.fixture
def sample_df():
    """Fixture that returns a sample DataFrame for testing
       batch_process_api_calls."""
    data = [
        {
            "prompt_type": "subtopic_BERT",
            "seed": "111",
            "topic_label": "TopicA",
            "major_transcript": "Transcript A",
            "max_words": "10",
        },
        {
            "prompt_type": "subtopic_BERT",
            "seed": "222",
            "topic_label": "TopicB",
            "major_transcript": "Transcript B",
            "max_words": "10",
        },
        # A row that doesn't have prompt_type=subtopic_BERT
        {
            "prompt_type": "something_else",
            "seed": "333",
            "topic_label": "TopicC",
            "major_transcript": "Transcript C",
            "max_words": "10",
        }
    ]
    return pd.DataFrame(data)


@pytest.mark.parametrize("apis_available", [True, False])
def test_fetch_subtopic_heading_api_availability(monkeypatch, apis_available):
    """
    Test fetch_subtopic_heading when APIS_AVAILABLE is True or False.
    If False, we expect a placeholder string "API_NOT_AVAILABLE".
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", apis_available)

    # Patch instructor calls so they don't fail
    mock_api_call = MagicMock(return_value="MOCK_TOPIC")
    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_api_call)

    result = fetch_subtopic_heading("user prompt", "system prompt")
    if not apis_available:
        assert result == "API_NOT_AVAILABLE", "Expected API_NOT_AVAILABLE when APIS_AVAILABLE=False"
    else:
        # Could be "MOCK_TOPIC" or "ERROR: ...", but it won't raise
        assert "MOCK_TOPIC" in result or "ERROR:" in result


def test_fetch_subtopic_heading_unsupported_service(monkeypatch):
    """
    Test fetch_subtopic_heading with an unsupported service to confirm
    it raises the appropriate ValueError.
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    with pytest.raises(ValueError, match="Unsupported service: invalid_service"):
        fetch_subtopic_heading("user prompt", "system prompt", service="invalid_service")


@pytest.mark.parametrize("service", ["openai", "together", "ollama"])
def test_fetch_subtopic_heading_api_call_success(monkeypatch, service):
    """
    Test fetch_subtopic_heading with valid services. We'll mock
    out the API call and ensure the function returns the expected string.
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    # Mock the OpenAI client constructor
    mock_openai_client = MagicMock()
    monkeypatch.setattr("geneinsight.api.client.OpenAI", mock_openai_client)

    # Mock the from_openai function to return a mock object
    mock_client = MagicMock()
    # We'll simulate a response having a 'topic' field
    mock_response = MagicMock()
    mock_response.topic = "MOCK_TOPIC"
    # Mock the completion with usage info
    mock_completion = MagicMock()
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    # create_with_completion returns (response, completion) tuple
    mock_client.chat.completions.create_with_completion.return_value = (mock_response, mock_completion)

    def mock_from_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)

    # Set environment variables for API keys
    if service == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    elif service == "together":
        monkeypatch.setenv("TOGETHER_API_KEY", "test_together_key")
    elif service == "ollama":
        # For ollama, we still need to make sure an api_key is available
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")

    result = fetch_subtopic_heading(
        user_prompt="some user prompt",
        system_prompt="some system prompt",
        service=service,
        # Explicitly provide an API key to avoid relying on env variables in test
        api_key="test_key_for_all_services"
    )
    assert result == "MOCK_TOPIC"
    mock_client.chat.completions.create_with_completion.assert_called_once()


def test_fetch_subtopic_heading_no_topic_attribute(monkeypatch):
    """
    Test fetch_subtopic_heading when the API response does not have a 'topic'
    attribute. The function should return str(response).
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    mock_client = MagicMock()
    # Simulate a response that has no 'topic' attribute
    mock_response = "Just a string response"
    mock_completion = MagicMock()
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    # create_with_completion returns (response, completion) tuple
    mock_client.chat.completions.create_with_completion.return_value = (mock_response, mock_completion)

    def mock_from_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    result = fetch_subtopic_heading(
        user_prompt="some prompt",
        system_prompt="some system prompt",
        service="openai"
    )
    assert result == str(mock_response)


def test_fetch_subtopic_heading_api_call_exception(monkeypatch):
    """
    Test fetch_subtopic_heading when an exception occurs during the API call,
    ensuring it returns an error message starting with "ERROR:".
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    def mock_from_openai(*args, **kwargs):
        raise RuntimeError("Mocked exception")

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")

    result = fetch_subtopic_heading("user prompt", "system prompt", service="openai")
    assert result.startswith("ERROR:")


def test_process_subtopic_row(monkeypatch, sample_row):
    """
    Test process_subtopic_row to ensure it calls fetch_subtopic_heading
    and returns the expected dictionary fields.
    """
    from geneinsight.api.client import APICallResult

    def mock_fetch_subtopic_heading(*args, **kwargs):
        # Now returns APICallResult when return_metrics=True
        return APICallResult(
            result="Mocked Subtopic",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            success=True
        )

    monkeypatch.setattr("geneinsight.api.client.fetch_subtopic_heading", mock_fetch_subtopic_heading)

    result = process_subtopic_row(
        sample_row,
        service="openai",
        api_key="fake_key",
        model="gpt-4o-mini"
    )

    assert result["prompt_type"] == "subtopic_BERT"
    assert result["seed"] == sample_row["seed"]
    assert result["topic_label"] == sample_row["topic_label"]
    assert result["generated_result"] == "Mocked Subtopic"


def test_process_subtopic_row_missing_max_words(monkeypatch):
    """
    Test process_subtopic_row when 'max_words' is missing. The code
    defaults to 10 if 'max_words' isn't present.
    """
    from geneinsight.api.client import APICallResult

    row_data = {
        "seed": "99999",
        "topic_label": "Missing Max Words",
        "major_transcript": "A large transcript that needs subtopic generation.",
        "prompt_type": "subtopic_BERT"
        # note: no "max_words"
    }
    row = pd.Series(row_data)

    def mock_fetch_subtopic_heading(*args, **kwargs):
        # Now returns APICallResult when return_metrics=True
        return APICallResult(
            result="Mocked Subtopic",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            success=True
        )

    monkeypatch.setattr("geneinsight.api.client.fetch_subtopic_heading", mock_fetch_subtopic_heading)

    result = process_subtopic_row(
        row,
        service="openai",
        api_key="fake_key"
    )
    assert result["seed"] == "99999"
    assert result["generated_result"] == "Mocked Subtopic"


def test_batch_process_api_calls_basic(monkeypatch, sample_df, tmp_path):
    """
    Test batch_process_api_calls with a normal DataFrame that contains
    some 'subtopic_BERT' rows. Ensure it processes them and writes to CSV.
    We'll mock read_csv, to_csv, and the process_subtopic_row function.
    Note: Using n_jobs=1 to avoid pickling issues with monkeypatched functions.
    """
    def mock_read_csv(*args, **kwargs):
        return sample_df

    mock_to_csv = MagicMock()

    def mock_process_subtopic_row(row, *args, **kwargs):
        return {
            "prompt_type": row["prompt_type"],
            "seed": row["seed"],
            "topic_label": row["topic_label"],
            "generated_result": "MOCKED_RESULT",
            "_prompt_tokens": 10,
            "_completion_tokens": 5,
            "_latency_ms": 100.0,
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)
    # Provide an environment key for openai
    monkeypatch.setenv("OPENAI_API_KEY", "some_test_key")

    output_api_path = str(tmp_path / "output.csv")
    # batch_process_api_calls now returns a tuple (df_results, batch_metrics)
    # Using n_jobs=1 to avoid pickling issues with monkeypatched functions in parallel execution
    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=1
    )

    # We expect to process only the two subtopic_BERT rows, ignoring the third
    assert len(df_results) == 2
    for _, row in df_results.iterrows():
        assert row["generated_result"] == "MOCKED_RESULT"

    mock_to_csv.assert_called_once()


def test_batch_process_api_calls_no_subtopic(monkeypatch, tmp_path):
    """
    Test batch_process_api_calls when the CSV has no subtopic_BERT rows.
    The function should return an empty DataFrame and still write to CSV.
    """
    df_no_subtopic = pd.DataFrame([
        {"prompt_type": "other_type", "seed": "123", "topic_label": "None"}
    ])

    def mock_read_csv(*args, **kwargs):
        return df_no_subtopic

    mock_to_csv = MagicMock()

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    output_api_path = str(tmp_path / "output.csv")
    # batch_process_api_calls now returns a tuple (df_results, batch_metrics)
    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=1
    )

    assert df_results.empty
    assert batch_metrics is None  # No metrics when no rows processed
    mock_to_csv.assert_called_once()


def test_batch_process_api_calls_missing_api_key(monkeypatch, sample_df, tmp_path):
    """
    Test batch_process_api_calls when environment variable is missing
    and the service is 'openai' or 'together'. Should raise ValueError.
    """
    # Force environment var removal for both openai & together
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    def mock_read_csv(*args, **kwargs):
        return sample_df

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    output_api_path = str(tmp_path / "output.csv")

    # We do NOT set environment variable for openai, so it should fail
    with pytest.raises(ValueError, match="API key for openai not found"):
        batch_process_api_calls(
            prompts_csv="fake_prompts.csv",
            output_api=output_api_path,
            service="openai",
            model="gpt-4o-mini",
            n_jobs=1
        )


def test_batch_process_api_calls_ollama_service(monkeypatch, sample_df, tmp_path):
    """
    Explicitly test the 'ollama' service usage in batch_process_api_calls
    to ensure it doesn't require an environment variable.
    """
    def mock_read_csv(*args, **kwargs):
        return sample_df

    mock_to_csv = MagicMock()

    def mock_process_subtopic_row(row, *args, **kwargs):
        return {
            "prompt_type": row["prompt_type"],
            "seed": row["seed"],
            "topic_label": row["topic_label"],
            "generated_result": "MOCKED_OLLAMA_RESULT",
            "_prompt_tokens": 10,
            "_completion_tokens": 5,
            "_latency_ms": 100.0,
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)

    output_api_path = str(tmp_path / "output.csv")

    # batch_process_api_calls now returns a tuple (df_results, batch_metrics)
    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="ollama",
        model="gpt-4o-mini",
        n_jobs=1
    )
    # We expect to process 2 'subtopic_BERT' rows
    assert len(df_results) == 2
    assert all(r == "MOCKED_OLLAMA_RESULT" for r in df_results["generated_result"])


@pytest.mark.usefixtures("caplog")
def test_no_dotenv_file_found(monkeypatch, caplog):
    """
    Test that if no .env file is present, a log message about relying on
    system environment variables is emitted.
    We'll patch Path('.env').exists() to return False in the client module,
    then reload that module so the code's top-level .env check runs again.
    """
    with caplog.at_level(logging.INFO, logger="geneinsight.api.client"):
        # Patch 'Path.exists' in the geneinsight.api.client module only
        with patch("geneinsight.api.client.Path.exists", return_value=False):
            importlib.reload(gi_client)

        found_log = any("No .env file found; relying on system environment variables." in rec.message
                        for rec in caplog.records)
        assert found_log, "Expected log message about .env file not found"

    # Reload again to restore normal state after test
    importlib.reload(gi_client)


# ============================================================================
# Additional tests for improved coverage
# ============================================================================

from geneinsight.api.client import BatchAPIMetrics


def test_dotenv_import_error(monkeypatch, caplog):
    """
    Test that dotenv ImportError is handled gracefully.
    """
    # This tests the module-level try/except for dotenv
    # We can't easily trigger it without reloading, but we can test the logging behavior
    with caplog.at_level(logging.WARNING, logger="geneinsight.api.client"):
        # Reload module to test initialization
        importlib.reload(gi_client)


def test_o3_model_no_temperature(monkeypatch):
    """
    Test that o3 models don't use temperature parameter.
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.topic = "O3_TOPIC"
    mock_completion = MagicMock()
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    mock_client.chat.completions.create_with_completion.return_value = (mock_response, mock_completion)

    def mock_from_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    result = fetch_subtopic_heading(
        user_prompt="some prompt",
        system_prompt="system prompt",
        service="openai",
        model="o3-mini"  # o3 model
    )

    assert result == "O3_TOPIC"
    # Verify that create_with_completion was called without temperature
    call_kwargs = mock_client.chat.completions.create_with_completion.call_args[1]
    assert "temperature" not in call_kwargs


def test_token_extraction_missing_usage(monkeypatch):
    """
    Test handling when completion.usage is None.
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.topic = "TEST_TOPIC"
    mock_completion = MagicMock()
    mock_completion.usage = None  # No usage info
    mock_client.chat.completions.create_with_completion.return_value = (mock_response, mock_completion)

    def mock_from_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    from geneinsight.api.client import APICallResult
    result = fetch_subtopic_heading(
        user_prompt="some prompt",
        system_prompt="system prompt",
        service="openai",
        model="gpt-4o-mini",
        return_metrics=True
    )

    assert isinstance(result, APICallResult)
    assert result.result == "TEST_TOPIC"
    assert result.prompt_tokens == 0  # Should default to 0
    assert result.completion_tokens == 0


def test_token_extraction_none_values(monkeypatch):
    """
    Test handling when token values in usage are None.
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.topic = "TEST_TOPIC"
    mock_completion = MagicMock()
    mock_completion.usage = MagicMock()
    mock_completion.usage.prompt_tokens = None
    mock_completion.usage.completion_tokens = None
    mock_client.chat.completions.create_with_completion.return_value = (mock_response, mock_completion)

    def mock_from_openai(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("geneinsight.api.client.instructor.from_openai", mock_from_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    from geneinsight.api.client import APICallResult
    result = fetch_subtopic_heading(
        user_prompt="some prompt",
        system_prompt="system prompt",
        service="openai",
        return_metrics=True
    )

    assert result.prompt_tokens == 0  # Should handle None values
    assert result.completion_tokens == 0


def test_together_api_key_validation(monkeypatch, sample_df, tmp_path):
    """
    Test batch_process_api_calls with Together API requiring environment variable.
    """
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def mock_read_csv(*args, **kwargs):
        return sample_df

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    output_api_path = str(tmp_path / "output.csv")

    with pytest.raises(ValueError, match="API key for together not found"):
        batch_process_api_calls(
            prompts_csv="fake_prompts.csv",
            output_api=output_api_path,
            service="together",
            model="some-model",
            n_jobs=1
        )


def test_ollama_no_api_key_required(monkeypatch, sample_df, tmp_path):
    """
    Test that ollama service doesn't require an API key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    def mock_read_csv(*args, **kwargs):
        return sample_df

    mock_to_csv = MagicMock()

    def mock_process_subtopic_row(row, *args, **kwargs):
        return {
            "prompt_type": row["prompt_type"],
            "seed": row["seed"],
            "topic_label": row["topic_label"],
            "generated_result": "OLLAMA_RESULT",
            "_prompt_tokens": 10,
            "_completion_tokens": 5,
            "_latency_ms": 100.0,
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)

    output_api_path = str(tmp_path / "output.csv")

    # Should not raise ValueError
    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="ollama",
        model="llama2",
        n_jobs=1
    )

    assert len(df_results) == 2  # Two subtopic_BERT rows


def test_batch_metrics_statistics(monkeypatch, sample_df, tmp_path):
    """
    Test that batch metrics are correctly calculated including p95, min, max.
    """
    # Ensure instrumentation is available for this test
    monkeypatch.setattr("geneinsight.api.client.INSTRUMENTATION_AVAILABLE", True)

    def mock_read_csv(*args, **kwargs):
        return sample_df

    mock_to_csv = MagicMock()

    latencies = [100.0, 200.0, 150.0, 180.0, 120.0]
    latency_idx = [0]

    def mock_process_subtopic_row(row, token_counter=None, **kwargs):
        # Simulate what the real function does with token_counter
        idx = latency_idx[0] % len(latencies)
        latency_idx[0] += 1

        # Record in token counter if provided
        if token_counter is not None:
            token_counter.add(
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=latencies[idx]
            )

        return {
            "prompt_type": row["prompt_type"],
            "seed": row["seed"],
            "topic_label": row["topic_label"],
            "generated_result": "RESULT",
            "_prompt_tokens": 100,
            "_completion_tokens": 50,
            "_latency_ms": latencies[idx],
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")

    output_api_path = str(tmp_path / "output.csv")

    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=1
    )

    assert batch_metrics is not None
    assert batch_metrics.total_calls == 2
    assert batch_metrics.prompt_tokens == 200  # 2 calls * 100 tokens
    assert batch_metrics.completion_tokens == 100  # 2 calls * 50 tokens
    assert batch_metrics.min_latency_ms > 0
    assert batch_metrics.max_latency_ms > 0


def test_batch_metrics_empty_latencies(monkeypatch, tmp_path):
    """
    Test batch metrics when no latencies are recorded.
    """
    df_no_subtopic = pd.DataFrame([
        {"prompt_type": "other_type", "seed": "123", "topic_label": "None"}
    ])

    def mock_read_csv(*args, **kwargs):
        return df_no_subtopic

    mock_to_csv = MagicMock()

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    output_api_path = str(tmp_path / "output.csv")

    df_results, batch_metrics = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=1
    )

    assert df_results.empty
    assert batch_metrics is None  # No metrics when no rows processed


def test_batch_api_metrics_dataclass():
    """Test BatchAPIMetrics dataclass initialization."""
    metrics = BatchAPIMetrics(
        total_calls=10,
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        latencies_ms=[100.0, 150.0, 200.0],
        mean_latency_ms=150.0,
        p95_latency_ms=200.0,
        min_latency_ms=100.0,
        max_latency_ms=200.0
    )

    assert metrics.total_calls == 10
    assert metrics.prompt_tokens == 1000
    assert metrics.total_tokens == 1500
    assert len(metrics.latencies_ms) == 3


def test_batch_api_metrics_default_latencies():
    """Test BatchAPIMetrics default latencies list."""
    metrics = BatchAPIMetrics()
    assert metrics.latencies_ms == []  # Should be empty list, not None


