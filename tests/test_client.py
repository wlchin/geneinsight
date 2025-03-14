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
    mock_client.chat.completions.create.return_value = mock_response

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
    mock_client.chat.completions.create.assert_called_once()


def test_fetch_subtopic_heading_no_topic_attribute(monkeypatch):
    """
    Test fetch_subtopic_heading when the API response does not have a 'topic'
    attribute. The function should return str(response).
    """
    monkeypatch.setattr("geneinsight.api.client.APIS_AVAILABLE", True)

    mock_client = MagicMock()
    # Simulate a response that has no 'topic' attribute
    mock_response = "Just a string response"
    mock_client.chat.completions.create.return_value = mock_response

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

    def mock_fetch_subtopic_heading(*args, **kwargs):
        return "Mocked Subtopic"

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
    row_data = {
        "seed": "99999",
        "topic_label": "Missing Max Words",
        "major_transcript": "A large transcript that needs subtopic generation.",
        "prompt_type": "subtopic_BERT"
        # note: no "max_words"
    }
    row = pd.Series(row_data)

    def mock_fetch_subtopic_heading(*args, **kwargs):
        return "Mocked Subtopic"

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
    some 'subtopic_BERT' rows. Ensure it processes them in parallel and
    writes to CSV. We'll mock read_csv, to_csv, and the process_subtopic_row
    function.
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
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)
    # Provide an environment key for openai
    monkeypatch.setenv("OPENAI_API_KEY", "some_test_key")

    output_api_path = str(tmp_path / "output.csv")
    df_results = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=2
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
    df_results = batch_process_api_calls(
        prompts_csv="fake_prompts.csv",
        output_api=output_api_path,
        service="openai",
        model="gpt-4o-mini",
        n_jobs=1
    )

    assert df_results.empty
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
        }

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)
    monkeypatch.setattr("geneinsight.api.client.process_subtopic_row", mock_process_subtopic_row)

    output_api_path = str(tmp_path / "output.csv")

    df_results = batch_process_api_calls(
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


