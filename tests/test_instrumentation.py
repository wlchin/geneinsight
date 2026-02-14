# tests/test_instrumentation.py
"""
Tests for the geneinsight.instrumentation module.
"""

import os
import json
import time
import pytest
import tempfile
import threading
from unittest.mock import patch, MagicMock

from geneinsight.instrumentation import (
    TokenUsage,
    StageMetrics,
    PipelineMetrics,
    MetricsCollector,
    TokenCounter,
    format_duration,
    calculate_cost,
    format_metrics_summary,
    timed,
)


# ============================================================================
# Tests for TokenUsage
# ============================================================================

class TestTokenUsage:

    def test_basic_instantiation(self):
        """Test basic TokenUsage creation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_zero_tokens(self):
        """Test TokenUsage with zero tokens."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_addition(self):
        """Test adding two TokenUsage objects."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100)
        result = usage1 + usage2
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 150
        assert result.total_tokens == 450

    def test_to_dict(self):
        """Test TokenUsage to_dict conversion."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        d = usage.to_dict()
        assert d == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }


# ============================================================================
# Tests for StageMetrics
# ============================================================================

class TestStageMetrics:

    def test_basic_instantiation(self):
        """Test basic StageMetrics creation."""
        stage = StageMetrics(name="Test Stage", duration_seconds=30.5)
        assert stage.name == "Test Stage"
        assert stage.duration_seconds == 30.5

    def test_api_latency_stats_empty(self):
        """Test API latency stats with no latencies."""
        stage = StageMetrics(name="Test", api_latencies_ms=[])
        stats = stage.api_latency_stats
        assert stats == {}

    def test_api_latency_stats_few_latencies(self):
        """Test API latency stats with fewer than 20 latencies."""
        latencies = [100.0, 150.0, 200.0, 120.0, 180.0]
        stage = StageMetrics(name="Test", api_latencies_ms=latencies)
        stats = stage.api_latency_stats

        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 200.0
        assert stats["count"] == 5
        # p95 should be max when fewer than 20 latencies
        assert stats["p95_ms"] == 200.0

    def test_api_latency_stats_many_latencies(self):
        """Test API latency stats with 20+ latencies (proper p95 calculation)."""
        latencies = list(range(100, 200))  # 100 latencies from 100 to 199
        stage = StageMetrics(name="Test", api_latencies_ms=latencies)
        stats = stage.api_latency_stats

        assert stats["min_ms"] == 100
        assert stats["max_ms"] == 199
        assert stats["count"] == 100
        # p95 should be at index 95
        assert stats["p95_ms"] == 195

    def test_duration_formatted(self):
        """Test duration formatting."""
        stage = StageMetrics(name="Test", duration_seconds=90.5)
        assert "1m" in stage.duration_formatted or "90" in stage.duration_formatted

    def test_to_dict(self):
        """Test StageMetrics to_dict conversion."""
        stage = StageMetrics(
            name="Test Stage",
            duration_seconds=30.5,
            api_calls=10,
            token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500)
        )
        d = stage.to_dict()
        assert d["name"] == "Test Stage"
        assert d["duration_seconds"] == 30.5
        assert d["api_calls"] == 10
        assert d["token_usage"]["total_tokens"] == 1500


# ============================================================================
# Tests for calculate_cost
# ============================================================================

class TestCalculateCost:

    def test_gpt4o_mini_cost(self):
        """Test cost calculation for gpt-4o-mini model."""
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = calculate_cost(usage, model="gpt-4o-mini")

        # gpt-4o-mini: $0.15 input, $0.60 output per 1M tokens
        expected_input = 1_000_000 * 0.15 / 1_000_000  # $0.15
        expected_output = 500_000 * 0.60 / 1_000_000   # $0.30
        assert cost["input_cost"] == pytest.approx(expected_input)
        assert cost["output_cost"] == pytest.approx(expected_output)
        assert cost["total_cost"] == pytest.approx(expected_input + expected_output)

    def test_gpt4o_cost(self):
        """Test cost calculation for gpt-4o model."""
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = calculate_cost(usage, model="gpt-4o")

        # gpt-4o: $2.50 input, $10.00 output per 1M tokens
        expected_input = 1_000_000 * 2.50 / 1_000_000   # $2.50
        expected_output = 500_000 * 10.00 / 1_000_000   # $5.00
        assert cost["input_cost"] == pytest.approx(expected_input)
        assert cost["output_cost"] == pytest.approx(expected_output)

    def test_gpt4_turbo_cost(self):
        """Test cost calculation for gpt-4-turbo model."""
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = calculate_cost(usage, model="gpt-4-turbo")

        # gpt-4-turbo: $10.00 input, $30.00 output per 1M tokens
        expected_input = 1_000_000 * 10.00 / 1_000_000  # $10.00
        expected_output = 500_000 * 30.00 / 1_000_000   # $15.00
        assert cost["input_cost"] == pytest.approx(expected_input)
        assert cost["output_cost"] == pytest.approx(expected_output)

    def test_unknown_model_fallback(self):
        """Test that unknown models fall back to gpt-4o-mini rates."""
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        cost = calculate_cost(usage, model="unknown-model-xyz")

        # Should use gpt-4o-mini rates
        expected_input = 1_000_000 * 0.15 / 1_000_000
        assert cost["input_cost"] == pytest.approx(expected_input)
        assert cost["model"] == "unknown-model-xyz"

    def test_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        cost = calculate_cost(usage, model="gpt-4o-mini")

        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0


# ============================================================================
# Tests for TokenCounter
# ============================================================================

class TestTokenCounter:

    def test_basic_add(self):
        """Test basic token counter add operation."""
        counter = TokenCounter()
        counter.add(prompt_tokens=100, completion_tokens=50, latency_ms=150.0)

        total = counter.get_total()
        assert total.prompt_tokens == 100
        assert total.completion_tokens == 50
        assert counter.get_call_count() == 1
        assert len(counter.get_latencies()) == 1
        assert counter.get_latencies()[0] == 150.0

    def test_multiple_adds(self):
        """Test multiple add operations."""
        counter = TokenCounter()
        counter.add(prompt_tokens=100, completion_tokens=50)
        counter.add(prompt_tokens=200, completion_tokens=100)
        counter.add(prompt_tokens=150, completion_tokens=75)

        total = counter.get_total()
        assert total.prompt_tokens == 450
        assert total.completion_tokens == 225
        assert counter.get_call_count() == 3

    def test_thread_safety_concurrent_adds(self):
        """Test thread safety with concurrent add operations."""
        counter = TokenCounter()
        num_threads = 10
        adds_per_thread = 100

        def add_tokens():
            for _ in range(adds_per_thread):
                counter.add(prompt_tokens=1, completion_tokens=1, latency_ms=1.0)

        threads = [threading.Thread(target=add_tokens) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = counter.get_total()
        expected_total = num_threads * adds_per_thread
        assert total.prompt_tokens == expected_total
        assert total.completion_tokens == expected_total
        assert counter.get_call_count() == expected_total
        assert len(counter.get_latencies()) == expected_total

    def test_reset(self):
        """Test resetting the counter."""
        counter = TokenCounter()
        counter.add(prompt_tokens=100, completion_tokens=50, latency_ms=100.0)
        counter.add(prompt_tokens=100, completion_tokens=50, latency_ms=100.0)

        counter.reset()

        total = counter.get_total()
        assert total.prompt_tokens == 0
        assert total.completion_tokens == 0
        assert counter.get_call_count() == 0
        assert len(counter.get_latencies()) == 0

    def test_getters(self):
        """Test all getter methods."""
        counter = TokenCounter()
        counter.add(prompt_tokens=100, completion_tokens=50, latency_ms=100.0)

        assert counter.get_total().total_tokens == 150
        assert counter.get_call_count() == 1
        assert counter.get_latencies() == [100.0]


# ============================================================================
# Tests for timed decorator
# ============================================================================

class TestTimedDecorator:

    def test_successful_timing(self):
        """Test that timed decorator works on successful function."""
        @timed
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"

    def test_exception_propagation(self):
        """Test that exceptions are propagated through timed decorator."""
        @timed
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_func()


# ============================================================================
# Tests for MetricsCollector
# ============================================================================

class TestMetricsCollector:

    def test_basic_initialization(self):
        """Test basic MetricsCollector initialization."""
        collector = MetricsCollector(run_id="test_run", model="gpt-4o")
        assert collector.metrics.run_id == "test_run"
        assert collector.metrics.model == "gpt-4o"

    def test_time_stage_context_manager(self):
        """Test the time_stage context manager."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        with collector.time_stage("Test Stage"):
            time.sleep(0.01)

        assert "Test Stage" in collector.metrics.stages
        stage = collector.metrics.stages["Test Stage"]
        assert stage.duration_seconds > 0

    def test_time_stage_disabled(self):
        """Test time_stage when collector is disabled."""
        collector = MetricsCollector(run_id="test_run")
        collector.set_enabled(False)

        with collector.time_stage("Test Stage"):
            pass

        # Stage should not be recorded when disabled
        assert "Test Stage" not in collector.metrics.stages

    def test_record_api_call(self):
        """Test recording API call metrics."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        with collector.time_stage("API Stage"):
            collector.record_api_call(
                latency_ms=150.5,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50)
            )

        stage = collector.metrics.stages["API Stage"]
        assert stage.api_calls == 1
        assert 150.5 in stage.api_latencies_ms
        assert stage.token_usage.total_tokens == 150

    def test_record_api_call_no_stage(self):
        """Test recording API call without current stage logs warning."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        # Should not raise, just log warning
        collector.record_api_call(latency_ms=100.0)

    def test_add_stage_metadata(self):
        """Test adding metadata to a stage."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        with collector.time_stage("Test Stage"):
            pass

        collector.add_stage_metadata("Test Stage", "key1", "value1")
        assert collector.metrics.stages["Test Stage"].metadata["key1"] == "value1"

    def test_to_json_serialization(self):
        """Test JSON serialization of metrics."""
        collector = MetricsCollector(run_id="test_run", model="gpt-4o")
        collector.start_pipeline()

        with collector.time_stage("Stage 1"):
            time.sleep(0.01)

        json_str = collector.to_json()
        data = json.loads(json_str)

        assert data["run_id"] == "test_run"
        assert data["model"] == "gpt-4o"
        assert "Stage 1" in data["stages"]

    def test_save_to_file(self, tmp_path):
        """Test saving metrics to a file."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        with collector.time_stage("Stage 1"):
            pass

        output_path = tmp_path / "metrics.json"
        collector.save(str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["run_id"] == "test_run"

    def test_finalize(self):
        """Test finalizing metrics collection."""
        collector = MetricsCollector(run_id="test_run")
        collector.start_pipeline()

        with collector.time_stage("Stage 1"):
            time.sleep(0.01)
            collector.record_api_call(
                latency_ms=100.0,
                token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50)
            )

        metrics = collector.finalize()

        assert metrics.total_duration_seconds > 0
        assert metrics.total_token_usage.total_tokens == 150
        assert metrics.total_api_calls == 1


# ============================================================================
# Tests for format_duration
# ============================================================================

class TestFormatDuration:

    def test_seconds(self):
        """Test formatting durations less than a minute."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(0.5) == "0.5s"

    def test_minutes(self):
        """Test formatting durations in minutes."""
        result = format_duration(90)
        assert "1m" in result and "30s" in result

    def test_hours(self):
        """Test formatting durations in hours."""
        result = format_duration(3700)  # 1 hour 1 minute 40 seconds
        assert "1h" in result


# ============================================================================
# Tests for format_metrics_summary
# ============================================================================

class TestFormatMetricsSummary:

    def test_with_stages(self):
        """Test formatting metrics summary with stages."""
        metrics = PipelineMetrics(
            run_id="test_run",
            total_duration_seconds=100.0
        )
        metrics.stages["Stage 1"] = StageMetrics(
            name="Stage 1",
            duration_seconds=50.0
        )
        metrics.stages["Stage 2"] = StageMetrics(
            name="Stage 2",
            duration_seconds=30.0
        )

        summary = format_metrics_summary(metrics)

        assert "PIPELINE METRICS SUMMARY" in summary
        assert "test_run" in summary
        assert "Stage 1" in summary
        assert "Stage 2" in summary

    def test_empty_stages(self):
        """Test formatting metrics summary with no stages."""
        metrics = PipelineMetrics(
            run_id="empty_run",
            total_duration_seconds=0.0
        )

        summary = format_metrics_summary(metrics)

        assert "PIPELINE METRICS SUMMARY" in summary
        assert "empty_run" in summary

    def test_with_token_usage(self):
        """Test formatting with token usage."""
        metrics = PipelineMetrics(
            run_id="test_run",
            total_duration_seconds=100.0,
            total_token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500),
            model="gpt-4o-mini"
        )

        summary = format_metrics_summary(metrics, show_cost=True)

        assert "TOKEN USAGE" in summary
        assert "1,000" in summary or "1000" in summary  # prompt tokens
        assert "500" in summary  # completion tokens

    def test_with_api_metrics(self):
        """Test formatting with API call statistics."""
        metrics = PipelineMetrics(
            run_id="test_run",
            total_duration_seconds=100.0,
            total_api_calls=10
        )
        metrics.stages["API Stage"] = StageMetrics(
            name="API Stage",
            duration_seconds=50.0,
            api_calls=10,
            api_latencies_ms=[100.0, 150.0, 200.0, 120.0, 180.0] * 4  # 20 latencies
        )

        summary = format_metrics_summary(metrics)

        assert "API CALL STATISTICS" in summary
        assert "Total calls: 10" in summary


# ============================================================================
# Tests for PipelineMetrics
# ============================================================================

class TestPipelineMetrics:

    def test_stages_by_duration(self):
        """Test sorting stages by duration."""
        metrics = PipelineMetrics(run_id="test")
        metrics.stages["Slow"] = StageMetrics(name="Slow", duration_seconds=100.0)
        metrics.stages["Fast"] = StageMetrics(name="Fast", duration_seconds=10.0)
        metrics.stages["Medium"] = StageMetrics(name="Medium", duration_seconds=50.0)

        sorted_stages = metrics.stages_by_duration
        assert sorted_stages[0].name == "Slow"
        assert sorted_stages[1].name == "Medium"
        assert sorted_stages[2].name == "Fast"

    def test_slowest_stages(self):
        """Test getting top 3 slowest stages."""
        metrics = PipelineMetrics(run_id="test")
        for i in range(5):
            metrics.stages[f"Stage{i}"] = StageMetrics(
                name=f"Stage{i}",
                duration_seconds=float(i * 10)
            )

        slowest = metrics.slowest_stages
        assert len(slowest) == 3
        assert slowest[0].duration_seconds == 40.0

    def test_to_dict(self):
        """Test PipelineMetrics to_dict conversion."""
        metrics = PipelineMetrics(
            run_id="test_run",
            total_duration_seconds=100.0,
            model="gpt-4o"
        )
        metrics.stages["Stage 1"] = StageMetrics(
            name="Stage 1",
            duration_seconds=50.0
        )

        d = metrics.to_dict()
        assert d["run_id"] == "test_run"
        assert d["model"] == "gpt-4o"
        assert "Stage 1" in d["stages"]
