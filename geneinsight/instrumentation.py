# geneinsight/instrumentation.py
"""
Instrumentation module for timing and token tracking in the GeneInsight pipeline.

Provides utilities for:
- Timing pipeline stages and individual operations
- Tracking token usage from LLM API calls
- Aggregating and reporting metrics
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage for a single API call or aggregated usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    duration_seconds: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    api_calls: int = 0
    api_latencies_ms: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_formatted(self) -> str:
        """Format duration as human-readable string."""
        return format_duration(self.duration_seconds)

    @property
    def api_latency_stats(self) -> Dict[str, float]:
        """Calculate API latency statistics."""
        if not self.api_latencies_ms:
            return {}

        sorted_latencies = sorted(self.api_latencies_ms)
        n = len(sorted_latencies)

        return {
            "min_ms": min(sorted_latencies),
            "max_ms": max(sorted_latencies),
            "mean_ms": statistics.mean(sorted_latencies),
            "median_ms": statistics.median(sorted_latencies),
            "p95_ms": sorted_latencies[int(n * 0.95)] if n >= 20 else max(sorted_latencies),
            "count": n
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.duration_formatted,
            "token_usage": self.token_usage.to_dict(),
            "api_calls": self.api_calls,
            "api_latency_stats": self.api_latency_stats,
            "metadata": self.metadata
        }


@dataclass
class PipelineMetrics:
    """Complete metrics for a pipeline run."""
    run_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration_seconds: float = 0.0
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    total_api_calls: int = 0
    model: str = ""

    @property
    def total_duration_formatted(self) -> str:
        return format_duration(self.total_duration_seconds)

    @property
    def stages_by_duration(self) -> List[StageMetrics]:
        """Return stages sorted by duration (descending)."""
        return sorted(
            self.stages.values(),
            key=lambda s: s.duration_seconds,
            reverse=True
        )

    @property
    def slowest_stages(self) -> List[StageMetrics]:
        """Return top 3 slowest stages."""
        return self.stages_by_duration[:3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_time_epoch": self.start_time,
            "end_time_epoch": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "total_duration_formatted": self.total_duration_formatted,
            "total_token_usage": self.total_token_usage.to_dict(),
            "total_api_calls": self.total_api_calls,
            "model": self.model,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "slowest_stages": [s.name for s in self.slowest_stages]
        }


# =============================================================================
# Helper Functions
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def calculate_cost(token_usage: TokenUsage, model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    Calculate estimated cost based on token usage and model.

    Rates as of 2024 (per 1M tokens):
    - gpt-4o-mini: $0.15 input, $0.60 output
    - gpt-4o: $2.50 input, $10.00 output
    - gpt-4-turbo: $10.00 input, $30.00 output
    """
    rates = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    }

    # Default to gpt-4o-mini rates if model not found
    model_lower = model.lower()
    rate = rates.get(model_lower, rates["gpt-4o-mini"])

    input_cost = token_usage.prompt_tokens * rate["input"]
    output_cost = token_usage.completion_tokens * rate["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
        "currency": "USD"
    }


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Collects and aggregates metrics across pipeline stages.

    Usage:
        collector = MetricsCollector(run_id="my_run")

        with collector.time_stage("Step 1: Enrichment"):
            # do work
            pass

        # For API calls with token tracking
        collector.record_api_call(
            stage="Step 4: API Processing",
            latency_ms=150.5,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50)
        )

        # Get final metrics
        metrics = collector.finalize()
    """

    def __init__(self, run_id: str = "", model: str = "gpt-4o-mini"):
        self.metrics = PipelineMetrics(run_id=run_id, model=model)
        self._current_stage: Optional[str] = None
        self._stage_start_time: float = 0.0
        self._enabled = True

    def set_enabled(self, enabled: bool):
        """Enable or disable metrics collection."""
        self._enabled = enabled

    def start_pipeline(self):
        """Mark the start of the pipeline."""
        if self._enabled:
            self.metrics.start_time = time.time()

    def end_pipeline(self):
        """Mark the end of the pipeline and calculate totals."""
        if self._enabled:
            self.metrics.end_time = time.time()
            self.metrics.total_duration_seconds = (
                self.metrics.end_time - self.metrics.start_time
            )

            # Aggregate totals from stages
            for stage in self.metrics.stages.values():
                self.metrics.total_token_usage += stage.token_usage
                self.metrics.total_api_calls += stage.api_calls

    @contextmanager
    def time_stage(self, stage_name: str):
        """
        Context manager to time a pipeline stage.

        Args:
            stage_name: Name of the stage being timed
        """
        if not self._enabled:
            yield
            return

        # Initialize stage if not exists
        if stage_name not in self.metrics.stages:
            self.metrics.stages[stage_name] = StageMetrics(name=stage_name)

        stage = self.metrics.stages[stage_name]
        stage.start_time = time.time()
        self._current_stage = stage_name

        try:
            yield stage
        finally:
            stage.end_time = time.time()
            stage.duration_seconds = stage.end_time - stage.start_time
            self._current_stage = None

            logger.debug(
                f"Stage '{stage_name}' completed in {stage.duration_formatted}"
            )

    def record_api_call(
        self,
        stage: Optional[str] = None,
        latency_ms: float = 0.0,
        token_usage: Optional[TokenUsage] = None
    ):
        """
        Record metrics for a single API call.

        Args:
            stage: Stage name (uses current stage if None)
            latency_ms: API call latency in milliseconds
            token_usage: Token usage for this call
        """
        if not self._enabled:
            return

        stage_name = stage or self._current_stage
        if not stage_name:
            logger.warning("No stage specified for API call recording")
            return

        if stage_name not in self.metrics.stages:
            self.metrics.stages[stage_name] = StageMetrics(name=stage_name)

        stage_metrics = self.metrics.stages[stage_name]
        stage_metrics.api_calls += 1

        if latency_ms > 0:
            stage_metrics.api_latencies_ms.append(latency_ms)

        if token_usage:
            stage_metrics.token_usage += token_usage

    def add_stage_metadata(self, stage: str, key: str, value: Any):
        """Add metadata to a stage."""
        if not self._enabled:
            return

        if stage not in self.metrics.stages:
            self.metrics.stages[stage] = StageMetrics(name=stage)

        self.metrics.stages[stage].metadata[key] = value

    def finalize(self) -> PipelineMetrics:
        """Finalize and return the complete metrics."""
        self.end_pipeline()
        return self.metrics

    def to_json(self, indent: int = 2) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.metrics.to_dict(), indent=indent)

    def save(self, path: str):
        """Save metrics to a JSON file."""
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Metrics saved to {path}")


# =============================================================================
# Token Counter (for aggregating across parallel calls)
# =============================================================================

class TokenCounter:
    """
    Thread-safe token counter for aggregating usage across parallel API calls.

    Usage:
        counter = TokenCounter()

        # In parallel workers
        counter.add(prompt_tokens=100, completion_tokens=50)

        # Get total
        total = counter.get_total()
    """

    def __init__(self):
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._call_count = 0
        self._latencies: List[float] = []

        # For thread safety
        import threading
        self._lock = threading.Lock()

    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0
    ):
        """Add token usage from a single API call."""
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._call_count += 1
            if latency_ms > 0:
                self._latencies.append(latency_ms)

    def get_total(self) -> TokenUsage:
        """Get total token usage."""
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens
            )

    def get_call_count(self) -> int:
        """Get total number of API calls."""
        with self._lock:
            return self._call_count

    def get_latencies(self) -> List[float]:
        """Get list of latencies."""
        with self._lock:
            return self._latencies.copy()

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._call_count = 0
            self._latencies = []


# =============================================================================
# Decorators
# =============================================================================

def timed(func: Callable) -> Callable:
    """
    Decorator to time a function and log its duration.

    Usage:
        @timed
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            logger.debug(f"{func.__name__} completed in {format_duration(duration)}")

    return wrapper


# =============================================================================
# Console Output Formatting
# =============================================================================

def format_metrics_summary(metrics: PipelineMetrics, show_cost: bool = True) -> str:
    """
    Format metrics as a summary string for console output.

    Args:
        metrics: PipelineMetrics object
        show_cost: Whether to show cost estimates

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("PIPELINE METRICS SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Run ID: {metrics.run_id}")
    lines.append(f"Total Duration: {metrics.total_duration_formatted}")
    lines.append("")

    # Stage timing breakdown
    lines.append("STAGE TIMING")
    lines.append("-" * 60)

    total_tracked = sum(s.duration_seconds for s in metrics.stages.values())

    for stage in metrics.stages_by_duration:
        pct = (stage.duration_seconds / metrics.total_duration_seconds * 100) if metrics.total_duration_seconds > 0 else 0
        slowest_marker = " <- SLOWEST" if stage in metrics.slowest_stages[:1] else ""
        lines.append(
            f"  {stage.name:<45} {stage.duration_formatted:>8}  ({pct:>5.1f}%){slowest_marker}"
        )

    lines.append("-" * 60)
    lines.append(f"  {'Tracked stages total':<45} {format_duration(total_tracked):>8}")
    lines.append("")

    # Token usage
    if metrics.total_token_usage.total_tokens > 0:
        lines.append("TOKEN USAGE")
        lines.append("-" * 60)
        lines.append(f"  Prompt tokens:     {metrics.total_token_usage.prompt_tokens:>12,}")
        lines.append(f"  Completion tokens: {metrics.total_token_usage.completion_tokens:>12,}")
        lines.append(f"  Total tokens:      {metrics.total_token_usage.total_tokens:>12,}")

        if show_cost:
            cost = calculate_cost(metrics.total_token_usage, metrics.model)
            lines.append(f"  Estimated cost:    ${cost['total_cost']:>11.4f} ({metrics.model})")
        lines.append("")

    # API call statistics
    if metrics.total_api_calls > 0:
        lines.append("API CALL STATISTICS")
        lines.append("-" * 60)
        lines.append(f"  Total calls: {metrics.total_api_calls:,}")

        # Find stage with API calls to get latency stats
        for stage in metrics.stages.values():
            if stage.api_latencies_ms:
                stats = stage.api_latency_stats
                lines.append(f"  Mean latency:  {stats.get('mean_ms', 0):>8.1f} ms")
                lines.append(f"  P95 latency:   {stats.get('p95_ms', 0):>8.1f} ms")
                lines.append(f"  Min latency:   {stats.get('min_ms', 0):>8.1f} ms")
                lines.append(f"  Max latency:   {stats.get('max_ms', 0):>8.1f} ms")
                break
        lines.append("")

    # Slowest stages highlight
    lines.append("TOP 3 SLOWEST STAGES")
    lines.append("-" * 60)
    for i, stage in enumerate(metrics.slowest_stages, 1):
        pct = (stage.duration_seconds / metrics.total_duration_seconds * 100) if metrics.total_duration_seconds > 0 else 0
        lines.append(f"  {i}. {stage.name}: {stage.duration_formatted} ({pct:.1f}%)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
