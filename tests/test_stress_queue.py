"""Unit and Integration Tests for High-Scale Queue Stress Benchmark."""

import os
import tempfile
import pytest

from experiments.stress_test_queue import benchmark_queue_scaling


def test_benchmark_queue_scaling_in_memory():
    """Verify queue stress test runs cleanly under in-memory mode."""
    results = benchmark_queue_scaling(
        job_counts=[50, 100],
        use_sqlite=False,
        seed=123,
    )

    assert results["benchmark"] == "High-Scale Job Queue Stress & Bottleneck Benchmark"
    assert "50" in results["results_by_count"]
    assert "100" in results["results_by_count"]

    res_50 = results["results_by_count"]["50"]
    assert res_50["num_jobs"] == 50
    assert res_50["enqueue_throughput_jobs_per_sec"] > 0
    assert "FIFO" in res_50["scheduler_metrics"]
    assert "Priority" in res_50["scheduler_metrics"]


def test_benchmark_queue_scaling_sqlite():
    """Verify queue stress test runs cleanly with SQLite store enabled."""
    results = benchmark_queue_scaling(
        job_counts=[50],
        use_sqlite=True,
        seed=456,
    )

    assert "50" in results["results_by_count"]
    res_50 = results["results_by_count"]["50"]
    assert res_50["num_jobs"] == 50
    assert res_50["avg_status_transition_ms"] >= 0.0
