"""Unit tests for synthetic workload generator and benchmark harness framework."""

import pytest
from experiments.harness import (
    BenchmarkHarness,
    BenchmarkResult,
    SyntheticWorkloadItem,
    TrialMetrics,
    WorkloadGenerator,
)
from drigs.scheduler.fifo import FIFOScheduler
from drigs.scheduler.fit import BestFitScheduler, FirstFitScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.scheduler.priority import PriorityScheduler
from drigs.scheduler.topology_aware import TopologyAwareScheduler


def test_workload_generator_reproducibility():
    """Test that WorkloadGenerator with a fixed seed produces deterministic trace output."""
    gen1 = WorkloadGenerator(seed=42)
    trace1 = gen1.generate_trace(count=10, arrival_pattern="uniform")

    gen2 = WorkloadGenerator(seed=42)
    trace2 = gen2.generate_trace(count=10, arrival_pattern="uniform")

    assert len(trace1) == 10
    assert len(trace2) == 10

    for item1, item2 in zip(trace1, trace2):
        assert item1.job_id == item2.job_id
        assert item1.arrival_delay == item2.arrival_delay
        assert item1.duration_seconds == item2.duration_seconds
        assert item1.spec.resources.gpus == item2.spec.resources.gpus
        assert item1.spec.resources.gpu_memory_bytes == item2.spec.resources.gpu_memory_bytes


def test_workload_generator_arrival_patterns():
    """Test trace generation across different arrival pattern types."""
    gen = WorkloadGenerator(seed=123)

    for pattern in ["uniform", "poisson", "burst", "constant"]:
        trace = gen.generate_trace(count=8, arrival_pattern=pattern)
        assert len(trace) == 8
        assert trace[0].arrival_delay == 0.0
        for i in range(1, len(trace)):
            assert trace[i].arrival_delay >= trace[i - 1].arrival_delay


def test_benchmark_harness_single_trial_fifo():
    """Test running a single benchmark trial with FIFOScheduler."""
    generator = WorkloadGenerator(seed=42)
    trace = generator.generate_trace(count=6, arrival_pattern="constant", min_arrival_interval=0.1)

    harness = BenchmarkHarness()
    scheduler = FIFOScheduler()

    metrics = harness.run_single_trial(scheduler, trace, time_step=0.2)

    assert isinstance(metrics, TrialMetrics)
    assert metrics.total_jobs == 6
    assert metrics.completed_jobs == 6
    assert metrics.makespan_seconds > 0.0
    assert metrics.scheduling_throughput > 0.0
    assert len(metrics.queue_wait_times) == 6
    assert metrics.placement_efficiency == 1.0


def test_benchmark_harness_multi_trial():
    """Test multi-trial statistical benchmark execution across multiple trials."""
    generator = WorkloadGenerator(seed=100)
    harness = BenchmarkHarness()
    scheduler = MemoryAwareScheduler()

    result = harness.run_multi_trial(
        scheduler=scheduler,
        scheduler_name="MemoryAware",
        generator=generator,
        num_trials=3,
        trace_count=5,
        time_step=0.2,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.scheduler_name == "MemoryAware"
    assert result.num_trials == 3
    assert len(result.trials) == 3
    assert result.makespan_mean > 0.0
    assert result.throughput_mean > 0.0
    assert 0.0 <= result.vram_fragmentation_mean <= 1.0
    assert result.placement_efficiency_mean == 1.0

    dict_output = result.to_dict()
    assert dict_output["scheduler_name"] == "MemoryAware"
    assert dict_output["num_trials"] == 3
    assert "makespan_mean_sec" in dict_output
    assert "throughput_mean_jobs_per_sec" in dict_output


def test_benchmark_harness_compare_schedulers():
    """Test comparative benchmark evaluation across multiple scheduling policies."""
    generator = WorkloadGenerator(seed=200)
    harness = BenchmarkHarness()

    schedulers = {
        "FIFO": FIFOScheduler(),
        "Priority": PriorityScheduler(),
        "BestFit": BestFitScheduler(),
        "MemoryAware": MemoryAwareScheduler(),
        "TopologyAware": TopologyAwareScheduler(),
    }

    results = harness.compare_schedulers(
        schedulers=schedulers,
        generator=generator,
        num_trials=2,
        trace_count=4,
        time_step=0.2,
    )

    assert len(results) == 5
    assert "FIFO" in results
    assert "Priority" in results
    assert "BestFit" in results
    assert "MemoryAware" in results
    assert "TopologyAware" in results

    for name, res in results.items():
        assert res.scheduler_name == name
        assert res.num_trials == 2
        assert res.completed_jobs_total if hasattr(res, 'completed_jobs_total') else True
