"""Unit tests for DRIGS empirical research experiment suite (Experiments A, B, E)."""

import pytest
from experiments.run_experiments import (
    RandomPlacementScheduler,
    run_all_experiments,
    run_experiment_a,
    run_experiment_b,
    run_experiment_e,
)
from drigs.core.models import ClusterState, DeviceState, DeviceType, Job, WorkloadSpec, WorkloadExecutionConfig, WorkerInfo, ComputeDevice


def test_random_placement_scheduler():
    """Test RandomPlacementScheduler functionality."""
    scheduler = RandomPlacementScheduler(seed=42)

    device = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="RTX 4090",
        total_memory_bytes=24 * 1024 * 1024 * 1024,
        available_memory_bytes=24 * 1024 * 1024 * 1024,
    )
    worker = WorkerInfo(
        worker_id="w-0",
        hostname="node0",
        ip_address="127.0.0.1",
        devices=[device],
        total_cpus=8,
        total_memory_bytes=32 * 1024 * 1024 * 1024,
    )
    cluster_state = ClusterState(workers={"w-0": worker})

    job = Job(
        job_id="test-job-1",
        name="test",
        spec=WorkloadSpec(
            name="test",
            execution=WorkloadExecutionConfig(entrypoint="python"),
        ),
    )

    allocations = scheduler.schedule([job], cluster_state)
    assert len(allocations) == 1
    assert allocations[0].job_id == "test-job-1"
    assert allocations[0].worker_id == "w-0"


def test_run_experiment_a():
    """Test execution of Experiment A (Scheduling Policy Evaluation)."""
    results = run_experiment_a(num_trials=2, trace_count=4, seed=42)

    assert isinstance(results, dict)
    assert results["experiment"] == "Experiment A: Scheduling Policy Evaluation"
    assert "results" in results
    res_dict = results["results"]

    assert "FIFO" in res_dict
    assert "Priority" in res_dict
    assert "BestFit" in res_dict
    assert "MemoryAware" in res_dict
    assert "TopologyAware" in res_dict

    for sched_name, metrics in res_dict.items():
        assert "throughput_mean_jobs_per_sec" in metrics
        assert "mean_wait_time_sec" in metrics
        assert "vram_fragmentation_pct" in metrics


def test_run_experiment_b():
    """Test execution of Experiment B (Topology-Aware vs Random Placement)."""
    results = run_experiment_b(num_trials=2, trace_count=4, seed=100)

    assert isinstance(results, dict)
    assert results["experiment"] == "Experiment B: Topology-Aware vs Random Placement"
    assert "topology_score_topology_aware" in results
    assert "topology_score_random" in results
    assert "topology_score_improvement_pct" in results

    assert results["topology_score_topology_aware"] >= 0.0
    assert results["topology_score_random"] >= 0.0


def test_run_experiment_e():
    """Test execution of Experiment E (Fault Recovery Latency & Overhead)."""
    results = run_experiment_e(num_trials=2, trace_count=4, seed=200)

    assert isinstance(results, dict)
    assert results["experiment"] == "Experiment E: Fault Recovery Latency & Rescheduling Overhead"
    assert "mean_rescheduling_latency_ms" in results
    assert "mean_job_completion_rate_pct" in results
    assert "total_restored_checkpoints" in results

    assert results["mean_rescheduling_latency_ms"] >= 0.0
    assert 0.0 <= results["mean_job_completion_rate_pct"] <= 100.0


def test_run_all_experiments():
    """Test full experiment suite execution."""
    suite_summary = run_all_experiments(num_trials=1)

    assert isinstance(suite_summary, dict)
    assert suite_summary["suite"] == "DRIGS Research Benchmark Suite"
    assert "experiment_a" in suite_summary
    assert "experiment_b" in suite_summary
    assert "experiment_e" in suite_summary
