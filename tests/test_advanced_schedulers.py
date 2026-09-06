"""Unit tests for BinPackScheduler and PriorityScheduler in drigs.scheduler."""

from datetime import datetime, timedelta, timezone
import pytest

from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.scheduler.binpack import BinPackScheduler
from drigs.scheduler.priority import PriorityScheduler


def _make_gpu(dev_id: str) -> ComputeDevice:
    return ComputeDevice(
        device_id=dev_id,
        device_type=DeviceType.GPU,
        model_name="RTX 4090",
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=24 * 1024**3,
    )


def test_binpack_scheduler_protocol():
    scheduler = BinPackScheduler()
    assert isinstance(scheduler, Scheduler)


def test_binpack_scheduler_packing():
    # Worker 1: 4 GPUs, 64 GB RAM
    w1_devs = [_make_gpu(f"w1-g{i}") for i in range(4)]
    worker1 = WorkerInfo(
        worker_id="worker-empty",
        hostname="node-empty",
        ip_address="127.0.0.1",
        devices=w1_devs,
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )

    # Worker 2: 1 GPU, 16 GB RAM (Smaller remaining surplus)
    w2_devs = [_make_gpu("w2-g0")]
    worker2 = WorkerInfo(
        worker_id="worker-small",
        hostname="node-small",
        ip_address="127.0.0.2",
        devices=w2_devs,
        total_cpus=4,
        total_memory_bytes=16 * 1024**3,
        status=DeviceState.HEALTHY,
    )

    cluster = ClusterState(workers={"worker-empty": worker1, "worker-small": worker2})

    spec = WorkloadSpec(
        name="small-job",
        resources=ResourceRequirements(gpus=1, cpus=2, memory_bytes=8 * 1024**3),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )
    job = Job(name="job-1", spec=spec)

    scheduler = BinPackScheduler()
    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    alloc = allocations[0]
    # Should pick worker-small to pack tightly and keep worker-empty free
    assert alloc.worker_id == "worker-small"


def test_priority_scheduler_protocol():
    scheduler = PriorityScheduler()
    assert isinstance(scheduler, Scheduler)


def test_priority_scheduler_strict_ordering():
    dev = _make_gpu("g0")
    worker = WorkerInfo(
        worker_id="worker-single",
        hostname="node-1",
        ip_address="127.0.0.1",
        devices=[dev],
        total_cpus=2,
        total_memory_bytes=8 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    cluster = ClusterState(workers={"worker-single": worker})

    def _make_job(name: str, priority: int) -> Job:
        spec = WorkloadSpec(
            name=name,
            resources=ResourceRequirements(gpus=1, cpus=1),
            execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
        )
        return Job(name=name, priority=priority, spec=spec)

    j_low = _make_job("job-low", priority=1)
    j_high = _make_job("job-high", priority=10)
    j_med = _make_job("job-med", priority=5)

    scheduler = PriorityScheduler(aging_factor=0.0)
    allocations = scheduler.schedule([j_low, j_high, j_med], cluster)

    assert len(allocations) == 1
    # Highest priority job (priority=10) must be scheduled
    assert allocations[0].job_id == j_high.id


def test_priority_scheduler_aging_boost():
    now = datetime.now(timezone.utc)
    old_submitted = now - timedelta(seconds=1000)

    spec = WorkloadSpec(
        name="aging-test",
        resources=ResourceRequirements(gpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )

    job_new = Job(name="new", priority=5, spec=spec, submitted_at=now)
    job_old = Job(name="old", priority=1, spec=spec, submitted_at=old_submitted)

    scheduler = PriorityScheduler(aging_factor=0.01)

    eff_new = scheduler.compute_effective_priority(job_new, now)
    eff_old = scheduler.compute_effective_priority(job_old, now)

    # job_old priority=1 + 0.01 * 1000 = 11.0 > job_new priority=5.0
    assert eff_old > eff_new
