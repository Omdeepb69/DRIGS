"""Unit tests for GangScheduler in drigs.scheduler.gang."""

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
from drigs.scheduler.gang import GangScheduler


def _make_spec(gpus: int = 4, cpus: int = 4) -> WorkloadSpec:
    return WorkloadSpec(
        name="gang-spec",
        resources=ResourceRequirements(
            gpus=gpus,
            gpu_memory_bytes=4 * 1024**3,
            cpus=cpus,
            memory_bytes=8 * 1024**3,
        ),
        execution=WorkloadExecutionConfig(entrypoint="python train.py"),
    )


def _make_worker(worker_id: str, num_gpus: int = 2) -> WorkerInfo:
    devices = [
        ComputeDevice(
            device_id=f"gpu-{worker_id}-{i}",
            device_type=DeviceType.GPU,
            model_name="NVIDIA V100",
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=16 * 1024**3,
        )
        for i in range(num_gpus)
    ]
    return WorkerInfo(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        ip_address="127.0.0.1",
        devices=devices,
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )


def test_gang_scheduler_protocol():
    assert isinstance(GangScheduler(), Scheduler)


def test_gang_scheduler_all_or_nothing_insufficient():
    # Worker 1 has 2 GPUs, Worker 2 has 1 GPU. Total = 3 GPUs.
    w1 = _make_worker("worker-1", num_gpus=2)
    w2 = _make_worker("worker-2", num_gpus=1)
    cluster = ClusterState(workers={"worker-1": w1, "worker-2": w2})

    job = Job(name="heavy-gang", spec=_make_spec(gpus=4))
    scheduler = GangScheduler()

    allocations = scheduler.schedule([job], cluster)

    # All-or-Nothing: Cannot satisfy 4 GPUs, so 0 allocations produced
    assert len(allocations) == 0


def test_gang_scheduler_single_node_satisfaction():
    w1 = _make_worker("worker-1", num_gpus=4)
    cluster = ClusterState(workers={"worker-1": w1})

    job = Job(name="single-node-gang", spec=_make_spec(gpus=4))
    scheduler = GangScheduler()

    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    assert allocations[0].worker_id == "worker-1"
    assert len(allocations[0].assigned_device_ids) == 4


def test_gang_scheduler_multi_node_satisfaction():
    # Worker 1 has 2 GPUs, Worker 2 has 2 GPUs. Total = 4 GPUs.
    w1 = _make_worker("worker-1", num_gpus=2)
    w2 = _make_worker("worker-2", num_gpus=2)
    cluster = ClusterState(workers={"worker-1": w1, "worker-2": w2})

    job = Job(name="multi-node-gang", spec=_make_spec(gpus=4))
    scheduler = GangScheduler()

    allocations = scheduler.schedule([job], cluster)

    # Multi-node gang allocation across 2 workers
    assert len(allocations) == 2
    allocated_worker_ids = {a.worker_id for a in allocations}
    assert allocated_worker_ids == {"worker-1", "worker-2"}

    total_gpus = sum(len(a.assigned_device_ids) for a in allocations)
    assert total_gpus == 4
