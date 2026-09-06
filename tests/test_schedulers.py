"""Unit tests for DRIGS pluggable schedulers."""

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
from drigs.scheduler import (
    FIFOScheduler,
    FirstFitScheduler,
    BestFitScheduler,
    MemoryAwareScheduler,
)


def _make_spec(gpus: int = 1, gpu_memory_bytes: int = 4 * 1024**3, cpus: int = 2, memory_bytes: int = 8 * 1024**3) -> WorkloadSpec:
    return WorkloadSpec(
        name="test-spec",
        resources=ResourceRequirements(
            gpus=gpus,
            gpu_memory_bytes=gpu_memory_bytes,
            cpus=cpus,
            memory_bytes=memory_bytes,
        ),
        execution=WorkloadExecutionConfig(
            entrypoint="python test.py",
        ),
    )


def _make_worker(worker_id: str, num_gpus: int = 2, vram_per_gpu: int = 16 * 1024**3, cpus: int = 16, status: DeviceState = DeviceState.HEALTHY) -> WorkerInfo:
    devices = [
        ComputeDevice(
            device_id=f"gpu-{i}",
            device_type=DeviceType.GPU,
            model_name="NVIDIA RTX 4090",
            total_memory_bytes=vram_per_gpu,
            available_memory_bytes=vram_per_gpu,
        )
        for i in range(num_gpus)
    ]
    return WorkerInfo(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        ip_address="127.0.0.1",
        devices=devices,
        total_cpus=cpus,
        total_memory_bytes=128 * 1024**3,
        status=status,
    )


def test_scheduler_protocols():
    assert isinstance(FIFOScheduler(), Scheduler)
    assert isinstance(FirstFitScheduler(), Scheduler)
    assert isinstance(BestFitScheduler(), Scheduler)
    assert isinstance(MemoryAwareScheduler(), Scheduler)


def test_fifo_scheduler_execution_order():
    now = datetime.now(timezone.utc)
    job1 = Job(name="job1", spec=_make_spec(gpus=1), submitted_at=now - timedelta(seconds=10))
    job2 = Job(name="job2", spec=_make_spec(gpus=1), submitted_at=now - timedelta(seconds=5))

    worker1 = _make_worker("worker-1", num_gpus=2)
    cluster = ClusterState(workers={"worker-1": worker1})

    scheduler = FIFOScheduler()
    allocations = scheduler.schedule([job2, job1], cluster)

    assert len(allocations) == 2
    assert allocations[0].job_id == job1.id
    assert allocations[1].job_id == job2.id
    assert allocations[0].worker_id == "worker-1"
    assert allocations[1].worker_id == "worker-1"
    assert allocations[0].assigned_device_ids == ["gpu-0"]
    assert allocations[1].assigned_device_ids == ["gpu-1"]


def test_fifo_scheduler_insufficient_resources():
    job1 = Job(name="job1", spec=_make_spec(gpus=4))  # requests 4 GPUs, worker only has 2
    worker1 = _make_worker("worker-1", num_gpus=2)
    cluster = ClusterState(workers={"worker-1": worker1})

    scheduler = FIFOScheduler()
    allocations = scheduler.schedule([job1], cluster)
    assert len(allocations) == 0


def test_first_fit_scheduler_priority_and_placement():
    job_low = Job(name="low", priority=1, spec=_make_spec(gpus=1))
    job_high = Job(name="high", priority=10, spec=_make_spec(gpus=1))

    worker1 = _make_worker("worker-1", num_gpus=1)
    worker2 = _make_worker("worker-2", num_gpus=1)
    cluster = ClusterState(workers={"worker-1": worker1, "worker-2": worker2})

    scheduler = FirstFitScheduler()
    allocations = scheduler.schedule([job_low, job_high], cluster)

    assert len(allocations) == 2
    # High priority job allocated first
    assert allocations[0].job_id == job_high.id
    assert allocations[0].worker_id == "worker-1"
    assert allocations[1].job_id == job_low.id
    assert allocations[1].worker_id == "worker-2"


def test_best_fit_scheduler_surplus_minimization():
    job = Job(name="job", spec=_make_spec(gpus=1, cpus=8))

    # Worker 1: 16 CPUs (surplus 8)
    worker1 = _make_worker("worker-1", num_gpus=1, cpus=16)
    # Worker 2: 10 CPUs (surplus 2) -> tighter fit!
    worker2 = _make_worker("worker-2", num_gpus=1, cpus=10)

    cluster = ClusterState(workers={"worker-1": worker1, "worker-2": worker2})

    scheduler = BestFitScheduler()
    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    assert allocations[0].worker_id == "worker-2"


def test_memory_aware_scheduler_vram_routing():
    # Job requires 12 GB VRAM
    job = Job(name="heavy-job", spec=_make_spec(gpus=1, gpu_memory_bytes=12 * 1024**3))

    # Worker 1 has 4GB VRAM free
    worker1 = _make_worker("worker-1", num_gpus=1, vram_per_gpu=4 * 1024**3)
    # Worker 2 has 16GB VRAM free
    worker2 = _make_worker("worker-2", num_gpus=1, vram_per_gpu=16 * 1024**3)

    cluster = ClusterState(workers={"worker-1": worker1, "worker-2": worker2})

    scheduler = MemoryAwareScheduler()
    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    assert allocations[0].worker_id == "worker-2"
    assert allocations[0].assigned_device_ids == ["gpu-0"]
