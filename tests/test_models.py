"""Unit tests for DRIGS core domain models."""

from datetime import datetime
import pytest
from pydantic import ValidationError

from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    ExecutionBackendType,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    ResourceState,
    SchedulingPolicy,
    WorkerInfo,
    WorkloadCheckpointConfig,
    WorkloadDistributionConfig,
    WorkloadExecutionConfig,
    WorkloadRuntimeConfig,
    WorkloadSpec,
)


def test_enum_values():
    """Verify enum str values for system compatibility."""
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.RUNNING == "RUNNING"
    assert DeviceType.GPU == "GPU"
    assert DeviceState.HEALTHY == "HEALTHY"
    assert ResourceState.AVAILABLE == "AVAILABLE"
    assert SchedulingPolicy.MEMORY_AWARE == "MEMORY_AWARE"
    assert ExecutionBackendType.NATIVE == "NATIVE"


def test_compute_device_creation_and_immutability():
    """Test ComputeDevice instantiation and frozen model behavior."""
    device = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        vendor="NVIDIA",
        model_name="NVIDIA RTX 4090",
        total_memory_bytes=24 * 1024 * 1024 * 1024,
        available_memory_bytes=20 * 1024 * 1024 * 1024,
        utilization_pct=15.5,
        temperature_celsius=65.0,
        power_usage_watts=250.0,
        compute_capability="8.9",
        pcie_bus_id="0000:01:00.0",
        numa_node=0,
        topology_tags={"nvlink_group": "group_0"},
    )
    assert device.device_id == "gpu-0"
    assert device.vendor == "NVIDIA"
    assert device.available_memory_bytes == 20 * 1024 * 1024 * 1024

    with pytest.raises(ValidationError):
        # Attempting to mutate a frozen model raises ValidationError
        device.utilization_pct = 50.0  # type: ignore


def test_compute_device_validation_limits():
    """Test validation constraints on ComputeDevice."""
    with pytest.raises(ValidationError):
        # Negative memory
        ComputeDevice(
            device_id="gpu-0",
            device_type=DeviceType.GPU,
            model_name="Test",
            total_memory_bytes=-100,
            available_memory_bytes=0,
        )

    with pytest.raises(ValidationError):
        # Utilization > 100%
        ComputeDevice(
            device_id="gpu-0",
            device_type=DeviceType.GPU,
            model_name="Test",
            total_memory_bytes=1000,
            available_memory_bytes=1000,
            utilization_pct=120.0,
        )


def test_resource_requirements_defaults():
    """Test ResourceRequirements default values and bounds."""
    req = ResourceRequirements()
    assert req.gpus == 0
    assert req.cpus == 1
    assert req.gpu_memory_bytes == 0
    assert not req.topology_aware

    req_custom = ResourceRequirements(
        gpus=2,
        gpu_memory_bytes=16 * 1024 * 1024 * 1024,
        cpus=8,
        memory_bytes=32 * 1024 * 1024 * 1024,
        topology_aware=True,
    )
    assert req_custom.gpus == 2
    assert req_custom.gpu_memory_bytes == 16 * 1024 * 1024 * 1024


def test_workload_spec_serialization():
    """Test WorkloadSpec construction and JSON round-trip."""
    spec = WorkloadSpec(
        name="training-job",
        resources=ResourceRequirements(
            gpus=2,
            gpu_memory_bytes=12 * 1024 * 1024 * 1024,
            cpus=8,
        ),
        execution=WorkloadExecutionConfig(
            backend=ExecutionBackendType.PYTORCH,
            entrypoint="train.py",
            args=["--batch-size", "64"],
            env={"LR": "0.001"},
        ),
        distribution=WorkloadDistributionConfig(
            mode="distributed",
            world_size=2,
        ),
        checkpoint=WorkloadCheckpointConfig(
            enabled=True,
            interval_seconds=600,
            checkpoint_dir="/tmp/checkpoints",
        ),
    )

    json_str = spec.model_dump_json()
    reconstructed = WorkloadSpec.model_validate_json(json_str)

    assert reconstructed.name == "training-job"
    assert reconstructed.resources.gpus == 2
    assert reconstructed.execution.backend == ExecutionBackendType.PYTORCH
    assert reconstructed.execution.args == ["--batch-size", "64"]
    assert reconstructed.checkpoint.enabled is True


def test_job_lifecycle():
    """Test Job state transitions and allocation coupling."""
    spec = WorkloadSpec(
        name="inference-job",
        execution=WorkloadExecutionConfig(
            backend=ExecutionBackendType.NATIVE,
            entrypoint="serve.py",
        ),
    )
    job = Job(name="inference-job", spec=spec, priority=10)
    assert job.status == JobStatus.PENDING
    assert job.job_id is not None
    assert job.priority == 10
    assert job.started_at is None

    # Transition to RUNNING
    allocation = ResourceAllocation(
        job_id=job.job_id,
        worker_id="node-1",
        assigned_device_ids=["gpu-0"],
        assigned_cpu_cores=[0, 1, 2, 3],
        memory_bytes=8 * 1024 * 1024 * 1024,
    )
    job.allocation = allocation
    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()

    assert job.status == JobStatus.RUNNING
    assert job.allocation.worker_id == "node-1"
    assert job.allocation.assigned_device_ids == ["gpu-0"]


def test_worker_info_and_cluster_state():
    """Test WorkerInfo registration and ClusterState aggregation."""
    device = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="RTX 3090",
        total_memory_bytes=24 * 1024 * 1024 * 1024,
        available_memory_bytes=24 * 1024 * 1024 * 1024,
    )

    worker = WorkerInfo(
        worker_id="worker-01",
        hostname="gpu-node-01",
        ip_address="192.168.1.100",
        devices=[device],
        total_cpus=16,
        total_memory_bytes=64 * 1024 * 1024 * 1024,
    )

    cluster = ClusterState(workers={worker.worker_id: worker})
    assert "worker-01" in cluster.workers
    assert cluster.workers["worker-01"].total_cpus == 16
    assert len(cluster.workers["worker-01"].devices) == 1
