"""Unit tests for DRIGS ResourceManager."""

import pytest
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceStatus,
    DeviceType,
    WorkerInfo,
)
from drigs.core.resource_manager import ResourceError, ResourceManager


def create_mock_worker(worker_id: str = "worker-0", num_gpus: int = 2) -> WorkerInfo:
    """Helper to construct a mock WorkerInfo with GPUs."""
    devices = [
        ComputeDevice(
            device_id=f"gpu-{i}",
            device_type=DeviceType.GPU,
            model_name="RTX 4090",
            total_memory_bytes=24 * 1024 * 1024 * 1024,
            available_memory_bytes=24 * 1024 * 1024 * 1024,
        )
        for i in range(num_gpus)
    ]
    return WorkerInfo(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        ip_address="127.0.0.1",
        devices=devices,
        total_cpus=16,
        total_memory_bytes=64 * 1024 * 1024 * 1024,
    )


def test_resource_manager_registration():
    """Test worker registration, availability listing, and unregistration."""
    rm = ResourceManager()
    worker = create_mock_worker("w-1", num_gpus=2)

    rm.register_worker(worker)
    available = rm.get_available_devices("w-1")
    assert len(available) == 2

    state = rm.get_cluster_state()
    assert "w-1" in state.workers

    rm.unregister_worker("w-1")
    available_after = rm.get_available_devices("w-1")
    assert len(available_after) == 0


def test_resource_allocation_and_deallocation():
    """Test successful resource allocation and deallocation lifecycle."""
    rm = ResourceManager()
    worker = create_mock_worker("w-1", num_gpus=2)
    rm.register_worker(worker)

    alloc = rm.allocate_resources(
        job_id="job-100",
        worker_id="w-1",
        device_ids=["gpu-0"],
        cpus=4,
        memory_bytes=8 * 1024 * 1024 * 1024,
        gpu_memory_bytes=12 * 1024 * 1024 * 1024,
    )

    assert alloc.job_id == "job-100"
    assert alloc.assigned_device_ids == ["gpu-0"]

    # gpu-0 should no longer be listed as available
    available = rm.get_available_devices("w-1")
    assert len(available) == 1
    assert available[0].device_id == "gpu-1"

    # Attempting to allocate gpu-0 again should fail
    with pytest.raises(ResourceError):
        rm.allocate_resources(
            job_id="job-101",
            worker_id="w-1",
            device_ids=["gpu-0"],
            cpus=2,
            memory_bytes=4 * 1024 * 1024 * 1024,
        )

    # Deallocate job-100
    dealloc_success = rm.deallocate_resources("job-100")
    assert dealloc_success is True

    # gpu-0 should be available again
    available_restored = rm.get_available_devices("w-1")
    assert len(available_restored) == 2


def test_device_status_update_degradation():
    """Test device status updates marking devices unavailable."""
    rm = ResourceManager()
    worker = create_mock_worker("w-1", num_gpus=2)
    rm.register_worker(worker)

    rm.update_device_status("w-1", DeviceStatus(device_id="gpu-1", state=DeviceState.OFFLINE))

    available = rm.get_available_devices("w-1")
    assert len(available) == 1
    assert available[0].device_id == "gpu-0"
