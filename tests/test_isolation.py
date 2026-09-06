"""Unit tests for GPUIsolationManager in drigs.execution.isolation."""

import pytest

from drigs.core.models import ComputeDevice, DeviceType, ResourceAllocation
from drigs.execution.isolation import (
    GPUIsolationManager,
    IsolationError,
    build_isolated_environment,
    clean_gpu_id,
    format_cuda_visible_devices,
)


def test_clean_gpu_id():
    assert clean_gpu_id("gpu-0") == "0"
    assert clean_gpu_id("GPU-1") == "1"
    assert clean_gpu_id("2") == "2"


def test_format_cuda_visible_devices():
    assert format_cuda_visible_devices([]) == ""
    assert format_cuda_visible_devices(["gpu-0", "gpu-1"]) == "0,1"
    assert format_cuda_visible_devices(["GPU-2"]) == "2"


def test_format_cuda_visible_devices_pcie():
    dev0 = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="NVIDIA A100",
        total_memory_bytes=80 * 1024**3,
        available_memory_bytes=80 * 1024**3,
        pcie_bus_id="0000:01:00.0",
    )
    dev1 = ComputeDevice(
        device_id="gpu-1",
        device_type=DeviceType.GPU,
        model_name="NVIDIA A100",
        total_memory_bytes=80 * 1024**3,
        available_memory_bytes=80 * 1024**3,
        pcie_bus_id="0000:02:00.0",
    )
    devices_map = {"gpu-0": dev0, "gpu-1": dev1}

    result = format_cuda_visible_devices(
        ["gpu-0", "gpu-1"], use_pcie_bus_id=True, devices_map=devices_map
    )
    assert result == "0000:01:00.0,0000:02:00.0"


def test_build_isolated_environment():
    alloc = ResourceAllocation(
        job_id="job-123",
        worker_id="worker-0",
        assigned_device_ids=["gpu-0", "gpu-1"],
        assigned_cpu_cores=[0, 1, 2, 3],
        memory_bytes=16 * 1024**3,
    )
    env = build_isolated_environment(alloc)

    assert env["DRIGS_JOB_ID"] == "job-123"
    assert env["DRIGS_ALLOCATION_ID"] == alloc.allocation_id
    assert env["DRIGS_WORKER_ID"] == "worker-0"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["DRIGS_CPU_CORES"] == "0,1,2,3"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"


def test_gpu_isolation_manager_reservations():
    mgr = GPUIsolationManager()

    dev0 = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="NVIDIA RTX 3090",
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=24 * 1024**3,
    )
    dev1 = ComputeDevice(
        device_id="gpu-1",
        device_type=DeviceType.GPU,
        model_name="NVIDIA RTX 3090",
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=24 * 1024**3,
    )
    mgr.register_devices([dev0, dev1])

    # Reserve GPU 0 for job 1
    cuda_str1 = mgr.reserve_gpus("job-1", ["gpu-0"])
    assert cuda_str1 == "0"
    assert mgr.is_gpu_reserved("gpu-0") is True
    assert mgr.get_owner("gpu-0") == "job-1"

    # Attempting to reserve GPU 0 again for job 2 should fail
    with pytest.raises(IsolationError, match="already reserved by job job-1"):
        mgr.reserve_gpus("job-2", ["gpu-0"])

    # Reserve GPU 1 for job 2
    cuda_str2 = mgr.reserve_gpus("job-2", ["gpu-1"])
    assert cuda_str2 == "1"

    active = mgr.get_active_reservations()
    assert active == {"job-1": ["gpu-0"], "job-2": ["gpu-1"]}

    # Release job 1
    released = mgr.release_gpus("job-1")
    assert released == ["gpu-0"]
    assert mgr.is_gpu_reserved("gpu-0") is False

    # Now job 3 can reserve GPU 0
    cuda_str3 = mgr.reserve_gpus("job-3", ["gpu-0"])
    assert cuda_str3 == "0"
