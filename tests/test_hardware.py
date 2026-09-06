"""Unit tests for DRIGS hardware discovery backends."""

import pytest
from drigs.core.interfaces import HardwareBackend
from drigs.hardware.cpu import CPUBackend
from drigs.hardware.cuda import CUDABackend
from drigs.hardware.simulated import SimulatedBackend


def test_cpu_backend():
    """Test CPUBackend hardware discovery and protocol compliance."""
    backend = CPUBackend()
    assert isinstance(backend, HardwareBackend)

    devices = backend.discover_devices()
    assert len(devices) == 1
    cpu_dev = devices[0]
    assert cpu_dev.device_id == "cpu-0"
    assert cpu_dev.total_memory_bytes > 0

    status = backend.get_device_status("cpu-0")
    assert status.device_id == "cpu-0"

    topology = backend.get_topology()
    assert "logical_cores" in topology


def test_simulated_backend():
    """Test SimulatedBackend device generation, state updates, and topology matrix."""
    backend = SimulatedBackend(num_gpus=4, model_name="Simulated H100")
    assert isinstance(backend, HardwareBackend)

    devices = backend.discover_devices()
    assert len(devices) == 4
    assert devices[0].model_name == "Simulated H100"
    assert devices[0].available_memory_bytes == 24 * 1024 * 1024 * 1024

    # Update simulated VRAM usage and utilization
    backend.update_simulated_state(
        device_id="gpu-0",
        used_memory_bytes=8 * 1024 * 1024 * 1024,
        utilization_pct=85.0,
        temperature_celsius=72.0,
    )

    status = backend.get_device_status("gpu-0")
    assert status.utilization_pct == 85.0
    assert status.memory_free_bytes == 16 * 1024 * 1024 * 1024
    assert status.temperature_celsius == 72.0

    topology = backend.get_topology()
    assert "topology_matrix" in topology
    assert topology["topology_matrix"]["gpu-0"]["gpu-0"] == "SELF"
    assert topology["topology_matrix"]["gpu-0"]["gpu-1"] == "NVLINK"


def test_cuda_backend_safe_fallback():
    """Test CUDABackend executes safely without crashing on non-GPU environment."""
    backend = CUDABackend()
    assert isinstance(backend, HardwareBackend)

    devices = backend.discover_devices()
    assert isinstance(devices, list)

    status = backend.get_device_status("gpu-0")
    assert status is not None
