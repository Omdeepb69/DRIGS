"""Unit tests for dynamic NVML topology matrix discovery in CUDABackend."""

from unittest.mock import MagicMock, patch
import pytest

from drigs.core.models import ComputeDevice, DeviceType
from drigs.hardware.cuda import CUDABackend
from drigs.hardware.topology import InterconnectLinkType, TopologyGraph, TopologyMatrix


def test_cuda_backend_get_topology_fallback():
    """Test get_topology returns valid matrix structure when NVML is unavailable."""
    backend = CUDABackend()
    topo = backend.get_topology()

    assert "device_count" in topo
    assert "devices" in topo
    assert "topology_matrix" in topo
    assert isinstance(topo["topology_matrix"], dict)


def test_nvml_topology_common_ancestor_mapping():
    """Test NVML topology common ancestor constants mapping to InterconnectLinkTypes."""
    backend = CUDABackend()
    backend._nvml_initialized = True

    dev0 = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="NVIDIA A100",
        pcie_bus_id="0000:01:00.0",
        total_memory_bytes=40 * 1024**3,
        available_memory_bytes=40 * 1024**3,
    )
    dev1 = ComputeDevice(
        device_id="gpu-1",
        device_type=DeviceType.GPU,
        model_name="NVIDIA A100",
        pcie_bus_id="0000:02:00.0",
        total_memory_bytes=40 * 1024**3,
        available_memory_bytes=40 * 1024**3,
    )

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 2
    mock_h0 = MagicMock()
    mock_h1 = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: mock_h0 if idx == 0 else mock_h1
    
    # Mock NVLink failure so it falls back to common ancestor
    mock_pynvml.nvmlDeviceGetNvLinkRemotePciInfo.side_effect = Exception("No NVLink")
    
    # Set COMMON ANCESTOR to SINGLE (PCIe switch)
    mock_pynvml.NVML_TOPOLOGY_SINGLE = 1
    mock_pynvml.nvmlDeviceGetTopologyCommonAncestor.return_value = 1

    with patch.object(backend, "discover_devices", return_value=[dev0, dev1]):
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            matrix = backend.get_topology_matrix()
            assert matrix["gpu-0"]["gpu-0"] == "SELF"
            assert matrix["gpu-1"]["gpu-1"] == "SELF"
            assert matrix["gpu-0"]["gpu-1"] == "PCIE_SWITCH"
            assert matrix["gpu-1"]["gpu-0"] == "PCIE_SWITCH"


def test_nvml_topology_nvlink_mapping():
    """Test NVLink remote PCI match mapping to NVLINK in topology matrix."""
    backend = CUDABackend()
    backend._nvml_initialized = True

    dev0 = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="NVIDIA H100",
        pcie_bus_id="0000:01:00.0",
        total_memory_bytes=80 * 1024**3,
        available_memory_bytes=80 * 1024**3,
    )
    dev1 = ComputeDevice(
        device_id="gpu-1",
        device_type=DeviceType.GPU,
        model_name="NVIDIA H100",
        pcie_bus_id="0000:02:00.0",
        total_memory_bytes=80 * 1024**3,
        available_memory_bytes=80 * 1024**3,
    )

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 2
    mock_h0 = MagicMock()
    mock_h1 = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: mock_h0 if idx == 0 else mock_h1
    
    # Mock NVLink remote PCI on link 0 returning dev1's PCI bus ID
    mock_pci_info = MagicMock()
    mock_pci_info.busId = b"0000:02:00.0"
    mock_pynvml.nvmlDeviceGetNvLinkRemotePciInfo.return_value = mock_pci_info
    mock_pynvml.NVML_NVLINK_MAX_LINKS = 6

    with patch.object(backend, "discover_devices", return_value=[dev0, dev1]):
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            matrix = backend.get_topology_matrix()
            assert matrix["gpu-0"]["gpu-1"] == "NVLINK"
            assert matrix["gpu-1"]["gpu-0"] == "NVLINK"


def test_topology_graph_integration():
    """Test integration of CUDABackend topology output with TopologyGraph."""
    backend = CUDABackend()
    
    raw_topo = {
        "device_count": 2,
        "devices": ["gpu-0", "gpu-1"],
        "topology_matrix": {
            "gpu-0": {"gpu-0": "SELF", "gpu-1": "NVLINK"},
            "gpu-1": {"gpu-0": "NVLINK", "gpu-1": "SELF"},
        },
    }

    graph = TopologyGraph(raw_topology=raw_topo)
    matrix = graph.get_matrix()
    
    assert matrix.get_link_type("gpu-0", "gpu-1") == InterconnectLinkType.NVLINK
    assert matrix.get_bandwidth_score("gpu-0", "gpu-1") == 90.0
    assert matrix.get_group_topology_score(["gpu-0", "gpu-1"]) == 90.0
