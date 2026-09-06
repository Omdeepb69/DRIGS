"""Unit tests for TopologyGraph and TopologyMatrix in drigs.hardware.topology."""

import pytest

from drigs.core.models import ComputeDevice, DeviceType
from drigs.hardware.simulated import SimulatedBackend
from drigs.hardware.topology import (
    InterconnectLinkType,
    TopologyGraph,
    TopologyMatrix,
)


def test_topology_matrix_scores():
    matrix = TopologyMatrix()
    matrix.set_link("dev-0", "dev-1", InterconnectLinkType.NVLINK)
    matrix.set_link("dev-0", "dev-2", InterconnectLinkType.PCIE)

    assert matrix.get_link_type("dev-0", "dev-0") == InterconnectLinkType.SELF
    assert matrix.get_link_type("dev-0", "dev-1") == InterconnectLinkType.NVLINK
    assert matrix.get_link_type("dev-0", "dev-2") == InterconnectLinkType.PCIE
    assert matrix.get_link_type("dev-1", "dev-2") == InterconnectLinkType.UNKNOWN

    assert matrix.get_bandwidth_score("dev-0", "dev-0") == 100.0
    assert matrix.get_bandwidth_score("dev-0", "dev-1") == 90.0
    assert matrix.get_bandwidth_score("dev-0", "dev-2") == 50.0
    assert matrix.get_bandwidth_score("dev-1", "dev-2") == 0.0

    group_score = matrix.get_group_topology_score(["dev-0", "dev-1"])
    assert group_score == 90.0


def test_topology_graph_inference_nvlink():
    backend = SimulatedBackend(num_gpus=4)
    devices = backend.discover_devices()

    graph = TopologyGraph(devices=devices)
    matrix = graph.get_matrix()

    # gpu-0 and gpu-1 are in nvlink_group "group_0"
    assert matrix.get_link_type("gpu-0", "gpu-1") == InterconnectLinkType.NVLINK
    assert graph.get_pairwise_score("gpu-0", "gpu-1") == 90.0

    # gpu-0 and gpu-2 are in different NVLink groups -> PCIE
    assert matrix.get_link_type("gpu-0", "gpu-2") != InterconnectLinkType.NVLINK


def test_topology_graph_inference_pcie_and_numa():
    d0 = ComputeDevice(
        device_id="g0",
        device_type=DeviceType.GPU,
        model_name="V100",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        pcie_bus_id="0000:01:00.0",
        numa_node=0,
    )
    d1 = ComputeDevice(
        device_id="g1",
        device_type=DeviceType.GPU,
        model_name="V100",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        pcie_bus_id="0000:01:00.1",
        numa_node=0,
    )
    d2 = ComputeDevice(
        device_id="g2",
        device_type=DeviceType.GPU,
        model_name="V100",
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        pcie_bus_id="0000:02:00.0",
        numa_node=1,
    )

    graph = TopologyGraph(devices=[d0, d1, d2])
    matrix = graph.get_matrix()

    assert matrix.get_link_type("g0", "g1") == InterconnectLinkType.PCIE_SWITCH
    assert matrix.get_bandwidth_score("g0", "g1") == 70.0

    assert matrix.get_link_type("g0", "g2") == InterconnectLinkType.NUMA
    assert matrix.get_bandwidth_score("g0", "g2") == 30.0


def test_topology_graph_find_best_clique():
    graph = TopologyGraph()
    # Explicitly set links: g0-g1 is NVLINK (90), g2-g3 is NVLINK (90), others are PCIE (50)
    graph.add_link("g0", "g1", InterconnectLinkType.NVLINK)
    graph.add_link("g2", "g3", InterconnectLinkType.NVLINK)
    graph.add_link("g0", "g2", InterconnectLinkType.PCIE)
    graph.add_link("g0", "g3", InterconnectLinkType.PCIE)
    graph.add_link("g1", "g2", InterconnectLinkType.PCIE)
    graph.add_link("g1", "g3", InterconnectLinkType.PCIE)

    best_2 = graph.find_best_clique(2, ["g0", "g1", "g2", "g3"])
    assert best_2 in (["g0", "g1"], ["g2", "g3"])

    # Score of best clique should be 90.0
    score = graph.get_matrix().get_group_topology_score(best_2)
    assert score == 90.0


def test_topology_graph_raw_topology_loading():
    backend = SimulatedBackend(num_gpus=4)
    raw_topo = backend.get_topology()

    graph = TopologyGraph(raw_topology=raw_topo)
    matrix = graph.get_matrix()

    assert matrix.get_link_type("gpu-0", "gpu-1") == InterconnectLinkType.NVLINK
    assert matrix.get_bandwidth_score("gpu-0", "gpu-1") == 90.0
    assert matrix.get_link_type("gpu-0", "gpu-2") == InterconnectLinkType.PCIE
