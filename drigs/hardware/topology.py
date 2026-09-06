"""Topology Graph & Matrix Representation for Hardware Interconnects."""

from enum import Enum
import itertools
import logging
from typing import Any, Dict, List, Optional, Union

from drigs.core.models import ComputeDevice

logger = logging.getLogger(__name__)


class InterconnectLinkType(str, Enum):
    """Types of interconnect links between compute devices."""

    SELF = "SELF"
    NVLINK = "NVLINK"
    PCIE_SWITCH = "PCIE_SWITCH"
    PCIE = "PCIE"
    NUMA = "NUMA"
    NETWORK = "NETWORK"
    UNKNOWN = "UNKNOWN"


LINK_TYPE_SCORES: Dict[InterconnectLinkType, float] = {
    InterconnectLinkType.SELF: 100.0,
    InterconnectLinkType.NVLINK: 90.0,
    InterconnectLinkType.PCIE_SWITCH: 70.0,
    InterconnectLinkType.PCIE: 50.0,
    InterconnectLinkType.NUMA: 30.0,
    InterconnectLinkType.NETWORK: 10.0,
    InterconnectLinkType.UNKNOWN: 0.0,
}


class TopologyMatrix:
    """Matrix representation of pairwise interconnect links and bandwidth scores."""

    def __init__(self, matrix: Optional[Dict[str, Dict[str, str]]] = None):
        self._matrix: Dict[str, Dict[str, str]] = matrix or {}

    def set_link(self, device_a: str, device_b: str, link_type: Union[InterconnectLinkType, str]) -> None:
        """Set pairwise link type between device_a and device_b."""
        l_str = link_type.value if isinstance(link_type, InterconnectLinkType) else str(link_type)
        if device_a not in self._matrix:
            self._matrix[device_a] = {}
        if device_b not in self._matrix:
            self._matrix[device_b] = {}
        self._matrix[device_a][device_b] = l_str
        self._matrix[device_b][device_a] = l_str

    def get_link_type(self, device_a: str, device_b: str) -> InterconnectLinkType:
        """Get link type between device_a and device_b."""
        if device_a == device_b:
            return InterconnectLinkType.SELF

        raw = self._matrix.get(device_a, {}).get(device_b) or self._matrix.get(device_b, {}).get(device_a)
        if not raw:
            return InterconnectLinkType.UNKNOWN

        try:
            return InterconnectLinkType(raw)
        except ValueError:
            return InterconnectLinkType.UNKNOWN

    def get_bandwidth_score(self, device_a: str, device_b: str) -> float:
        """Get numeric bandwidth connectivity score between device_a and device_b."""
        link_type = self.get_link_type(device_a, device_b)
        return LINK_TYPE_SCORES.get(link_type, 0.0)

    def get_group_topology_score(self, device_ids: List[str]) -> float:
        """Compute the average pairwise topology score for a group of devices."""
        if not device_ids:
            return 0.0
        if len(device_ids) == 1:
            return 100.0

        pairs = list(itertools.combinations(device_ids, 2))
        total_score = sum(self.get_bandwidth_score(a, b) for a, b in pairs)
        return total_score / len(pairs)

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Return raw matrix dictionary representation."""
        return {k: dict(v) for k, v in self._matrix.items()}


class TopologyGraph:
    """Graph structure modeling interconnect topology across devices and nodes."""

    def __init__(
        self,
        devices: Optional[List[ComputeDevice]] = None,
        raw_topology: Optional[Dict[str, Any]] = None,
    ):
        self._devices: Dict[str, ComputeDevice] = {}
        self._matrix = TopologyMatrix()

        if devices:
            for dev in devices:
                self.add_device(dev)

        if raw_topology:
            self.load_raw_topology(raw_topology)

    def add_device(self, device: ComputeDevice) -> None:
        """Add a ComputeDevice to the graph and infer interconnect links with existing devices."""
        dev_id = device.device_id
        self._devices[dev_id] = device
        self._matrix.set_link(dev_id, dev_id, InterconnectLinkType.SELF)

        for other_id, other in self._devices.items():
            if other_id == dev_id:
                continue

            link_type = self._infer_link_type(device, other)
            self._matrix.set_link(dev_id, other_id, link_type)

    def _infer_link_type(self, dev_a: ComputeDevice, dev_b: ComputeDevice) -> InterconnectLinkType:
        """Infer link type between two compute devices based on topology tags, PCIe, and NUMA nodes."""
        nv_a = dev_a.topology_tags.get("nvlink_group")
        nv_b = dev_b.topology_tags.get("nvlink_group")
        if nv_a and nv_b and nv_a == nv_b:
            return InterconnectLinkType.NVLINK

        if dev_a.pcie_bus_id and dev_b.pcie_bus_id:
            prefix_a = dev_a.pcie_bus_id.split(":")[1] if ":" in dev_a.pcie_bus_id else ""
            prefix_b = dev_b.pcie_bus_id.split(":")[1] if ":" in dev_b.pcie_bus_id else ""
            if prefix_a and prefix_a == prefix_b:
                return InterconnectLinkType.PCIE_SWITCH

        if dev_a.numa_node is not None and dev_b.numa_node is not None:
            if dev_a.numa_node == dev_b.numa_node:
                return InterconnectLinkType.PCIE
            else:
                return InterconnectLinkType.NUMA

        return InterconnectLinkType.UNKNOWN

    def load_raw_topology(self, raw_topology: Dict[str, Any]) -> None:
        """Load topology links from backend get_topology() dictionary output."""
        matrix_dict = raw_topology.get("topology_matrix") or raw_topology
        if isinstance(matrix_dict, dict):
            for dev_a, neighbors in matrix_dict.items():
                if isinstance(neighbors, dict):
                    for dev_b, link_str in neighbors.items():
                        self._matrix.set_link(dev_a, dev_b, str(link_str))

    def add_link(self, device_a: str, device_b: str, link_type: Union[InterconnectLinkType, str]) -> None:
        """Explicitly record a link type between two device IDs."""
        self._matrix.set_link(device_a, device_b, link_type)

    def get_matrix(self) -> TopologyMatrix:
        """Get underlying TopologyMatrix."""
        return self._matrix

    def get_pairwise_score(self, device_a: str, device_b: str) -> float:
        """Get interconnect score between device_a and device_b."""
        return self._matrix.get_bandwidth_score(device_a, device_b)

    def find_best_clique(self, device_count: int, available_device_ids: List[str]) -> List[str]:
        """Find subset of available_device_ids of size device_count maximizing group topology score."""
        if device_count <= 0 or device_count > len(available_device_ids):
            return []

        if device_count == 1:
            return [available_device_ids[0]]

        best_combination = None
        best_score = -1.0

        for combination in itertools.combinations(available_device_ids, device_count):
            score = self._matrix.get_group_topology_score(list(combination))
            if score > best_score:
                best_score = score
                best_combination = list(combination)

        return best_combination or []
