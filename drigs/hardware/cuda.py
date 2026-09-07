"""NVIDIA CUDA hardware discovery backend for DRIGS."""

import logging
from typing import Any, Dict, List
from drigs.core.models import ComputeDevice, DeviceState, DeviceStatus, DeviceType

logger = logging.getLogger(__name__)


class CUDABackend:
    """NVIDIA CUDA & NVML GPU discovery and topology mapping backend."""

    def __init__(self):
        self._nvml_initialized = False
        self._try_init_nvml()

    def _try_init_nvml(self) -> bool:
        """Attempt to initialize pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except Exception as e:
            logger.debug(f"pynvml initialization skipped/failed: {e}")
            self._nvml_initialized = False
            return False

    def discover_devices(self) -> List[ComputeDevice]:
        """Discover physical NVIDIA GPUs using NVML or PyTorch CUDA API."""
        devices: List[ComputeDevice] = []

        if self._nvml_initialized:
            try:
                import pynvml
                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                    bus_id = pci_info.busId
                    if isinstance(bus_id, bytes):
                        bus_id = bus_id.decode("utf-8")

                    device = ComputeDevice(
                        device_id=f"gpu-{i}",
                        device_type=DeviceType.GPU,
                        vendor="NVIDIA",
                        model_name=name,
                        total_memory_bytes=mem_info.total,
                        available_memory_bytes=mem_info.free,
                        utilization_pct=float(util.gpu),
                        temperature_celsius=float(temp),
                        power_usage_watts=float(power),
                        pcie_bus_id=bus_id,
                    )
                    devices.append(device)
                return devices
            except Exception as e:
                logger.warning(f"Error querying NVML devices: {e}")

        # Fallback to PyTorch CUDA API if pynvml unavailable
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                for i in range(count):
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory
                    free_mem, _ = torch.cuda.mem_get_info(i) if hasattr(torch.cuda, "mem_get_info") else (total_mem, total_mem)
                    capability = f"{props.major}.{props.minor}"

                    device = ComputeDevice(
                        device_id=f"gpu-{i}",
                        device_type=DeviceType.GPU,
                        vendor="NVIDIA",
                        model_name=props.name,
                        total_memory_bytes=total_mem,
                        available_memory_bytes=free_mem,
                        compute_capability=capability,
                    )
                    devices.append(device)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error querying PyTorch CUDA devices: {e}")

        return devices

    def get_device_status(self, device_id: str) -> DeviceStatus:
        """Fetch status for a specific GPU device."""
        devices = self.discover_devices()
        for dev in devices:
            if dev.device_id == device_id:
                used_mem = dev.total_memory_bytes - dev.available_memory_bytes
                return DeviceStatus(
                    device_id=device_id,
                    state=DeviceState.HEALTHY,
                    utilization_pct=dev.utilization_pct,
                    memory_used_bytes=used_mem,
                    memory_free_bytes=dev.available_memory_bytes,
                    temperature_celsius=dev.temperature_celsius,
                )
        return DeviceStatus(device_id=device_id, state=DeviceState.UNAVAILABLE)

    def get_topology_matrix(self) -> Dict[str, Dict[str, str]]:
        """Construct pairwise interconnect link type matrix via NVML or PCIe/NUMA fallback."""
        devices = self.discover_devices()
        matrix: Dict[str, Dict[str, str]] = {}
        dev_ids = [d.device_id for d in devices]

        for d_id in dev_ids:
            matrix[d_id] = {d_id: "SELF"}

        if self._nvml_initialized and len(devices) > 1:
            try:
                import pynvml
                count = pynvml.nvmlDeviceGetCount()
                handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

                def get_nvlink_remote_pci(handle, link_idx):
                    try:
                        pci_info = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link_idx)
                        bus = pci_info.busId
                        return bus.decode("utf-8") if isinstance(bus, bytes) else bus
                    except Exception:
                        return None

                pci_to_dev = {}
                for dev in devices:
                    if dev.pcie_bus_id:
                        pci_to_dev[dev.pcie_bus_id.lower()] = dev.device_id

                for i in range(min(count, len(devices))):
                    dev_i = devices[i]
                    handle_i = handles[i]

                    nvlink_connected_devs = set()
                    try:
                        max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 6)
                        for link in range(max_links):
                            remote_pci = get_nvlink_remote_pci(handle_i, link)
                            if remote_pci and remote_pci.lower() in pci_to_dev:
                                nvlink_connected_devs.add(pci_to_dev[remote_pci.lower()])
                    except Exception:
                        pass

                    for j in range(i + 1, min(count, len(devices))):
                        dev_j = devices[j]
                        handle_j = handles[j]
                        link_type = "UNKNOWN"

                        if dev_j.device_id in nvlink_connected_devs:
                            link_type = "NVLINK"
                        else:
                            try:
                                ancestor = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                                nvlink_val = getattr(pynvml, "NVML_TOPOLOGY_NVLINK", -1)
                                single_val = getattr(pynvml, "NVML_TOPOLOGY_SINGLE", 1)
                                multiple_val = getattr(pynvml, "NVML_TOPOLOGY_MULTIPLE", 2)
                                host_val = getattr(pynvml, "NVML_TOPOLOGY_HOSTBRIDGE", 3)
                                node_val = getattr(pynvml, "NVML_TOPOLOGY_NODE", 4)
                                sys_val = getattr(pynvml, "NVML_TOPOLOGY_SYSTEM", 5)

                                if ancestor == nvlink_val:
                                    link_type = "NVLINK"
                                elif ancestor == single_val:
                                    link_type = "PCIE_SWITCH"
                                elif ancestor in (multiple_val, host_val):
                                    link_type = "PCIE"
                                elif ancestor in (node_val, sys_val):
                                    link_type = "NUMA"
                                else:
                                    link_type = "PCIE"
                            except Exception:
                                link_type = self._fallback_infer_link(dev_i, dev_j)

                        matrix[dev_i.device_id][dev_j.device_id] = link_type
                        matrix[dev_j.device_id][dev_i.device_id] = link_type
                return matrix
            except Exception as e:
                logger.warning(f"Error querying NVML topology common ancestor: {e}")

        # Fallback topology inference when NVML is not available
        for i in range(len(devices)):
            dev_i = devices[i]
            for j in range(i + 1, len(devices)):
                dev_j = devices[j]
                link_type = self._fallback_infer_link(dev_i, dev_j)
                matrix[dev_i.device_id][dev_j.device_id] = link_type
                matrix[dev_j.device_id][dev_i.device_id] = link_type

        return matrix

    def _fallback_infer_link(self, dev_a: ComputeDevice, dev_b: ComputeDevice) -> str:
        """Infer link type between devices when NVML queries are unavailable."""
        if dev_a.pcie_bus_id and dev_b.pcie_bus_id:
            prefix_a = dev_a.pcie_bus_id.split(":")[1] if ":" in dev_a.pcie_bus_id else ""
            prefix_b = dev_b.pcie_bus_id.split(":")[1] if ":" in dev_b.pcie_bus_id else ""
            if prefix_a and prefix_a == prefix_b:
                return "PCIE_SWITCH"
        if dev_a.numa_node is not None and dev_b.numa_node is not None:
            if dev_a.numa_node == dev_b.numa_node:
                return "PCIE"
            return "NUMA"
        return "PCIE"

    def get_topology(self) -> Dict[str, Any]:
        """Fetch GPU NVLink/PCIe topology graph and matrix."""
        devices = self.discover_devices()
        matrix = self.get_topology_matrix()
        return {
            "device_count": len(devices),
            "devices": [d.device_id for d in devices],
            "topology_matrix": matrix,
        }
