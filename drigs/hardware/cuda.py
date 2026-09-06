"""NVIDIA CUDA hardware discovery backend for DRIGS."""

import logging
from typing import Any, Dict, List
from drigs.core.models import ComputeDevice, DeviceState, DeviceStatus, DeviceType

logger = logging.getLogger(__name__)


class CUDABackend:
    """NVIDIA CUDA & NVML GPU discovery backend."""

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

    def get_topology(self) -> Dict[str, Any]:
        """Fetch GPU NVLink/PCIe topology graph."""
        devices = self.discover_devices()
        return {"device_count": len(devices), "devices": [d.device_id for d in devices]}
