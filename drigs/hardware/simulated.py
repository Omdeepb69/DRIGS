"""Simulated GPU & hardware discovery backend for DRIGS testing & research."""

from typing import Any, Dict, List, Optional
from drigs.core.models import ComputeDevice, DeviceState, DeviceStatus, DeviceType


class SimulatedBackend:
    """Configurable synthetic GPU cluster discovery backend."""

    def __init__(
        self,
        num_gpus: int = 4,
        vram_per_gpu_bytes: int = 24 * 1024 * 1024 * 1024,
        model_name: str = "NVIDIA RTX 4090 (Simulated)",
        vendor: str = "NVIDIA",
        compute_capability: str = "8.9",
    ):
        self.num_gpus = num_gpus
        self.vram_per_gpu_bytes = vram_per_gpu_bytes
        self.model_name = model_name
        self.vendor = vendor
        self.compute_capability = compute_capability
        self._devices: Dict[str, ComputeDevice] = {}
        self._initialize_simulated_devices()

    def _initialize_simulated_devices(self) -> None:
        """Construct synthetic GPU devices."""
        for i in range(self.num_gpus):
            dev_id = f"gpu-{i}"
            device = ComputeDevice(
                device_id=dev_id,
                device_type=DeviceType.GPU,
                vendor=self.vendor,
                model_name=self.model_name,
                total_memory_bytes=self.vram_per_gpu_bytes,
                available_memory_bytes=self.vram_per_gpu_bytes,
                utilization_pct=0.0,
                temperature_celsius=35.0,
                power_usage_watts=50.0,
                compute_capability=self.compute_capability,
                pcie_bus_id=f"0000:0{i+1}:00.0",
                numa_node=i % 2,
                topology_tags={"nvlink_group": f"group_{i // 2}"},
            )
            self._devices[dev_id] = device

    def update_simulated_state(
        self,
        device_id: str,
        used_memory_bytes: int,
        utilization_pct: float,
        temperature_celsius: Optional[float] = None,
    ) -> None:
        """Simulate real-time resource state mutations."""
        if device_id in self._devices:
            dev = self._devices[device_id]
            available = max(0, dev.total_memory_bytes - used_memory_bytes)
            self._devices[device_id] = ComputeDevice(
                device_id=dev.device_id,
                device_type=dev.device_type,
                vendor=dev.vendor,
                model_name=dev.model_name,
                total_memory_bytes=dev.total_memory_bytes,
                available_memory_bytes=available,
                utilization_pct=max(0.0, min(100.0, utilization_pct)),
                temperature_celsius=temperature_celsius or dev.temperature_celsius,
                power_usage_watts=dev.power_usage_watts,
                compute_capability=dev.compute_capability,
                pcie_bus_id=dev.pcie_bus_id,
                numa_node=dev.numa_node,
                topology_tags=dev.topology_tags,
            )

    def discover_devices(self) -> List[ComputeDevice]:
        """Return list of simulated compute devices."""
        return list(self._devices.values())

    def get_device_status(self, device_id: str) -> DeviceStatus:
        """Get device status snippet."""
        if device_id not in self._devices:
            return DeviceStatus(device_id=device_id, state=DeviceState.UNAVAILABLE)

        dev = self._devices[device_id]
        used_mem = dev.total_memory_bytes - dev.available_memory_bytes
        return DeviceStatus(
            device_id=device_id,
            state=DeviceState.HEALTHY,
            utilization_pct=dev.utilization_pct,
            memory_used_bytes=used_mem,
            memory_free_bytes=dev.available_memory_bytes,
            temperature_celsius=dev.temperature_celsius,
        )

    def get_topology(self) -> Dict[str, Any]:
        """Return synthetic interconnect topology matrix."""
        matrix: Dict[str, Dict[str, str]] = {}
        device_ids = list(self._devices.keys())
        for dev_a in device_ids:
            matrix[dev_a] = {}
            for dev_b in device_ids:
                if dev_a == dev_b:
                    matrix[dev_a][dev_b] = "SELF"
                elif self._devices[dev_a].topology_tags.get("nvlink_group") == self._devices[dev_b].topology_tags.get("nvlink_group"):
                    matrix[dev_a][dev_b] = "NVLINK"
                else:
                    matrix[dev_a][dev_b] = "PCIE"
        return {"topology_matrix": matrix}
