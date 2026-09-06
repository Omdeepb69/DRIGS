"""CPU & Host RAM hardware discovery backend for DRIGS."""

import os
from typing import Any, Dict, List
import psutil

from drigs.core.models import ComputeDevice, DeviceState, DeviceStatus, DeviceType


class CPUBackend:
    """Discovers host CPU cores, memory capacity, and load status."""

    def __init__(self, device_id: str = "cpu-0"):
        self.device_id = device_id

    def discover_devices(self) -> List[ComputeDevice]:
        """Discover CPU as a compute device."""
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        cpu_count = psutil.cpu_count(logical=True) or 1
        cpu_percent = psutil.cpu_percent(interval=None)

        device = ComputeDevice(
            device_id=self.device_id,
            device_type=DeviceType.CPU,
            vendor="Generic CPU",
            model_name=f"Host CPU ({cpu_count} cores)",
            total_memory_bytes=total_memory,
            available_memory_bytes=available_memory,
            utilization_pct=float(cpu_percent),
            topology_tags={"cores": str(cpu_count)},
        )
        return [device]

    def get_device_status(self, device_id: str) -> DeviceStatus:
        """Get live status snippet for the CPU."""
        vm = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        return DeviceStatus(
            device_id=device_id,
            state=DeviceState.HEALTHY,
            utilization_pct=float(cpu_percent),
            memory_used_bytes=vm.used,
            memory_free_bytes=vm.available,
        )

    def get_topology(self) -> Dict[str, Any]:
        """Return host CPU and memory topology information."""
        return {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_ram_bytes": psutil.virtual_memory().total,
        }
