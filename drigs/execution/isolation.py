"""Multi-GPU Isolation Manager for DRIGS."""

import logging
import os
import re
from threading import RLock
from typing import Dict, List, Optional, Set

from drigs.core.models import ComputeDevice, ResourceAllocation

logger = logging.getLogger(__name__)


class IsolationError(Exception):
    """Exception raised when GPU isolation or reservation fails."""

    pass


def clean_gpu_id(dev_id: str) -> str:
    """Clean GPU device string to numeric index or canonical format."""
    val = str(dev_id).strip()
    if val.lower().startswith("gpu-"):
        val = val[4:]
    elif val.lower().startswith("gpu:"):
        val = val[4:]
    return val


def format_cuda_visible_devices(
    device_ids: List[str],
    use_pcie_bus_id: bool = False,
    devices_map: Optional[Dict[str, ComputeDevice]] = None,
) -> str:
    """Format CUDA_VISIBLE_DEVICES environment variable string from a list of GPU device IDs."""
    if not device_ids:
        return ""

    formatted_parts: List[str] = []

    for dev_id in device_ids:
        device = devices_map.get(dev_id) if devices_map else None

        if use_pcie_bus_id and device and device.pcie_bus_id:
            formatted_parts.append(device.pcie_bus_id)
        elif use_pcie_bus_id and re.match(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$", dev_id):
            formatted_parts.append(dev_id)
        else:
            formatted_parts.append(clean_gpu_id(dev_id))

    return ",".join(formatted_parts)


def build_isolated_environment(
    allocation: ResourceAllocation,
    base_env: Optional[Dict[str, str]] = None,
    devices_map: Optional[Dict[str, ComputeDevice]] = None,
    use_pcie_bus_id: bool = False,
) -> Dict[str, str]:
    """Construct an isolated process environment dictionary for a resource allocation."""
    env = dict(os.environ)
    if base_env:
        env.update(base_env)

    env["DRIGS_JOB_ID"] = allocation.job_id
    env["DRIGS_ALLOCATION_ID"] = allocation.allocation_id
    env["DRIGS_WORKER_ID"] = allocation.worker_id

    gpu_ids = allocation.assigned_device_ids
    env["CUDA_VISIBLE_DEVICES"] = format_cuda_visible_devices(
        gpu_ids,
        use_pcie_bus_id=use_pcie_bus_id,
        devices_map=devices_map,
    )

    if allocation.assigned_cpu_cores:
        core_strs = [str(c) for c in allocation.assigned_cpu_cores]
        env["DRIGS_CPU_CORES"] = ",".join(core_strs)
        env["OMP_NUM_THREADS"] = str(len(allocation.assigned_cpu_cores))
        env["MKL_NUM_THREADS"] = str(len(allocation.assigned_cpu_cores))

    return env


class GPUIsolationManager:
    """Thread-safe GPU device reservation and isolation manager for a worker node."""

    def __init__(self, use_pcie_bus_id: bool = False):
        self._lock = RLock()
        self.use_pcie_bus_id = use_pcie_bus_id
        self._reserved_gpus: Dict[str, List[str]] = {}  # job_id -> List[device_id]
        self._gpu_owners: Dict[str, str] = {}  # device_id -> job_id
        self._devices_map: Dict[str, ComputeDevice] = {}  # device_id -> ComputeDevice

    def register_devices(self, devices: List[ComputeDevice]) -> None:
        """Register compute devices for metadata lookup."""
        with self._lock:
            for dev in devices:
                self._devices_map[dev.device_id] = dev

    def reserve_gpus(self, job_id: str, device_ids: List[str]) -> str:
        """Reserve GPUs for a job and return formatted CUDA_VISIBLE_DEVICES string."""
        with self._lock:
            if job_id in self._reserved_gpus:
                raise IsolationError(f"Job {job_id} already has GPU reservations")

            # Check for overlaps
            for dev_id in device_ids:
                if dev_id in self._gpu_owners:
                    existing_owner = self._gpu_owners[dev_id]
                    raise IsolationError(
                        f"GPU device {dev_id} is already reserved by job {existing_owner}"
                    )

            # Record reservations
            self._reserved_gpus[job_id] = list(device_ids)
            for dev_id in device_ids:
                self._gpu_owners[dev_id] = job_id

            return format_cuda_visible_devices(
                device_ids,
                use_pcie_bus_id=self.use_pcie_bus_id,
                devices_map=self._devices_map,
            )

    def release_gpus(self, job_id: str) -> List[str]:
        """Release reserved GPUs for a job and return the list of released device IDs."""
        with self._lock:
            released = self._reserved_gpus.pop(job_id, [])
            for dev_id in released:
                self._gpu_owners.pop(dev_id, None)
            return released

    def is_gpu_reserved(self, device_id: str) -> bool:
        """Check if a specific GPU device ID is currently reserved."""
        with self._lock:
            return device_id in self._gpu_owners

    def get_owner(self, device_id: str) -> Optional[str]:
        """Get job_id owning a reserved GPU device."""
        with self._lock:
            return self._gpu_owners.get(device_id)

    def get_active_reservations(self) -> Dict[str, List[str]]:
        """Return snapshot of active job_id -> device_ids reservations."""
        with self._lock:
            return {k: list(v) for k, v in self._reserved_gpus.items()}
