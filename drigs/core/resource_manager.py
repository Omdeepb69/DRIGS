"""Resource State & Allocation Manager for DRIGS."""

import logging
from threading import RLock
from typing import Any, Dict, List, Optional, Set
from pydantic import ValidationError

from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceStatus,
    ResourceAllocation,
    ResourceState,
    WorkerInfo,
)

logger = logging.getLogger(__name__)


class ResourceError(Exception):
    """Exception raised when resource allocation or validation fails."""

    pass


class ResourceManager:
    """Thread-safe resource state and allocation manager."""

    def __init__(self, storage_backend: Optional[Any] = None):
        self._lock = RLock()
        self.storage_backend = storage_backend
        self._workers: Dict[str, WorkerInfo] = {}
        self._device_states: Dict[str, ResourceState] = {}  # key: f"{worker_id}:{device_id}"
        self._device_allocated_memory: Dict[str, int] = {}  # key: f"{worker_id}:{device_id}"
        self._allocations: Dict[str, ResourceAllocation] = {}  # allocation_id -> ResourceAllocation
        self._job_allocations: Dict[str, str] = {}  # job_id -> allocation_id

    def load_from_storage(self) -> int:
        """Load and restore workers and resource allocations from storage backend."""
        with self._lock:
            if not self.storage_backend:
                return 0

            stored_workers = self.storage_backend.load_all_workers()
            for worker in stored_workers:
                self.register_worker(worker)

            stored_allocs = self.storage_backend.load_all_allocations()
            for alloc in stored_allocs:
                self._allocations[alloc.allocation_id] = alloc
                self._job_allocations[alloc.job_id] = alloc.allocation_id
                for dev_id in alloc.assigned_device_ids:
                    key = f"{alloc.worker_id}:{dev_id}"
                    self._device_states[key] = ResourceState.ALLOCATED

            return len(stored_allocs)

    def register_worker(self, worker: WorkerInfo) -> None:
        """Register or update a worker node and its compute devices."""
        with self._lock:
            self._workers[worker.worker_id] = worker
            for device in worker.devices:
                key = f"{worker.worker_id}:{device.device_id}"
                if key not in self._device_states:
                    self._device_states[key] = ResourceState.AVAILABLE
                    self._device_allocated_memory[key] = 0
            if self.storage_backend:
                self.storage_backend.save_worker(worker)

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker node and mark its resources unavailable."""
        with self._lock:
            if worker_id in self._workers:
                worker = self._workers.pop(worker_id)
                for device in worker.devices:
                    key = f"{worker_id}:{device.device_id}"
                    self._device_states[key] = ResourceState.UNAVAILABLE
                if self.storage_backend:
                    self.storage_backend.delete_worker(worker_id)

    def update_device_status(self, worker_id: str, status: DeviceStatus) -> None:
        """Update live status for a specific compute device."""
        with self._lock:
            key = f"{worker_id}:{status.device_id}"
            if status.state == DeviceState.UNAVAILABLE or status.state == DeviceState.OFFLINE:
                self._device_states[key] = ResourceState.UNAVAILABLE
            elif status.state == DeviceState.DEGRADED:
                self._device_states[key] = ResourceState.DEGRADED

    def get_available_devices(self, worker_id: Optional[str] = None) -> List[ComputeDevice]:
        """Return list of available compute devices on a given worker or cluster-wide."""
        with self._lock:
            available: List[ComputeDevice] = []
            target_workers = [worker_id] if worker_id and worker_id in self._workers else list(self._workers.keys())

            for w_id in target_workers:
                worker = self._workers[w_id]
                for dev in worker.devices:
                    key = f"{w_id}:{dev.device_id}"
                    state = self._device_states.get(key, ResourceState.AVAILABLE)
                    if state == ResourceState.AVAILABLE:
                        available.append(dev)
            return available

    def allocate_resources(
        self,
        job_id: str,
        worker_id: str,
        device_ids: List[str],
        cpus: int,
        memory_bytes: int,
        gpu_memory_bytes: int = 0,
    ) -> ResourceAllocation:
        """Atomically reserve and allocate resources on a worker node."""
        with self._lock:
            if job_id in self._job_allocations:
                raise ResourceError(f"Job {job_id} already has an active allocation.")

            if worker_id not in self._workers:
                raise ResourceError(f"Worker node {worker_id} is not registered.")

            worker = self._workers[worker_id]
            device_map = {dev.device_id: dev for dev in worker.devices}

            # Validate requested devices
            for dev_id in device_ids:
                if dev_id not in device_map:
                    raise ResourceError(f"Device {dev_id} does not exist on worker {worker_id}.")
                key = f"{worker_id}:{dev_id}"
                state = self._device_states.get(key, ResourceState.AVAILABLE)
                if state != ResourceState.AVAILABLE:
                    raise ResourceError(f"Device {dev_id} on worker {worker_id} is not AVAILABLE (state={state}).")

                # Check GPU memory capacity if applicable
                dev = device_map[dev_id]
                current_allocated = self._device_allocated_memory.get(key, 0)
                if gpu_memory_bytes > 0 and (current_allocated + gpu_memory_bytes > dev.total_memory_bytes):
                    raise ResourceError(
                        f"Insufficient GPU VRAM on device {dev_id}. "
                        f"Requested {gpu_memory_bytes} bytes, available {dev.total_memory_bytes - current_allocated} bytes."
                    )

            # Assign CPU cores (simple sequential assignment)
            assigned_cores = list(range(min(cpus, worker.total_cpus)))

            # Mark devices allocated
            for dev_id in device_ids:
                key = f"{worker_id}:{dev_id}"
                self._device_states[key] = ResourceState.ALLOCATED
                self._device_allocated_memory[key] += gpu_memory_bytes

            allocation = ResourceAllocation(
                job_id=job_id,
                worker_id=worker_id,
                assigned_device_ids=device_ids,
                assigned_cpu_cores=assigned_cores,
                memory_bytes=memory_bytes,
            )

            self._allocations[allocation.allocation_id] = allocation
            self._job_allocations[job_id] = allocation.allocation_id
            if self.storage_backend:
                self.storage_backend.save_allocation(allocation)
            return allocation

    def deallocate_resources(self, allocation_id_or_job_id: str) -> bool:
        """Deallocate resources for an allocation or job ID."""
        with self._lock:
            alloc_id = self._job_allocations.pop(allocation_id_or_job_id, allocation_id_or_job_id)
            alloc = self._allocations.pop(alloc_id, None)
            if not alloc:
                return False

            worker_id = alloc.worker_id
            for dev_id in alloc.assigned_device_ids:
                key = f"{worker_id}:{dev_id}"
                self._device_states[key] = ResourceState.AVAILABLE
                self._device_allocated_memory[key] = 0

            if self.storage_backend:
                self.storage_backend.delete_allocation(alloc.job_id)

            return True

    def get_allocation(self, allocation_id_or_job_id: str) -> Optional[ResourceAllocation]:
        """Fetch active allocation by allocation_id or job_id."""
        with self._lock:
            alloc_id = allocation_id_or_job_id
            if alloc_id in self._job_allocations:
                alloc_id = self._job_allocations[alloc_id]
            return self._allocations.get(alloc_id)

    def get_cluster_state(self) -> ClusterState:
        """Get live ClusterState snapshot."""
        with self._lock:
            return ClusterState(workers=dict(self._workers))

