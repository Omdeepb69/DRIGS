"""Abstract protocol interfaces for DRIGS system components."""

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, Field

from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceStatus,
    Job,
    JobStatus,
    ResourceAllocation,
    WorkerInfo,
)


class ExecutionHandle(BaseModel):
    """Reference handle for a running or executed workload instance."""

    handle_id: str
    job_id: str
    backend_type: str
    pid: Optional[int] = None
    status: JobStatus = JobStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class HardwareBackend(Protocol):
    """Protocol for hardware discovery and device status collection."""

    def discover_devices(self) -> List[ComputeDevice]:
        """Discover all compute devices present on the local node."""
        ...

    def get_device_status(self, device_id: str) -> DeviceStatus:
        """Fetch current operational and metric status for a specific device."""
        ...

    def get_topology(self) -> Dict[str, Any]:
        """Fetch local device interconnect and NUMA topology graph/matrix."""
        ...


@runtime_checkable
class Scheduler(Protocol):
    """Protocol for workload placement and scheduling algorithms."""

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        """Given queued jobs and cluster state, compute resource allocations."""
        ...


@runtime_checkable
class ExecutionBackend(Protocol):
    """Protocol for launching, controlling, and logging workload processes."""

    def launch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Launch job process(es) bound to the provided resource allocation."""
        ...

    def stop(self, handle: ExecutionHandle) -> bool:
        """Terminate the running workload process."""
        ...

    def get_status(self, handle: ExecutionHandle) -> JobStatus:
        """Fetch current execution status of the workload handle."""
        ...

    async def stream_logs(self, handle: ExecutionHandle) -> AsyncIterator[str]:
        """Yield log stream lines from stdout/stderr of the workload handle."""
        ...


@runtime_checkable
class WorkerRegistryProtocol(Protocol):
    """Protocol for central cluster worker registration and health monitoring."""

    def register(self, worker: WorkerInfo) -> bool:
        """Register a new or restarted worker node into the cluster."""
        ...

    def heartbeat(self, worker_id: str, status: DeviceState) -> bool:
        """Record periodic heartbeat ping from a worker node."""
        ...

    def get_active_workers(self) -> List[WorkerInfo]:
        """Return list of all currently active and healthy worker nodes."""
        ...

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Return WorkerInfo for a given worker_id or None if not registered."""
        ...
