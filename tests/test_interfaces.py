"""Unit tests for DRIGS abstract protocol interfaces."""

from typing import AsyncIterator, List, Optional
import pytest

from drigs.core.interfaces import (
    ExecutionBackend,
    ExecutionHandle,
    HardwareBackend,
    Scheduler,
    WorkerRegistryProtocol,
)
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceStatus,
    DeviceType,
    Job,
    JobStatus,
    ResourceAllocation,
    WorkerInfo,
)


class DummyHardwareBackend:
    """Mock implementation of HardwareBackend."""

    def discover_devices(self) -> List[ComputeDevice]:
        return [
            ComputeDevice(
                device_id="gpu-0",
                device_type=DeviceType.GPU,
                model_name="Mock GPU",
                total_memory_bytes=1000,
                available_memory_bytes=1000,
            )
        ]

    def get_device_status(self, device_id: str) -> DeviceStatus:
        return DeviceStatus(device_id=device_id)

    def get_topology(self) -> dict:
        return {"nodes": ["gpu-0"]}


class DummyScheduler:
    """Mock implementation of Scheduler."""

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        return []


class DummyExecutionBackend:
    """Mock implementation of ExecutionBackend."""

    def launch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        return ExecutionHandle(
            handle_id="handle-1",
            job_id=job.job_id,
            backend_type="dummy",
            status=JobStatus.RUNNING,
        )

    def stop(self, handle: ExecutionHandle) -> bool:
        return True

    def get_status(self, handle: ExecutionHandle) -> JobStatus:
        return JobStatus.RUNNING

    async def stream_logs(self, handle: ExecutionHandle) -> AsyncIterator[str]:
        yield "log line 1\n"


class DummyWorkerRegistry:
    """Mock implementation of WorkerRegistryProtocol."""

    def register(self, worker: WorkerInfo) -> bool:
        return True

    def heartbeat(self, worker_id: str, status: DeviceState) -> bool:
        return True

    def get_active_workers(self) -> List[WorkerInfo]:
        return []

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        return None


class IncompleteBackend:
    """Incomplete class missing methods."""

    pass


def test_hardware_backend_protocol():
    """Verify runtime check for HardwareBackend protocol."""
    backend = DummyHardwareBackend()
    assert isinstance(backend, HardwareBackend)
    assert not isinstance(IncompleteBackend(), HardwareBackend)

    devices = backend.discover_devices()
    assert len(devices) == 1
    assert devices[0].device_id == "gpu-0"


def test_scheduler_protocol():
    """Verify runtime check for Scheduler protocol."""
    scheduler = DummyScheduler()
    assert isinstance(scheduler, Scheduler)
    assert not isinstance(IncompleteBackend(), Scheduler)

    allocations = scheduler.schedule([], ClusterState())
    assert allocations == []


def test_execution_backend_protocol():
    """Verify runtime check for ExecutionBackend protocol."""
    exec_backend = DummyExecutionBackend()
    assert isinstance(exec_backend, ExecutionBackend)
    assert not isinstance(IncompleteBackend(), ExecutionBackend)


def test_worker_registry_protocol():
    """Verify runtime check for WorkerRegistryProtocol."""
    registry = DummyWorkerRegistry()
    assert isinstance(registry, WorkerRegistryProtocol)
    assert not isinstance(IncompleteBackend(), WorkerRegistryProtocol)


def test_execution_handle():
    """Verify ExecutionHandle instantiation and attributes."""
    handle = ExecutionHandle(
        handle_id="h-100",
        job_id="j-200",
        backend_type="native",
        pid=12345,
        status=JobStatus.RUNNING,
        metadata={"cwd": "/tmp"},
    )
    assert handle.pid == 12345
    assert handle.backend_type == "native"
    assert handle.metadata["cwd"] == "/tmp"
