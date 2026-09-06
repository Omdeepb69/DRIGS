"""Worker Agent & Heartbeat Service for DRIGS node management."""

import asyncio
from datetime import datetime, timezone
import logging
import socket
from typing import AsyncIterator, Dict, List, Optional
import uuid

import psutil

from drigs.core.interfaces import (
    ExecutionBackend,
    ExecutionHandle,
    HardwareBackend,
    WorkerRegistryProtocol,
)
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    Job,
    JobStatus,
    ResourceAllocation,
    WorkerInfo,
)

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class WorkerAgent:
    """Worker node agent managing local hardware discovery, heartbeat reporting, and job execution."""

    def __init__(
        self,
        worker_id: Optional[str] = None,
        hostname: Optional[str] = None,
        ip_address: Optional[str] = None,
        hardware_backend: Optional[HardwareBackend] = None,
        execution_backend: Optional[ExecutionBackend] = None,
        registry: Optional[WorkerRegistryProtocol] = None,
        heartbeat_interval: float = 5.0,
    ):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.hostname = hostname or socket.gethostname()
        try:
            self.ip_address = ip_address or socket.gethostbyname(self.hostname)
        except Exception:
            self.ip_address = ip_address or "127.0.0.1"

        if hardware_backend is None:
            from drigs.hardware.cpu import CPUBackend

            self.hardware_backend: HardwareBackend = CPUBackend()
        else:
            self.hardware_backend = hardware_backend

        if execution_backend is None:
            from drigs.execution.native import NativeProcessBackend

            self.execution_backend: ExecutionBackend = NativeProcessBackend()
        else:
            self.execution_backend = execution_backend

        self.registry = registry
        self.heartbeat_interval = heartbeat_interval

        self.status = DeviceState.HEALTHY
        self._active_handles: Dict[str, ExecutionHandle] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
        self.heartbeat_count = 0
        self.last_heartbeat: Optional[datetime] = None

    def discover_devices(self) -> List[ComputeDevice]:
        """Discover compute devices on this worker node."""
        return self.hardware_backend.discover_devices()

    def get_worker_info(self) -> WorkerInfo:
        """Construct and return current WorkerInfo telemetry snapshot."""
        devices = self.discover_devices()
        total_cpus = psutil.cpu_count(logical=True) or 1
        total_memory = psutil.virtual_memory().total

        return WorkerInfo(
            worker_id=self.worker_id,
            hostname=self.hostname,
            ip_address=self.ip_address,
            devices=devices,
            total_cpus=total_cpus,
            total_memory_bytes=total_memory,
            status=self.status,
            last_heartbeat=self.last_heartbeat or _now_utc(),
        )

    def register(self) -> bool:
        """Register worker with central registry if configured."""
        if self.registry is not None:
            info = self.get_worker_info()
            return self.registry.register(info)
        return True

    def send_heartbeat(self) -> bool:
        """Send a single heartbeat ping to central registry and update timestamp."""
        self.last_heartbeat = _now_utc()
        self.heartbeat_count += 1
        if self.registry is not None:
            return self.registry.heartbeat(self.worker_id, self.status)
        return True

    async def start(self) -> None:
        """Start worker agent, register with control plane, and launch heartbeat loop."""
        if self._running:
            return
        self._running = True
        self.register()
        self.send_heartbeat()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop worker agent, cancel heartbeat loop, and terminate running jobs."""
        self._running = False
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        for handle in list(self._active_handles.values()):
            try:
                self.execution_backend.stop(handle)
            except Exception as e:
                logger.error("Error stopping job %s during worker shutdown: %s", handle.job_id, e)

    async def _heartbeat_loop(self) -> None:
        """Async background loop for periodic heartbeat dispatching."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._running:
                    break
                self.send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in worker agent heartbeat loop: %s", e)

    def launch_job(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Launch job process on local execution backend."""
        handle = self.execution_backend.launch(job, allocation)
        self._active_handles[job.id] = handle
        return handle

    def stop_job(self, job_id: str) -> bool:
        """Stop local job execution by job_id."""
        handle = self._active_handles.get(job_id)
        if not handle:
            return False
        return self.execution_backend.stop(handle)

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Fetch current status of a job running on this worker."""
        handle = self._active_handles.get(job_id)
        if not handle:
            return None
        return self.execution_backend.get_status(handle)

    async def stream_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """Stream log output for a job running on this worker."""
        handle = self._active_handles.get(job_id)
        if not handle:
            return
        async for line in self.execution_backend.stream_logs(handle):
            yield line

    async def __aenter__(self) -> "WorkerAgent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
