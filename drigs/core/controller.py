"""Local Control Plane Orchestrator for DRIGS."""

import asyncio
import logging
from threading import Event, RLock, Thread
from typing import Any, Dict, List, Optional, Union

from drigs.core.interfaces import ExecutionBackend, ExecutionHandle, Scheduler
from drigs.core.models import (
    ClusterState,
    Job,
    JobStatus,
    ResourceAllocation,
    WorkloadSpec,
)
from drigs.core.queue import AdmissionController, JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.core.spec import parse_workload_spec
from drigs.execution.native import NativeProcessBackend
from drigs.hardware.cpu import CPUBackend
from drigs.scheduler.fifo import FIFOScheduler

logger = logging.getLogger(__name__)


class LocalController:
    """Single-node local control plane loop managing submission, scheduling, execution, and cleanup."""

    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        scheduler: Optional[Scheduler] = None,
        execution_backend: Optional[ExecutionBackend] = None,
        admission_controller: Optional[AdmissionController] = None,
        poll_interval_seconds: float = 0.2,
        auto_register_local_node: bool = True,
    ):
        self.resource_manager = resource_manager or ResourceManager()
        self.scheduler = scheduler or FIFOScheduler()
        self.execution_backend = execution_backend or NativeProcessBackend()
        self.admission_controller = admission_controller or AdmissionController()
        self.job_queue = JobQueue()

        self.poll_interval_seconds = poll_interval_seconds
        self._lock = RLock()
        self._handles: Dict[str, ExecutionHandle] = {}  # job_id -> ExecutionHandle
        self._allocations: Dict[str, ResourceAllocation] = {}  # job_id -> ResourceAllocation

        self._running_event = Event()
        self._loop_thread: Optional[Thread] = None

        if auto_register_local_node:
            self._auto_register_local_worker()

    def _auto_register_local_worker(self) -> None:
        """Register the local machine host as worker-local if no workers are present."""
        if not self.resource_manager.get_cluster_state().workers:
            devices = []

            try:
                cpu_backend = CPUBackend()
                devices.extend(cpu_backend.discover_devices())
            except Exception as err:
                logger.warning("Could not discover CPU devices: %s", err)

            try:
                from drigs.hardware.cuda import CUDABackend
                cuda_backend = CUDABackend()
                devices.extend(cuda_backend.discover_devices())
            except Exception:
                pass

            import psutil
            total_cpus = psutil.cpu_count(logical=True) or 1
            total_memory = psutil.virtual_memory().total

            from drigs.core.models import WorkerInfo
            worker_info = WorkerInfo(
                worker_id="worker-local",
                hostname="localhost",
                ip_address="127.0.0.1",
                devices=devices,
                total_cpus=total_cpus,
                total_memory_bytes=total_memory,
            )
            self.resource_manager.register_worker(worker_info)

    def submit_job(
        self,
        spec_or_content: Union[WorkloadSpec, Dict[str, Any], str],
        priority: int = 0,
        job_name: Optional[str] = None,
    ) -> Job:
        """Submit a workload spec or YAML/dict into the DRIGS control plane."""
        if isinstance(spec_or_content, WorkloadSpec):
            spec = spec_or_content
        else:
            spec = parse_workload_spec(spec_or_content)

        name = job_name or spec.name
        job = Job(name=name, priority=priority, spec=spec)

        admitted_job = self.admission_controller.admit_job(job)
        self.job_queue.enqueue(admitted_job)
        logger.info("Admitted and queued job %s (priority=%d)", admitted_job.id, priority)
        return admitted_job

    def step(self) -> int:
        """Run a single control plane orchestration pass (schedule -> allocate -> execute -> monitor)."""
        with self._lock:
            cluster_state = self.resource_manager.get_cluster_state()
            pending_jobs = self.job_queue.get_pending_jobs()

            # 1. Compute scheduling allocations
            if pending_jobs and cluster_state.workers:
                scheduled_allocations = self.scheduler.schedule(pending_jobs, cluster_state)

                for alloc in scheduled_allocations:
                    job = self.job_queue.get_job(alloc.job_id)
                    if not job:
                        continue

                    try:
                        # 1. Update status to SCHEDULED
                        self.job_queue.update_job_status(job.id, JobStatus.SCHEDULED)

                        # 2. Reserve resources
                        real_alloc = self.resource_manager.allocate_resources(
                            job_id=job.id,
                            worker_id=alloc.worker_id,
                            device_ids=alloc.assigned_device_ids,
                            cpus=job.spec.resources.cpus,
                            memory_bytes=job.spec.resources.memory_bytes,
                            gpu_memory_bytes=job.spec.resources.gpu_memory_bytes,
                        )
                        job.allocation = real_alloc

                        # 3. Launch process via execution backend
                        handle = self.execution_backend.launch(job, real_alloc)
                        self._handles[job.id] = handle
                        self._allocations[job.id] = real_alloc

                        # 4. Update status to RUNNING
                        self.job_queue.update_job_status(job.id, JobStatus.RUNNING)
                        logger.info("Launched job %s on worker %s", job.id, alloc.worker_id)
                    except Exception as err:
                        logger.error("Failed to allocate or launch job %s: %s", job.id, err)
                        self.job_queue.update_job_status(job.id, JobStatus.FAILED, error_message=str(err))
                        self.resource_manager.deallocate_resources(job.id)

            # 5. Monitor and poll active execution handles
            active_job_ids = list(self._handles.keys())
            for job_id in active_job_ids:
                handle = self._handles[job_id]
                try:
                    status = self.execution_backend.get_status(handle)
                    if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                        logger.info("Job %s completed with status %s", job_id, status.value)
                        self.job_queue.update_job_status(job_id, status)
                        self.resource_manager.deallocate_resources(job_id)
                        self._handles.pop(job_id, None)
                        self._allocations.pop(job_id, None)
                except Exception as err:
                    logger.error("Error checking status for job %s: %s", job_id, err)

            return len(self._handles)

    def _loop(self) -> None:
        """Internal worker thread loop executing step() periodically."""
        while self._running_event.is_set():
            try:
                self.step()
            except Exception as err:
                logger.error("Error in LocalController loop: %s", err)
            self._running_event.wait(self.poll_interval_seconds)

    def start(self) -> None:
        """Start the background control loop thread."""
        with self._lock:
            if self._running_event.is_set():
                return
            self._running_event.set()
            self._loop_thread = Thread(target=self._loop, daemon=True, name="drigs-controller-loop")
            self._loop_thread.start()
            logger.info("LocalController loop started")

    def stop(self) -> None:
        """Stop the background control loop thread."""
        with self._lock:
            if not self._running_event.is_set():
                return
            self._running_event.clear()
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            logger.info("LocalController loop stopped")

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job and release its allocated resources."""
        with self._lock:
            job = self.job_queue.get_job(job_id)
            if not job:
                return False

            if job.id in self._handles:
                handle = self._handles.pop(job_id)
                self.execution_backend.stop(handle)
                self.resource_manager.deallocate_resources(job_id)
                self._allocations.pop(job_id, None)

            if job.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.SCHEDULED, JobStatus.RUNNING):
                self.job_queue.update_job_status(job_id, JobStatus.CANCELLED)
                return True
            return False

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job model by ID."""
        return self.job_queue.get_job(job_id)

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Retrieve current job status by ID."""
        job = self.job_queue.get_job(job_id)
        return job.status if job else None
