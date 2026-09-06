"""Remote Job Dispatcher for routing and managing execution across worker nodes."""

import logging
from threading import RLock
from typing import AsyncIterator, Dict, List, Optional

from drigs.core.interfaces import ExecutionHandle
from drigs.core.models import DeviceState, Job, JobStatus, ResourceAllocation
from drigs.workers.agent import WorkerAgent
from drigs.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class RemoteDispatcher:
    """Dispatches workload jobs to target worker nodes and monitors execution across cluster workers."""

    def __init__(self, registry: Optional[WorkerRegistry] = None):
        self.registry = registry or WorkerRegistry()
        self._agents: Dict[str, WorkerAgent] = {}  # worker_id -> WorkerAgent
        self._handles: Dict[str, ExecutionHandle] = {}  # job_id -> ExecutionHandle
        self._job_worker_map: Dict[str, str] = {}  # job_id -> worker_id
        self._lock = RLock()

    def register_agent(self, agent: WorkerAgent) -> bool:
        """Register a WorkerAgent instance and record its telemetry in the worker registry."""
        with self._lock:
            self._agents[agent.worker_id] = agent
            registered = self.registry.register(agent.get_worker_info())
            logger.info("Registered agent %s in RemoteDispatcher", agent.worker_id)
            return registered

    def unregister_agent(self, worker_id: str) -> bool:
        """Unregister a WorkerAgent instance and remove it from the worker registry."""
        with self._lock:
            self._agents.pop(worker_id, None)
            return self.registry.deregister(worker_id)

    def get_agent(self, worker_id: str) -> Optional[WorkerAgent]:
        """Fetch registered WorkerAgent instance by worker_id."""
        with self._lock:
            return self._agents.get(worker_id)

    def dispatch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Dispatch a job for remote execution to the designated worker node in allocation."""
        with self._lock:
            worker_id = allocation.worker_id
            worker_info = self.registry.get_worker(worker_id)

            if not worker_info:
                raise ValueError(f"Target worker {worker_id} is not registered in cluster registry")

            if worker_info.status in (DeviceState.OFFLINE, DeviceState.UNAVAILABLE):
                raise RuntimeError(
                    f"Cannot dispatch job {job.id}: Worker {worker_id} is currently {worker_info.status.value}"
                )

            agent = self._agents.get(worker_id)
            if not agent:
                raise RuntimeError(
                    f"No active agent client connected to RemoteDispatcher for worker {worker_id}"
                )

            handle = agent.launch_job(job, allocation)
            self._handles[job.id] = handle
            self._job_worker_map[job.id] = worker_id
            logger.info("Successfully dispatched job %s to worker %s", job.id, worker_id)
            return handle

    def stop_job(self, job_id: str) -> bool:
        """Stop a job running on its dispatched worker node."""
        with self._lock:
            worker_id = self._job_worker_map.get(job_id)
            if not worker_id:
                return False

            agent = self._agents.get(worker_id)
            if not agent:
                return False

            stopped = agent.stop_job(job_id)
            if stopped:
                self._handles.pop(job_id, None)
                self._job_worker_map.pop(job_id, None)
            return stopped

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get execution status for a job from its dispatched worker node."""
        with self._lock:
            worker_id = self._job_worker_map.get(job_id)
            if not worker_id:
                return None

            worker_info = self.registry.get_worker(worker_id)
            if worker_info and worker_info.status in (DeviceState.OFFLINE, DeviceState.UNAVAILABLE):
                logger.warning("Worker %s for job %s is %s", worker_id, job_id, worker_info.status.value)
                return JobStatus.FAILED

            agent = self._agents.get(worker_id)
            if not agent:
                return None

            return agent.get_job_status(job_id)

    async def stream_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """Stream log output for a job from its assigned worker node."""
        with self._lock:
            worker_id = self._job_worker_map.get(job_id)
            agent = self._agents.get(worker_id) if worker_id else None

        if not agent:
            return

        async for line in agent.stream_job_logs(job_id):
            yield line

    def get_active_worker_ids(self) -> List[str]:
        """Return list of active worker IDs registered with cluster."""
        active_workers = self.registry.get_active_workers()
        return [w.worker_id for w in active_workers]

    def get_dispatched_jobs(self) -> Dict[str, str]:
        """Return snapshot of job_id -> worker_id dispatch map."""
        with self._lock:
            return dict(self._job_worker_map)
