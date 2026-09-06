"""Worker and Job Failure Detector for DRIGS fault tolerance."""

import logging
from threading import Event, RLock, Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from drigs.core.interfaces import ExecutionBackend, ExecutionHandle
from drigs.core.models import DeviceState, Job, JobStatus
from drigs.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class FailureDetector:
    """Monitors worker node heartbeats and workload execution handles for failures and triggers recovery callbacks."""

    def __init__(
        self,
        registry: Optional[WorkerRegistry] = None,
        check_interval_seconds: float = 1.0,
    ):
        self.registry = registry or WorkerRegistry()
        self.check_interval_seconds = check_interval_seconds

        self._monitored_jobs: Dict[str, Dict[str, Any]] = {}
        self._failed_workers: Set[str] = set()
        self._failed_jobs: Set[str] = set()

        self._worker_failure_callbacks: List[Callable[[str], None]] = []
        self._job_failure_callbacks: List[Callable[[str, str], None]] = []

        self._lock = RLock()
        self._running_event = Event()
        self._loop_thread: Optional[Thread] = None

    def register_worker_failure_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback triggered when a worker node fails or times out."""
        with self._lock:
            self._worker_failure_callbacks.append(callback)

    def register_job_failure_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register callback triggered when a monitored job fails."""
        with self._lock:
            self._job_failure_callbacks.append(callback)

    def track_job(
        self,
        job: Job,
        worker_id: str,
        handle: ExecutionHandle,
        backend: ExecutionBackend,
    ) -> None:
        """Track an active job for process execution failure monitoring."""
        with self._lock:
            self._monitored_jobs[job.id] = {
                "job": job,
                "worker_id": worker_id,
                "handle": handle,
                "backend": backend,
            }
            logger.info("FailureDetector tracking job %s on worker %s", job.id, worker_id)

    def untrack_job(self, job_id: str) -> None:
        """Stop tracking a job (e.g. upon completion or cancellation)."""
        with self._lock:
            self._monitored_jobs.pop(job_id, None)

    def check_failures(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Scan workers and monitored jobs, triggering callbacks for newly detected failures."""
        with self._lock:
            new_worker_failures: List[str] = []
            new_job_failures: List[Tuple[str, str]] = []

            # 1. Detect worker node heartbeat timeouts
            timed_out_workers = self.registry.check_timeouts()
            for w_id in timed_out_workers:
                if w_id not in self._failed_workers:
                    self._failed_workers.add(w_id)
                    new_worker_failures.append(w_id)
                    logger.error("Detected worker node failure for worker_id=%s", w_id)

                    for cb in self._worker_failure_callbacks:
                        try:
                            cb(w_id)
                        except Exception as err:
                            logger.error("Error in worker failure callback: %s", err)

            # 2. Check jobs running on failed workers
            for job_id, info in list(self._monitored_jobs.items()):
                w_id = info["worker_id"]
                if w_id in self._failed_workers and job_id not in self._failed_jobs:
                    reason = f"Worker node {w_id} failed / timed out"
                    self._failed_jobs.add(job_id)
                    new_job_failures.append((job_id, reason))
                    logger.error("Detected job failure for job_id=%s: %s", job_id, reason)

                    for cb in self._job_failure_callbacks:
                        try:
                            cb(job_id, reason)
                        except Exception as err:
                            logger.error("Error in job failure callback: %s", err)

            # 3. Detect process execution failures via ExecutionBackend status
            for job_id, info in list(self._monitored_jobs.items()):
                if job_id in self._failed_jobs:
                    continue

                backend: ExecutionBackend = info["backend"]
                handle: ExecutionHandle = info["handle"]

                try:
                    status = backend.get_status(handle)
                    if status == JobStatus.FAILED:
                        reason = "Workload process terminated with non-zero exit code or error"
                        self._failed_jobs.add(job_id)
                        new_job_failures.append((job_id, reason))
                        logger.error("Detected process failure for job_id=%s: %s", job_id, reason)

                        for cb in self._job_failure_callbacks:
                            try:
                                cb(job_id, reason)
                            except Exception as err:
                                logger.error("Error in job failure callback: %s", err)
                except Exception as err:
                    reason = f"Failed to poll process status: {err}"
                    self._failed_jobs.add(job_id)
                    new_job_failures.append((job_id, reason))
                    logger.error("Error checking job %s status: %s", job_id, err)

            return new_worker_failures, new_job_failures

    def _loop(self) -> None:
        """Background thread loop periodically calling check_failures()."""
        while self._running_event.is_set():
            try:
                self.check_failures()
            except Exception as err:
                logger.error("Error in FailureDetector monitoring loop: %s", err)
            self._running_event.wait(self.check_interval_seconds)

    def start(self) -> None:
        """Start the background monitoring thread loop."""
        with self._lock:
            if self._running_event.is_set():
                return
            self._running_event.set()
            self._loop_thread = Thread(target=self._loop, daemon=True, name="drigs-failure-detector")
            self._loop_thread.start()
            logger.info("FailureDetector background monitoring thread started")

    def stop(self) -> None:
        """Stop the background monitoring thread loop."""
        with self._lock:
            if not self._running_event.is_set():
                return
            self._running_event.clear()
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            logger.info("FailureDetector background monitoring thread stopped")
