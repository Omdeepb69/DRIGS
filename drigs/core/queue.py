"""Admission Controller and Job Queue Management for DRIGS."""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Any, Dict, List, Optional, Set, Tuple

from drigs.core.models import Job, JobStatus

logger = logging.getLogger(__name__)


class AdmissionError(Exception):
    """Exception raised when a job fails admission validation."""

    pass


class StateTransitionError(Exception):
    """Exception raised when an invalid job status transition is attempted."""

    pass


# Valid state machine transitions
VALID_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.QUEUED: {JobStatus.SCHEDULED, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.SCHEDULED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RECOVERING},
    JobStatus.RECOVERING: {JobStatus.QUEUED, JobStatus.SCHEDULED, JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}


def validate_status_transition(current_status: JobStatus, new_status: JobStatus) -> None:
    """Validate whether transitioning from current_status to new_status is allowed."""
    if current_status == new_status:
        return

    allowed = VALID_TRANSITIONS.get(current_status, set())
    if new_status not in allowed:
        raise StateTransitionError(
            f"Invalid job status transition: cannot transition from {current_status.value} to {new_status.value}"
        )


class AdmissionController:
    """Validates workload specifications and admits jobs into the control plane."""

    def __init__(self, max_gpu_limit: Optional[int] = None, max_cpu_limit: Optional[int] = None):
        self.max_gpu_limit = max_gpu_limit
        self.max_cpu_limit = max_cpu_limit

    def validate_job(self, job: Job) -> Tuple[bool, Optional[str]]:
        """Validate job properties and resource requirements."""
        if not job.name or not job.name.strip():
            return False, "Job name cannot be empty"

        if not job.spec:
            return False, "Job specification is missing"

        if not job.spec.name or not job.spec.name.strip():
            return False, "Workload spec name cannot be empty"

        exec_config = job.spec.execution
        if not exec_config or not exec_config.entrypoint:
            return False, "Execution entrypoint cannot be empty"

        req = job.spec.resources
        if req.gpus < 0:
            return False, "Requested GPU count cannot be negative"
        if req.cpus < 1:
            return False, "Requested CPU count must be at least 1"
        if req.gpu_memory_bytes < 0:
            return False, "Requested GPU memory cannot be negative"
        if req.memory_bytes < 0:
            return False, "Requested system memory cannot be negative"

        if self.max_gpu_limit is not None and req.gpus > self.max_gpu_limit:
            return False, f"Requested GPUs ({req.gpus}) exceeds maximum allowed limit ({self.max_gpu_limit})"

        if self.max_cpu_limit is not None and req.cpus > self.max_cpu_limit:
            return False, f"Requested CPUs ({req.cpus}) exceeds maximum allowed limit ({self.max_cpu_limit})"

        return True, None

    def admit_job(self, job: Job) -> Job:
        """Validate and admit a job, transitioning status from PENDING to QUEUED."""
        valid, reason = self.validate_job(job)
        if not valid:
            raise AdmissionError(f"Job admission rejected: {reason}")

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.QUEUED
        return job


class JobQueue:
    """Thread-safe priority job queue managing queued and active jobs."""

    def __init__(self, max_capacity: Optional[int] = None, storage_backend: Optional[Any] = None):
        self._lock = RLock()
        self.max_capacity = max_capacity
        self.storage_backend = storage_backend
        self._queue: List[Job] = []  # Kept sorted by (-priority, submitted_at)
        self._jobs: Dict[str, Job] = {}  # job_id -> Job

    def load_from_storage(self) -> int:
        """Load and restore jobs from storage backend into queue."""
        with self._lock:
            if not self.storage_backend:
                return 0
            stored_jobs = self.storage_backend.load_all_jobs()
            for job in stored_jobs:
                self._jobs[job.id] = job
                if job.status == JobStatus.QUEUED:
                    if job not in self._queue:
                        self._queue.append(job)
            self._sort_queue()
            return len(stored_jobs)

    def _sort_queue(self) -> None:
        self._queue.sort(key=lambda j: (-j.priority, j.submitted_at))

    def enqueue(self, job: Job) -> None:
        """Add a job to the priority queue."""
        with self._lock:
            if self.max_capacity is not None and len(self._queue) >= self.max_capacity:
                raise AdmissionError(f"Job queue capacity exceeded (max {self.max_capacity})")

            if job.id in self._jobs:
                raise AdmissionError(f"Job {job.id} is already present in the queue")

            if job.status == JobStatus.PENDING:
                job.status = JobStatus.QUEUED

            self._jobs[job.id] = job
            self._queue.append(job)
            self._sort_queue()
            if self.storage_backend:
                self.storage_backend.save_job(job)

    def dequeue(self) -> Optional[Job]:
        """Pop and return the highest priority queued job."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.status == JobStatus.QUEUED:
                    return self._queue.pop(i)
            return None

    def peek(self) -> Optional[Job]:
        """Inspect the highest priority queued job without removing it."""
        with self._lock:
            for job in self._queue:
                if job.status == JobStatus.QUEUED:
                    return job
            return None

    def get_pending_jobs(self) -> List[Job]:
        """Return all jobs currently in QUEUED status in priority order."""
        with self._lock:
            return [j for j in self._queue if j.status == JobStatus.QUEUED]

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> Optional[Job]:
        """Remove job from queue by ID."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job and job in self._queue:
                self._queue.remove(job)
            if job and self.storage_backend:
                self.storage_backend.delete_job(job_id)
            return job

    def update_job_status(
        self,
        job_id: str,
        new_status: JobStatus,
        error_message: Optional[str] = None,
    ) -> Job:
        """Update job status ensuring valid state machine transitions."""
        with self._lock:
            job = self.get_job(job_id)
            if not job:
                raise StateTransitionError(f"Job {job_id} not found in queue")

            validate_status_transition(job.status, new_status)
            job.status = new_status

            now = datetime.now(timezone.utc)
            if new_status == JobStatus.RUNNING and job.started_at is None:
                job.started_at = now
            elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.started_at is None:
                    job.started_at = now
                job.completed_at = now

            if error_message is not None:
                job.error_message = error_message

            if self.storage_backend:
                self.storage_backend.save_job(job)

            return job

    def size(self) -> int:
        """Return total number of queued/tracked jobs."""
        with self._lock:
            return len(self._jobs)

    def queued_count(self) -> int:
        """Return count of jobs in QUEUED status."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
