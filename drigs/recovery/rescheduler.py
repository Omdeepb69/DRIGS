"""Automated Rescheduler for restoring failed jobs onto replacement worker nodes."""

import logging
from threading import RLock
from typing import Optional

from drigs.core.models import JobStatus
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.recovery.checkpoint import CheckpointManager
from drigs.recovery.detector import FailureDetector

logger = logging.getLogger(__name__)


class Rescheduler:
    """Orchestrates automated recovery and rescheduling of failed workloads onto healthy cluster nodes."""

    def __init__(
        self,
        job_queue: JobQueue,
        resource_manager: ResourceManager,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.job_queue = job_queue
        self.resource_manager = resource_manager
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self._lock = RLock()

    def reschedule_job(self, job_id: str, reason: str = "Automated recovery") -> bool:
        """Deallocate resources for a failed job, restore latest checkpoint, and re-queue for scheduling."""
        with self._lock:
            job = self.job_queue.get_job(job_id)
            if not job:
                logger.warning("Rescheduler cannot find job_id=%s in JobQueue", job_id)
                return False

            # 1. Release any previously allocated resources
            self.resource_manager.deallocate_resources(job_id)
            job.allocation = None

            # 2. Transition status to RECOVERING
            self.job_queue.update_job_status(job_id, JobStatus.RECOVERING, error_message=reason)
            logger.info("Job %s transitioned to RECOVERING (reason: %s)", job_id, reason)

            # 3. Check for latest checkpoint step
            latest_ckpt = self.checkpoint_manager.get_latest_checkpoint(job_id)
            if latest_ckpt:
                step = latest_ckpt.get("step", 0)
                path = latest_ckpt.get("filepath", "")
                logger.info("Restoring job %s from checkpoint step %d at %s", job_id, step, path)
                # Inject checkpoint environment variables into execution config
                job.spec.execution.env["DRIGS_RESTORE_CHECKPOINT"] = str(path)
                job.spec.execution.env["DRIGS_RESTORE_STEP"] = str(step)

            # 4. Transition status to QUEUED so scheduler re-allocates replacement nodes
            self.job_queue.update_job_status(job_id, JobStatus.QUEUED)
            logger.info("Successfully re-queued job %s for automated recovery rescheduling", job_id)
            return True

    def attach_failure_detector(self, detector: FailureDetector) -> None:
        """Attach FailureDetector callbacks to trigger automatic rescheduling on job failures."""
        detector.register_job_failure_callback(
            lambda job_id, reason: self.reschedule_job(job_id, reason)
        )
