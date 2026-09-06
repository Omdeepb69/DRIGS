"""Unit tests for AdmissionController and JobQueue in drigs.core.queue."""

from datetime import datetime, timedelta, timezone
import pytest

from drigs.core.models import (
    Job,
    JobStatus,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
)
from drigs.core.queue import (
    AdmissionController,
    AdmissionError,
    JobQueue,
    StateTransitionError,
)


def _make_job(name: str = "test-job", priority: int = 0, gpus: int = 1, cpus: int = 2) -> Job:
    return Job(
        name=name,
        priority=priority,
        spec=WorkloadSpec(
            name=name,
            resources=ResourceRequirements(gpus=gpus, cpus=cpus),
            execution=WorkloadExecutionConfig(entrypoint="python run.py"),
        ),
    )


def test_admission_controller_valid():
    controller = AdmissionController(max_gpu_limit=4, max_cpu_limit=16)
    job = _make_job(gpus=2, cpus=4)

    valid, reason = controller.validate_job(job)
    assert valid is True
    assert reason is None

    admitted_job = controller.admit_job(job)
    assert admitted_job.status == JobStatus.QUEUED


def test_admission_controller_invalid_requests():
    controller = AdmissionController(max_gpu_limit=2)

    # Empty job name
    invalid_job = _make_job(name="")
    valid, reason = controller.validate_job(invalid_job)
    assert valid is False
    assert "Job name cannot be empty" in reason

    # Exceeds max GPU limit
    heavy_job = _make_job(gpus=8)
    valid, reason = controller.validate_job(heavy_job)
    assert valid is False
    assert "exceeds maximum allowed limit" in reason

    with pytest.raises(AdmissionError, match="rejected"):
        controller.admit_job(heavy_job)


def test_job_queue_priority_ordering():
    now = datetime.now(timezone.utc)
    job_low = Job(
        name="low-prio",
        priority=0,
        spec=_make_job().spec,
        submitted_at=now - timedelta(seconds=10),
    )
    job_high = Job(
        name="high-prio",
        priority=10,
        spec=_make_job().spec,
        submitted_at=now - timedelta(seconds=5),
    )
    job_equal_old = Job(
        name="equal-old",
        priority=10,
        spec=_make_job().spec,
        submitted_at=now - timedelta(seconds=20),
    )

    queue = JobQueue()
    queue.enqueue(job_low)
    queue.enqueue(job_high)
    queue.enqueue(job_equal_old)

    assert queue.size() == 3
    assert queue.queued_count() == 3

    # First dequeued should be equal-old (prio 10, oldest)
    j1 = queue.dequeue()
    assert j1.id == job_equal_old.id

    # Second dequeued should be high-prio (prio 10)
    j2 = queue.dequeue()
    assert j2.id == job_high.id

    # Third dequeued should be low-prio (prio 0)
    j3 = queue.dequeue()
    assert j3.id == job_low.id

    assert queue.dequeue() is None


def test_job_queue_capacity_limit():
    queue = JobQueue(max_capacity=2)
    j1 = _make_job("j1")
    j2 = _make_job("j2")
    j3 = _make_job("j3")

    queue.enqueue(j1)
    queue.enqueue(j2)

    with pytest.raises(AdmissionError, match="capacity exceeded"):
        queue.enqueue(j3)


def test_job_queue_state_transitions():
    queue = JobQueue()
    job = _make_job("state-test")
    queue.enqueue(job)

    # QUEUED -> SCHEDULED
    job = queue.update_job_status(job.id, JobStatus.SCHEDULED)
    assert job.status == JobStatus.SCHEDULED

    # SCHEDULED -> RUNNING
    job = queue.update_job_status(job.id, JobStatus.RUNNING)
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None

    # RUNNING -> COMPLETED
    job = queue.update_job_status(job.id, JobStatus.COMPLETED)
    assert job.status == JobStatus.COMPLETED
    assert job.completed_at is not None

    # Cannot transition from COMPLETED to RUNNING
    with pytest.raises(StateTransitionError, match="Invalid job status transition"):
        queue.update_job_status(job.id, JobStatus.RUNNING)
