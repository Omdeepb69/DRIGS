"""Unit tests for FailureDetector in drigs.recovery.detector."""

import tempfile
import time
import pytest

from drigs.core.interfaces import ExecutionHandle
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.execution.native import NativeProcessBackend
from drigs.recovery.detector import FailureDetector
from drigs.workers.registry import WorkerRegistry


def _make_worker(worker_id: str = "w-1") -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        hostname="node-1",
        ip_address="127.0.0.1",
        devices=[],
        total_cpus=4,
        total_memory_bytes=16 * 1024**3,
        status=DeviceState.HEALTHY,
    )


def test_failure_detector_worker_timeout():
    registry = WorkerRegistry(timeout_seconds=0.1)
    detector = FailureDetector(registry=registry)

    failed_workers = []
    failed_jobs = []

    detector.register_worker_failure_callback(lambda w_id: failed_workers.append(w_id))
    detector.register_job_failure_callback(lambda j_id, reason: failed_jobs.append((j_id, reason)))

    w1 = _make_worker("w-timeout")
    registry.register(w1)

    spec = WorkloadSpec(
        name="job-timeout",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )
    job = Job(name="job-timeout", spec=spec)
    handle = ExecutionHandle(
        handle_id="h-1",
        job_id=job.id,
        backend_type="NATIVE",
        status=JobStatus.RUNNING,
    )

    detector.track_job(job, "w-timeout", handle, NativeProcessBackend())

    # Wait for timeout
    time.sleep(0.15)
    new_workers, new_jobs = detector.check_failures()

    assert "w-timeout" in new_workers
    assert "w-timeout" in failed_workers
    assert len(failed_jobs) == 1
    assert failed_jobs[0][0] == job.id
    assert "w-timeout" in failed_jobs[0][1]


def test_failure_detector_process_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        exec_backend = NativeProcessBackend(log_dir=tmpdir)
        registry = WorkerRegistry()
        detector = FailureDetector(registry=registry)

        failed_jobs = []
        detector.register_job_failure_callback(lambda j_id, reason: failed_jobs.append((j_id, reason)))

        w1 = _make_worker("w-healthy")
        registry.register(w1)

        spec = WorkloadSpec(
            name="failing-job",
            resources=ResourceRequirements(cpus=1),
            execution=WorkloadExecutionConfig(
                entrypoint=["python3", "-c", "import sys; sys.exit(1)"]
            ),
        )
        job = Job(name="failing-job", spec=spec)
        alloc = ResourceAllocation(
            job_id=job.id,
            worker_id="w-healthy",
            assigned_cpu_cores=[0],
        )

        handle = exec_backend.launch(job, alloc)
        detector.track_job(job, "w-healthy", handle, exec_backend)

        # Wait for process exit
        time.sleep(0.3)
        new_workers, new_jobs = detector.check_failures()

        assert len(new_jobs) == 1
        assert new_jobs[0][0] == job.id
        assert len(failed_jobs) == 1
        assert failed_jobs[0][0] == job.id


def test_failure_detector_background_loop():
    registry = WorkerRegistry(timeout_seconds=0.1)
    detector = FailureDetector(registry=registry, check_interval_seconds=0.05)

    failed_workers = []
    detector.register_worker_failure_callback(lambda w_id: failed_workers.append(w_id))

    w1 = _make_worker("w-loop-fail")
    registry.register(w1)

    detector.start()
    time.sleep(0.18)
    detector.stop()

    assert "w-loop-fail" in failed_workers
