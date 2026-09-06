"""Unit tests for CheckpointManager and Rescheduler in drigs.recovery."""

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
from drigs.core.queue import JobQueue, AdmissionController
from drigs.core.resource_manager import ResourceManager
from drigs.execution.native import NativeProcessBackend
from drigs.recovery.checkpoint import CheckpointManager
from drigs.recovery.detector import FailureDetector
from drigs.recovery.rescheduler import Rescheduler
from drigs.workers.registry import WorkerRegistry


def _make_worker(worker_id: str = "w-1") -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        hostname="node-1",
        ip_address="127.0.0.1",
        devices=[
            ComputeDevice(
                device_id=f"{worker_id}-gpu0",
                model_name="Tesla T4",
                device_type=DeviceType.GPU,
                total_memory_bytes=16 * 1024**3,
                available_memory_bytes=16 * 1024**3,
                state=DeviceState.HEALTHY,
            )
        ],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )


def test_checkpoint_manager_save_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(storage_dir=tmpdir)
        job_id = "job-101"

        path1 = ckpt_mgr.save_checkpoint(job_id, step=10, checkpoint_data={"loss": 0.5})
        path2 = ckpt_mgr.save_checkpoint(job_id, step=30, checkpoint_data={"loss": 0.2})
        path3 = ckpt_mgr.save_checkpoint(job_id, step=20, checkpoint_data={"loss": 0.3})

        manifests = ckpt_mgr.list_checkpoints(job_id)
        assert len(manifests) == 3
        assert [m["step"] for m in manifests] == [10, 20, 30]

        latest = ckpt_mgr.get_latest_checkpoint(job_id)
        assert latest is not None
        assert latest["step"] == 30
        assert latest["data"]["loss"] == 0.2
        assert latest["filepath"] == path2


def test_checkpoint_manager_cleanup():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(storage_dir=tmpdir)
        job_id = "job-cleanup"

        for step in range(1, 6):
            ckpt_mgr.save_checkpoint(job_id, step=step, checkpoint_data={"step": step})

        assert len(ckpt_mgr.list_checkpoints(job_id)) == 5

        deleted = ckpt_mgr.cleanup_checkpoints(job_id, keep_last_n=2)
        assert deleted == 3

        remaining = ckpt_mgr.list_checkpoints(job_id)
        assert len(remaining) == 2
        assert [m["step"] for m in remaining] == [4, 5]


def test_rescheduler_job_recovery_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        resource_mgr = ResourceManager()
        worker = _make_worker("w-1")
        resource_mgr.register_worker(worker)

        job_queue = JobQueue()

        ckpt_mgr = CheckpointManager(storage_dir=tmpdir)
        rescheduler = Rescheduler(
            job_queue=job_queue,
            resource_manager=resource_mgr,
            checkpoint_manager=ckpt_mgr,
        )

        spec = WorkloadSpec(
            name="train-job",
            resources=ResourceRequirements(cpus=2, gpus=1),
            execution=WorkloadExecutionConfig(entrypoint=["python3", "train.py"]),
        )
        job = Job(name="train-job", spec=spec)

        job_queue.enqueue(job)
        job_queue.update_job_status(job.id, JobStatus.SCHEDULED)
        job_queue.update_job_status(job.id, JobStatus.RUNNING)

        # Allocate resources
        alloc = resource_mgr.allocate_resources(
            job_id=job.id,
            worker_id="w-1",
            device_ids=["w-1-gpu0"],
            cpus=2,
            memory_bytes=4 * 1024**3,
        )
        assert alloc is not None
        job.allocation = alloc

        # Save checkpoint
        ckpt_path = ckpt_mgr.save_checkpoint(job.id, step=50, checkpoint_data={"epoch": 5})

        # Reschedule job
        success = rescheduler.reschedule_job(job.id, reason="GPU failure")
        assert success is True

        updated_job = job_queue.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == JobStatus.QUEUED
        assert updated_job.allocation is None
        assert updated_job.spec.execution.env.get("DRIGS_RESTORE_CHECKPOINT") == ckpt_path
        assert updated_job.spec.execution.env.get("DRIGS_RESTORE_STEP") == "50"


def test_rescheduler_failure_detector_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = WorkerRegistry(timeout_seconds=0.1)
        detector = FailureDetector(registry=registry)

        resource_mgr = ResourceManager()
        w1 = _make_worker("w-fail-integration")
        registry.register(w1)
        resource_mgr.register_worker(w1)

        job_queue = JobQueue()
        ckpt_mgr = CheckpointManager(storage_dir=tmpdir)
        rescheduler = Rescheduler(
            job_queue=job_queue,
            resource_manager=resource_mgr,
            checkpoint_manager=ckpt_mgr,
        )
        rescheduler.attach_failure_detector(detector)

        spec = WorkloadSpec(
            name="integration-job",
            resources=ResourceRequirements(cpus=1),
            execution=WorkloadExecutionConfig(entrypoint=["echo", "hi"]),
        )
        job = Job(name="integration-job", spec=spec)
        job_queue.enqueue(job)
        job_queue.update_job_status(job.id, JobStatus.SCHEDULED)
        job_queue.update_job_status(job.id, JobStatus.RUNNING)

        alloc = resource_mgr.allocate_resources(
            job_id=job.id,
            worker_id="w-fail-integration",
            device_ids=[],
            cpus=1,
            memory_bytes=1024**3,
        )
        job.allocation = alloc

        handle = ExecutionHandle(
            handle_id="h-integ",
            job_id=job.id,
            backend_type="NATIVE",
            status=JobStatus.RUNNING,
        )
        detector.track_job(job, "w-fail-integration", handle, NativeProcessBackend())

        # Wait for worker timeout
        time.sleep(0.15)
        detector.check_failures()

        recovered_job = job_queue.get_job(job.id)
        assert recovered_job is not None
        assert recovered_job.status == JobStatus.QUEUED
        assert recovered_job.allocation is None
