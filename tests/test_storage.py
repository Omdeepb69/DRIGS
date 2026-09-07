"""Unit tests for SQLite Control Plane Persistence Engine and Crash Recovery."""

from pathlib import Path
import tempfile
import pytest

from drigs.core.controller import LocalController
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
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.core.storage import SQLiteStore
from drigs.workers.registry import WorkerRegistry


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "test_drigs.db"


@pytest.fixture
def sample_worker():
    gpu = ComputeDevice(
        device_id="gpu-0",
        model_name="NVIDIA Tesla T4",
        device_type=DeviceType.GPU,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        state=DeviceState.HEALTHY,
    )
    return WorkerInfo(
        worker_id="worker-node-1",
        hostname="node1.local",
        ip_address="192.168.1.10",
        devices=[gpu],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )


@pytest.fixture
def sample_job():
    spec = WorkloadSpec(
        name="test-workload",
        resources=ResourceRequirements(gpus=1, cpus=2, memory_bytes=4 * 1024**3),
        execution=WorkloadExecutionConfig(entrypoint="python3 train.py"),
    )
    return Job(name="test-job", priority=10, spec=spec)


def test_sqlite_store_crud_jobs(temp_db_path, sample_job):
    store = SQLiteStore(temp_db_path)

    # Save job
    store.save_job(sample_job)
    retrieved = store.get_job(sample_job.id)
    assert retrieved is not None
    assert retrieved.id == sample_job.id
    assert retrieved.name == sample_job.name
    assert retrieved.priority == sample_job.priority
    assert retrieved.status == sample_job.status

    # Update job status
    sample_job.status = JobStatus.RUNNING
    store.save_job(sample_job)
    updated = store.get_job(sample_job.id)
    assert updated.status == JobStatus.RUNNING

    # Load all jobs
    jobs = store.load_all_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == sample_job.id

    # Delete job
    assert store.delete_job(sample_job.id) is True
    assert store.get_job(sample_job.id) is None


def test_sqlite_store_crud_workers(temp_db_path, sample_worker):
    store = SQLiteStore(temp_db_path)

    store.save_worker(sample_worker)
    workers = store.load_all_workers()
    assert len(workers) == 1
    assert workers[0].worker_id == sample_worker.worker_id
    assert workers[0].hostname == sample_worker.hostname
    assert len(workers[0].devices) == 1

    assert store.delete_worker(sample_worker.worker_id) is True
    assert len(store.load_all_workers()) == 0


def test_sqlite_store_crud_allocations(temp_db_path):
    store = SQLiteStore(temp_db_path)
    alloc = ResourceAllocation(
        job_id="job-100",
        worker_id="worker-node-1",
        assigned_device_ids=["gpu-0"],
        assigned_cpu_cores=[0, 1],
        memory_bytes=4 * 1024**3,
    )

    store.save_allocation(alloc)
    allocs = store.load_all_allocations()
    assert len(allocs) == 1
    assert allocs[0].job_id == "job-100"
    assert allocs[0].assigned_device_ids == ["gpu-0"]

    assert store.delete_allocation("job-100") is True
    assert len(store.load_all_allocations()) == 0


def test_job_queue_persistence_and_recovery(temp_db_path, sample_job):
    store = SQLiteStore(temp_db_path)
    jq1 = JobQueue(storage_backend=store)
    jq1.enqueue(sample_job)
    jq1.update_job_status(sample_job.id, JobStatus.SCHEDULED)

    # Instantiate new queue simulating restart
    jq2 = JobQueue(storage_backend=store)
    restored_count = jq2.load_from_storage()
    assert restored_count == 1
    restored_job = jq2.get_job(sample_job.id)
    assert restored_job is not None
    assert restored_job.status == JobStatus.SCHEDULED


def test_control_plane_crash_recovery_simulation(temp_db_path, sample_worker, sample_job):
    # Phase 1: Controller 1 starts, registers worker, submits job, and allocates resources
    store1 = SQLiteStore(temp_db_path)
    controller1 = LocalController(storage_backend=store1, auto_register_local_node=False)

    controller1.resource_manager.register_worker(sample_worker)
    admitted = controller1.submit_job(sample_job.spec, priority=20, job_name="crash-test-job")
    alloc = controller1.resource_manager.allocate_resources(
        job_id=admitted.id,
        worker_id=sample_worker.worker_id,
        device_ids=["gpu-0"],
        cpus=2,
        memory_bytes=4 * 1024**3,
    )
    controller1.job_queue.update_job_status(admitted.id, JobStatus.SCHEDULED)
    controller1.job_queue.update_job_status(admitted.id, JobStatus.RUNNING)

    # Phase 2: Simulate process crash by destroying Controller 1
    del controller1
    del store1

    # Phase 3: Controller 2 starts up pointing to the same SQLite database file
    store2 = SQLiteStore(temp_db_path)
    controller2 = LocalController(storage_backend=store2, auto_register_local_node=False)

    # Verify state restoration
    cluster_state = controller2.resource_manager.get_cluster_state()
    assert sample_worker.worker_id in cluster_state.workers
    restored_job = controller2.job_queue.get_job(admitted.id)
    assert restored_job is not None
    assert restored_job.status == JobStatus.RUNNING

    restored_alloc = controller2.resource_manager.get_allocation(admitted.id)
    assert restored_alloc is not None
    assert restored_alloc.worker_id == sample_worker.worker_id
    assert restored_alloc.assigned_device_ids == ["gpu-0"]
