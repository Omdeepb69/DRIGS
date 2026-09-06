"""Unit tests for NativeProcessBackend execution module."""

import asyncio
import sys
import tempfile
import time
import pytest

from drigs.core.interfaces import ExecutionBackend
from drigs.core.models import (
    ExecutionBackendType,
    Job,
    JobStatus,
    ResourceAllocation,
    WorkloadExecutionConfig,
    WorkloadSpec,
)
from drigs.execution.native import NativeProcessBackend


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def native_backend(temp_log_dir):
    return NativeProcessBackend(log_dir=temp_log_dir)


def create_test_job(
    entrypoint,
    env: dict = None,
    args: list = None,
) -> Job:
    spec = WorkloadSpec(
        name="test-native-job",
        execution=WorkloadExecutionConfig(
            backend=ExecutionBackendType.NATIVE,
            entrypoint=entrypoint,
            env=env or {},
            args=args or [],
        ),
    )
    return Job(name="test-native-job", spec=spec)


def create_test_allocation(
    job_id: str,
    gpu_indices: list = None,
) -> ResourceAllocation:
    device_ids = [f"gpu-{idx}" for idx in (gpu_indices or [])]
    return ResourceAllocation(
        job_id=job_id,
        worker_id="worker-0",
        assigned_device_ids=device_ids,
    )


def test_protocol_compliance(native_backend):
    assert isinstance(native_backend, ExecutionBackend)


def test_launch_and_completion(native_backend):
    job = create_test_job(entrypoint=[sys.executable, "-c", "print('hello from native backend')"])
    alloc = create_test_allocation(job.id)

    handle = native_backend.launch(job, alloc)
    assert handle.handle_id.startswith(f"exec_{job.id}")
    assert handle.pid is not None

    for _ in range(50):
        status = native_backend.get_status(handle)
        if status != JobStatus.RUNNING:
            break
        time.sleep(0.1)

    assert native_backend.get_status(handle) == JobStatus.COMPLETED

    logs = native_backend.read_logs(handle)
    assert "hello from native backend" in logs


def test_environment_variables(native_backend):
    code = (
        "import os\n"
        "print('CUDA:', os.environ.get('CUDA_VISIBLE_DEVICES'))\n"
        "print('CUSTOM:', os.environ.get('MY_VAR'))\n"
        "print('JOB_ID:', os.environ.get('DRIGS_JOB_ID'))\n"
    )
    job = create_test_job(
        entrypoint=[sys.executable, "-c", code],
        env={"MY_VAR": "custom_val"},
    )
    alloc = create_test_allocation(job.id, gpu_indices=[1, 3])

    handle = native_backend.launch(job, alloc)

    for _ in range(50):
        if native_backend.get_status(handle) != JobStatus.RUNNING:
            break
        time.sleep(0.1)

    assert native_backend.get_status(handle) == JobStatus.COMPLETED
    logs = native_backend.read_logs(handle)
    assert "CUDA: 1,3" in logs
    assert "CUSTOM: custom_val" in logs
    assert f"JOB_ID: {job.id}" in logs


def test_stop_running_job(native_backend):
    code = "import time; time.sleep(10)"
    job = create_test_job(entrypoint=[sys.executable, "-c", code])
    alloc = create_test_allocation(job.id)

    handle = native_backend.launch(job, alloc)
    assert native_backend.get_status(handle) == JobStatus.RUNNING

    stopped = native_backend.stop(handle)
    assert stopped is True
    assert native_backend.get_status(handle) == JobStatus.CANCELLED


def test_failed_command(native_backend):
    job = create_test_job(entrypoint=[sys.executable, "-c", "import sys; sys.exit(42)"])
    alloc = create_test_allocation(job.id)

    handle = native_backend.launch(job, alloc)

    for _ in range(50):
        if native_backend.get_status(handle) != JobStatus.RUNNING:
            break
        time.sleep(0.1)

    assert native_backend.get_status(handle) == JobStatus.FAILED


def test_stream_logs(native_backend):
    code = "import time; print('line 1', flush=True); time.sleep(0.1); print('line 2', flush=True)"
    job = create_test_job(entrypoint=[sys.executable, "-c", code])
    alloc = create_test_allocation(job.id)

    handle = native_backend.launch(job, alloc)

    async def _stream():
        lines = []
        async for line in native_backend.stream_logs(handle):
            lines.append(line.strip())
        return lines

    received_lines = asyncio.run(_stream())
    assert "line 1" in received_lines
    assert "line 2" in received_lines
