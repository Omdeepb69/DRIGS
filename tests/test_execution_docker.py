"""Unit tests for DockerBackend in drigs.execution.docker."""

import asyncio
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadRuntimeConfig,
    WorkloadSpec,
)
from drigs.execution.docker import DockerBackend


def _make_sample_job_and_allocation():
    runtime = WorkloadRuntimeConfig(
        container_image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
        volumes={"/host/data": "/container/data"},
    )
    spec = WorkloadSpec(
        name="docker-test-job",
        resources=ResourceRequirements(
            cpus=4,
            gpus=2,
            memory_bytes=8 * 1024**3,
            gpu_memory_bytes=4 * 1024**3,
        ),
        execution=WorkloadExecutionConfig(
            entrypoint=["python3", "train.py", "--batch-size", "32"],
            working_dir="/app",
            env={"MODEL_NAME": "resnet50"},
        ),
        runtime=runtime,
    )

    job = Job(name="docker-test-job", spec=spec)

    gpu1 = ComputeDevice(
        device_id="gpu-0",
        model_name="Tesla T4",
        device_type=DeviceType.GPU,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        state=DeviceState.HEALTHY,
        pcie_bus_id="0000:00:04.0",
    )
    gpu2 = ComputeDevice(
        device_id="gpu-1",
        model_name="Tesla T4",
        device_type=DeviceType.GPU,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        state=DeviceState.HEALTHY,
        pcie_bus_id="0000:00:05.0",
    )

    alloc = ResourceAllocation(
        job_id=job.id,
        worker_id="w-node1",
        assigned_device_ids=["gpu-0", "gpu-1"],
        assigned_cpu_cores=[0, 1, 2, 3],
        allocated_devices=[gpu1, gpu2],
    )
    return job, alloc


def test_docker_command_building():
    job, alloc = _make_sample_job_and_allocation()
    backend = DockerBackend(docker_cmd="docker")

    cmd = backend.build_docker_cmd(job, alloc, "test_container_123")

    assert cmd[0] == "docker"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert cmd[cmd.index("--name") + 1] == "test_container_123"

    # GPU passthrough check
    assert "--gpus" in cmd
    gpus_val = cmd[cmd.index("--gpus") + 1]
    assert "device=0,1" in gpus_val

    # CPU & Memory limits
    assert "--cpus" in cmd
    assert cmd[cmd.index("--cpus") + 1] == "4"
    assert "--memory" in cmd
    assert cmd[cmd.index("--memory") + 1] == f"{8 * 1024**3}b"

    # Environment variables
    assert "-e" in cmd
    assert f"DRIGS_JOB_ID={job.id}" in cmd
    assert "MODEL_NAME=resnet50" in cmd

    # Volumes
    assert "-v" in cmd
    assert "/host/data:/container/data:rw" in cmd

    # Image & Entrypoint
    assert "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime" in cmd
    assert "python3" in cmd
    assert "train.py" in cmd
    assert "--batch-size" in cmd


def test_docker_fallback_to_native():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DockerBackend(
            docker_cmd="non_existent_docker_bin_12345",
            fallback_to_native=True,
            log_dir=tmpdir,
        )

        assert not backend.is_docker_available()

        spec = WorkloadSpec(
            name="fallback-job",
            resources=ResourceRequirements(cpus=1),
            execution=WorkloadExecutionConfig(entrypoint=["echo", "fallback_success"]),
        )
        job = Job(name="fallback-job", spec=spec)
        alloc = ResourceAllocation(
            job_id=job.id,
            worker_id="w-local",
            assigned_cpu_cores=[0],
        )

        handle = backend.launch(job, alloc)
        assert handle.backend_type == "native"
        assert handle.status == JobStatus.RUNNING

        import time
        time.sleep(0.3)
        status = backend.get_status(handle)
        assert status == JobStatus.COMPLETED


def test_docker_mocked_launch_and_stop():
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = DockerBackend(log_dir=tmpdir)
        job, alloc = _make_sample_job_and_allocation()

        mock_popen = MagicMock()
        mock_popen.pid = 9999
        mock_popen.poll.return_value = None

        with patch.object(backend, "is_docker_available", return_value=True), \
             patch("subprocess.Popen", return_value=mock_popen), \
             patch("subprocess.run") as mock_run:

            handle = backend.launch(job, alloc)
            assert handle.backend_type == "docker"
            assert handle.pid == 9999
            assert handle.status == JobStatus.RUNNING
            assert handle.metadata["container_name"].startswith("drigs_container_")

            status = backend.get_status(handle)
            assert status == JobStatus.RUNNING

            stopped = backend.stop(handle)
            assert stopped is True
            assert backend.get_status(handle) == JobStatus.CANCELLED
            mock_run.assert_called_once()
