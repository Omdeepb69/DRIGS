"""Integration unit tests for LocalController in drigs.core.controller."""

import time
import pytest
import tempfile
from pathlib import Path

from drigs.core.controller import LocalController
from drigs.core.models import (
    JobStatus,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
)
from drigs.execution.native import NativeProcessBackend
from drigs.hardware.simulated import SimulatedBackend
from drigs.scheduler.fifo import FIFOScheduler


def _make_fast_spec(name: str = "fast-job") -> WorkloadSpec:
    return WorkloadSpec(
        name=name,
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(
            entrypoint=["python3", "-c", "import time; time.sleep(0.1); print('done')"]
        ),
    )


def test_local_controller_submit_and_step():
    controller = LocalController(auto_register_local_node=True)
    spec = _make_fast_spec("step-test")

    job = controller.submit_job(spec)
    assert job.status == JobStatus.QUEUED

    # First step should schedule and launch
    active_count = controller.step()
    assert active_count == 1
    assert controller.get_job_status(job.id) == JobStatus.RUNNING

    # Wait briefly for command to complete
    time.sleep(0.3)

    # Second step should detect completion and release resources
    active_count = controller.step()
    assert active_count == 0
    assert controller.get_job_status(job.id) == JobStatus.COMPLETED


def test_local_controller_background_loop():
    controller = LocalController(poll_interval_seconds=0.1, auto_register_local_node=True)
    spec = _make_fast_spec("loop-test")

    job = controller.submit_job(spec)
    controller.start()

    # Wait for completion via background loop
    start_time = time.time()
    while time.time() - start_time < 5.0:
        if controller.get_job_status(job.id) == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    assert controller.get_job_status(job.id) == JobStatus.COMPLETED
    controller.stop()


def test_local_controller_cancel_running_job():
    controller = LocalController(auto_register_local_node=True)
    long_spec = WorkloadSpec(
        name="long-job",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(
            entrypoint=["python3", "-c", "import time; time.sleep(30)"]
        ),
    )

    job = controller.submit_job(long_spec)
    controller.step()
    assert controller.get_job_status(job.id) == JobStatus.RUNNING

    # Cancel running job
    success = controller.cancel_job(job.id)
    assert success is True
    assert controller.get_job_status(job.id) == JobStatus.CANCELLED

    # Check resources deallocated
    assert len(controller._handles) == 0
    assert len(controller._allocations) == 0


def test_local_controller_yaml_submission():
    yaml_spec = """
name: yaml-job
resources:
  cpus: 1
execution:
  backend: NATIVE
  entrypoint: echo 'hello from yaml'
"""
    controller = LocalController(auto_register_local_node=True)
    job = controller.submit_job(yaml_spec)
    assert job.name == "yaml-job"
    assert job.status == JobStatus.QUEUED

    controller.step()
    assert controller.get_job_status(job.id) == JobStatus.RUNNING
    time.sleep(0.3)
    controller.step()
    assert controller.get_job_status(job.id) == JobStatus.COMPLETED
