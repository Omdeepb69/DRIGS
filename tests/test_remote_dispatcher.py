"""Unit tests for RemoteDispatcher in drigs.workers.dispatcher."""

import asyncio
import tempfile
import time
import pytest

from drigs.core.models import (
    DeviceState,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
)
from drigs.execution.native import NativeProcessBackend
from drigs.hardware.simulated import SimulatedBackend
from drigs.workers.agent import WorkerAgent
from drigs.workers.dispatcher import RemoteDispatcher
from drigs.workers.registry import WorkerRegistry


def test_remote_dispatcher_register_agent():
    registry = WorkerRegistry()
    dispatcher = RemoteDispatcher(registry=registry)

    hw1 = SimulatedBackend(num_gpus=1)
    agent1 = WorkerAgent(worker_id="w-node-1", hardware_backend=hw1)

    hw2 = SimulatedBackend(num_gpus=2)
    agent2 = WorkerAgent(worker_id="w-node-2", hardware_backend=hw2)

    assert dispatcher.register_agent(agent1) is True
    assert dispatcher.register_agent(agent2) is True

    assert dispatcher.get_agent("w-node-1") is agent1
    assert dispatcher.get_agent("w-node-2") is agent2

    active_ids = dispatcher.get_active_worker_ids()
    assert "w-node-1" in active_ids
    assert "w-node-2" in active_ids


def test_remote_dispatcher_job_dispatch_and_execution():
    def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = WorkerRegistry()
            dispatcher = RemoteDispatcher(registry=registry)

            exec_backend = NativeProcessBackend(log_dir=tmpdir)
            agent = WorkerAgent(
                worker_id="worker-node-alpha",
                execution_backend=exec_backend,
            )
            dispatcher.register_agent(agent)

            spec = WorkloadSpec(
                name="remote-job",
                resources=ResourceRequirements(cpus=1),
                execution=WorkloadExecutionConfig(
                    entrypoint=["echo", "hello from remote dispatcher"]
                ),
            )
            job = Job(name="job-alpha", spec=spec)
            alloc = ResourceAllocation(
                job_id=job.id,
                worker_id="worker-node-alpha",
                assigned_cpu_cores=[0],
            )

            handle = dispatcher.dispatch(job, alloc)
            assert handle is not None
            assert handle.job_id == job.id

            dispatched = dispatcher.get_dispatched_jobs()
            assert dispatched.get(job.id) == "worker-node-alpha"

            time.sleep(0.3)
            assert dispatcher.get_job_status(job.id) == JobStatus.COMPLETED

            async def _collect_logs():
                logs = []
                async for line in dispatcher.stream_job_logs(job.id):
                    logs.append(line)
                return logs

            logs = asyncio.run(_collect_logs())
            assert any("hello from remote dispatcher" in l for l in logs)

    _run()


def test_remote_dispatcher_unregistered_worker():
    registry = WorkerRegistry()
    dispatcher = RemoteDispatcher(registry=registry)

    spec = WorkloadSpec(
        name="job-missing",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "missing"]),
    )
    job = Job(name="job-missing", spec=spec)
    alloc = ResourceAllocation(
        job_id=job.id,
        worker_id="worker-nonexistent",
    )

    with pytest.raises(ValueError, match="not registered"):
        dispatcher.dispatch(job, alloc)


def test_remote_dispatcher_offline_worker():
    registry = WorkerRegistry(timeout_seconds=0.1)
    dispatcher = RemoteDispatcher(registry=registry)

    agent = WorkerAgent(worker_id="worker-offline-test")
    dispatcher.register_agent(agent)

    # Force worker to expire / timeout
    time.sleep(0.15)
    registry.check_timeouts()

    spec = WorkloadSpec(
        name="job-offline",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "offline"]),
    )
    job = Job(name="job-offline", spec=spec)
    alloc = ResourceAllocation(
        job_id=job.id,
        worker_id="worker-offline-test",
    )

    with pytest.raises(RuntimeError, match="Worker worker-offline-test is currently OFFLINE"):
        dispatcher.dispatch(job, alloc)


def test_remote_dispatcher_stop_job():
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = WorkerRegistry()
        dispatcher = RemoteDispatcher(registry=registry)

        exec_backend = NativeProcessBackend(log_dir=tmpdir)
        agent = WorkerAgent(
            worker_id="worker-stop-test",
            execution_backend=exec_backend,
        )
        dispatcher.register_agent(agent)

        spec = WorkloadSpec(
            name="job-long",
            resources=ResourceRequirements(cpus=1),
            execution=WorkloadExecutionConfig(
                entrypoint=["python3", "-c", "import time; time.sleep(30)"]
            ),
        )
        job = Job(name="job-long", spec=spec)
        alloc = ResourceAllocation(
            job_id=job.id,
            worker_id="worker-stop-test",
            assigned_cpu_cores=[0],
        )

        handle = dispatcher.dispatch(job, alloc)
        assert dispatcher.get_job_status(job.id) == JobStatus.RUNNING

        stopped = dispatcher.stop_job(job.id)
        assert stopped is True
        assert dispatcher.get_job_status(job.id) is None
