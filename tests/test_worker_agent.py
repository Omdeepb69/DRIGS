"""Unit tests for WorkerAgent in drigs.workers.agent."""

import asyncio
import tempfile
import time
import pytest

from drigs.core.interfaces import WorkerRegistryProtocol
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.hardware.simulated import SimulatedBackend
from drigs.execution.native import NativeProcessBackend
from drigs.workers.agent import WorkerAgent


class DummyWorkerRegistry:
    """In-memory dummy registry implementing WorkerRegistryProtocol for testing."""

    def __init__(self):
        self.registered_workers = {}
        self.heartbeats = {}

    def register(self, worker: WorkerInfo) -> bool:
        self.registered_workers[worker.worker_id] = worker
        return True

    def heartbeat(self, worker_id: str, status: DeviceState) -> bool:
        self.heartbeats[worker_id] = (status, time.time())
        return True

    def get_active_workers(self):
        return list(self.registered_workers.values())

    def get_worker(self, worker_id: str):
        return self.registered_workers.get(worker_id)


def test_worker_agent_initialization():
    hardware = SimulatedBackend(num_gpus=2, vram_per_gpu_bytes=16 * 1024**3)
    agent = WorkerAgent(
        worker_id="test-node-1",
        hostname="node-1.local",
        ip_address="192.168.1.50",
        hardware_backend=hardware,
    )

    assert agent.worker_id == "test-node-1"
    assert agent.hostname == "node-1.local"
    assert agent.ip_address == "192.168.1.50"

    devices = agent.discover_devices()
    assert len(devices) == 2
    assert devices[0].device_id == "gpu-0"

    info = agent.get_worker_info()
    assert isinstance(info, WorkerInfo)
    assert info.worker_id == "test-node-1"
    assert info.hostname == "node-1.local"
    assert info.status == DeviceState.HEALTHY
    assert len(info.devices) == 2


def test_worker_agent_registration_and_heartbeat():
    registry = DummyWorkerRegistry()
    hardware = SimulatedBackend(num_gpus=1)
    agent = WorkerAgent(
        worker_id="worker-reg-test",
        hardware_backend=hardware,
        registry=registry,
    )

    assert agent.register() is True
    assert "worker-reg-test" in registry.registered_workers
    registered_info = registry.get_worker("worker-reg-test")
    assert registered_info is not None
    assert registered_info.worker_id == "worker-reg-test"

    assert agent.send_heartbeat() is True
    assert "worker-reg-test" in registry.heartbeats
    status, _ = registry.heartbeats["worker-reg-test"]
    assert status == DeviceState.HEALTHY


def test_worker_agent_heartbeat_loop():
    async def _run():
        registry = DummyWorkerRegistry()
        agent = WorkerAgent(
            worker_id="worker-loop-test",
            registry=registry,
            heartbeat_interval=0.05,
        )

        await agent.start()
        assert agent.heartbeat_count >= 1
        await asyncio.sleep(0.18)
        assert agent.heartbeat_count >= 3
        await agent.stop()

        count_at_stop = agent.heartbeat_count
        await asyncio.sleep(0.10)
        assert agent.heartbeat_count == count_at_stop

    asyncio.run(_run())


def test_worker_agent_job_lifecycle():
    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            exec_backend = NativeProcessBackend(log_dir=tmpdir)
            agent = WorkerAgent(
                worker_id="worker-job-test",
                execution_backend=exec_backend,
            )

            spec = WorkloadSpec(
                name="worker-job",
                resources=ResourceRequirements(cpus=1),
                execution=WorkloadExecutionConfig(
                    entrypoint=["echo", "worker agent job execution test"]
                ),
            )
            job = Job(name="job-1", spec=spec)
            alloc = ResourceAllocation(
                job_id=job.id,
                worker_id=agent.worker_id,
                assigned_cpu_cores=[0],
            )

            handle = agent.launch_job(job, alloc)
            assert handle is not None
            assert handle.job_id == job.id

            for _ in range(20):
                status = agent.get_job_status(job.id)
                if status == JobStatus.COMPLETED:
                    break
                await asyncio.sleep(0.05)

            assert agent.get_job_status(job.id) == JobStatus.COMPLETED

            logs = []
            async for line in agent.stream_job_logs(job.id):
                logs.append(line)

            assert any("worker agent job execution test" in l for l in logs)

    asyncio.run(_run())


def test_worker_agent_async_context_manager():
    async def _run():
        registry = DummyWorkerRegistry()
        async with WorkerAgent(
            worker_id="context-worker",
            registry=registry,
            heartbeat_interval=0.05,
        ) as agent:
            assert agent.worker_id == "context-worker"
            assert agent.heartbeat_count >= 1
            await asyncio.sleep(0.12)
            assert agent.heartbeat_count >= 2

        final_count = agent.heartbeat_count
        await asyncio.sleep(0.10)
        assert agent.heartbeat_count == final_count

    asyncio.run(_run())
