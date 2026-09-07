"""Unit and concurrency load tests for async non-blocking control plane event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from fastapi.testclient import TestClient

from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    JobStatus,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.workers.registry import WorkerRegistry


def test_concurrent_api_requests_under_load():
    """Test 50 concurrent REST API requests executing without blocking or request drops."""
    controller = LocalController(auto_register_local_node=False)
    worker_registry = WorkerRegistry()

    # Register worker
    device = ComputeDevice(
        device_id="gpu-0",
        model_name="NVIDIA A100",
        device_type=DeviceType.GPU,
        total_memory_bytes=40 * 1024**3,
        available_memory_bytes=40 * 1024**3,
        state=DeviceState.HEALTHY,
    )
    worker = WorkerInfo(
        worker_id="load-worker-1",
        hostname="node1.local",
        ip_address="192.168.1.10",
        devices=[device],
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    worker_registry.register(worker)
    controller.resource_manager.register_worker(worker)

    server = APIServer(controller=controller, worker_registry=worker_registry)
    client = TestClient(server.app)

    def submit_one(idx: int):
        payload = {
            "name": f"load-job-{idx}",
            "spec": {
                "name": f"load-job-{idx}",
                "resources": {"gpus": 0, "cpus": 1},
                "execution": {"entrypoint": ["echo", f"load test {idx}"]},
            },
            "priority": idx % 10,
        }
        res = client.post("/v1/jobs", json=payload)
        return res.status_code, res.json()

    def query_metrics():
        res = client.get("/metrics")
        return res.status_code, res.text

    def query_gpus():
        res = client.get("/v1/gpus")
        return res.status_code

    # Execute 50 concurrent submission, metrics, and GPU queries via ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=20) as executor:
        submit_futures = [executor.submit(submit_one, i) for i in range(30)]
        metrics_futures = [executor.submit(query_metrics) for _ in range(10)]
        gpu_futures = [executor.submit(query_gpus) for _ in range(10)]

        submit_results = [f.result() for f in submit_futures]
        metrics_results = [f.result() for f in metrics_futures]
        gpu_results = [f.result() for f in gpu_futures]

    for status_code, data in submit_results:
        assert status_code == 201
        assert data["status"] == "QUEUED"

    for status_code, text in metrics_results:
        assert status_code == 200
        assert "drigs_queued_jobs" in text

    for status_code in gpu_results:
        assert status_code == 200


def test_async_local_controller_lifecycle():
    """Test LocalController start_async, async_step, and stop_async non-blocking loop."""
    controller = LocalController(auto_register_local_node=True, poll_interval_seconds=0.05)

    async def _test():
        await controller.start_async()
        assert controller._async_running is True
        assert controller._async_task is not None

        # Submit job into async controller
        spec = WorkloadSpec(
            name="async-test-job",
            resources={"cpus": 1, "gpus": 0},
            execution=WorkloadExecutionConfig(entrypoint=["sleep", "0.1"]),
        )
        job = controller.submit_job(spec)

        # Wait for async loop to process step and finish job
        start_t = time.time()
        while time.time() - start_t < 3.0:
            status = controller.get_job_status(job.id)
            if status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.05)

        assert controller.get_job_status(job.id) == JobStatus.COMPLETED

        await controller.stop_async()
        assert controller._async_running is False
        assert controller._async_task is None

    asyncio.run(_test())
