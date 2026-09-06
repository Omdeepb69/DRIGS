"""Unit tests for FastAPI REST Control Plane API in drigs.api.server."""

import pytest
from fastapi.testclient import TestClient

from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    JobStatus,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.workers.registry import WorkerRegistry


@pytest.fixture
def api_client():
    controller = LocalController()
    worker_registry = WorkerRegistry()

    # Register worker and GPU device
    gpu = ComputeDevice(
        device_id="gpu-0",
        model_name="NVIDIA A100",
        device_type=DeviceType.GPU,
        total_memory_bytes=40 * 1024**3,
        available_memory_bytes=40 * 1024**3,
        state=DeviceState.HEALTHY,
    )
    worker = WorkerInfo(
        worker_id="worker-node-1",
        hostname="node1.local",
        ip_address="192.168.1.10",
        devices=[gpu],
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    worker_registry.register(worker)
    controller.resource_manager.register_worker(worker)

    server = APIServer(controller=controller, worker_registry=worker_registry)
    client = TestClient(server.app)
    return client


def test_health_check_endpoint(api_client):
    res = api_client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "drigs-control-plane"


def test_job_submission_and_lifecycle(api_client):
    payload = {
        "name": "api-test-job",
        "spec": {
            "name": "api-test-job",
            "resources": {
                "gpus": 1,
                "cpus": 2,
                "gpu_memory_bytes": 4096,
                "memory_bytes": 8192,
            },
            "execution": {
                "entrypoint": ["python3", "-c", "print('hello from api')"],
            },
        },
        "priority": 10,
    }

    # 1. Submit job
    res = api_client.post("/v1/jobs", json=payload)
    assert res.status_code == 201
    resp_data = res.json()
    assert "job_id" in resp_data
    job_id = resp_data["job_id"]
    assert resp_data["status"] in ("QUEUED", "PENDING", "RUNNING")

    # 2. List jobs
    list_res = api_client.get("/v1/jobs")
    assert list_res.status_code == 200
    jobs_list = list_res.json()
    assert len(jobs_list) >= 1
    assert any(j["job_id"] == job_id for j in jobs_list)

    # 3. Get job detail
    detail_res = api_client.get(f"/v1/jobs/{job_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["job_id"] == job_id
    assert detail_data["name"] == "api-test-job"

    # 4. Cancel job
    cancel_res = api_client.post(f"/v1/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.json()
    assert cancel_data["job_id"] == job_id
    assert cancel_data["status"] == "CANCELLED"


def test_get_nonexistent_job(api_client):
    res = api_client.get("/v1/jobs/nonexistent-id-999")
    assert res.status_code == 404


def test_list_workers_and_gpus(api_client):
    # List workers
    w_res = api_client.get("/v1/workers")
    assert w_res.status_code == 200
    workers = w_res.json()
    assert len(workers) == 1
    assert workers[0]["worker_id"] == "worker-node-1"

    # List GPUs
    gpu_res = api_client.get("/v1/gpus")
    assert gpu_res.status_code == 200
    gpus = gpu_res.json()
    assert len(gpus) == 1
    assert gpus[0]["device_id"] == "gpu-0"
    assert gpus[0]["model_name"] == "NVIDIA A100"


def test_get_cluster_state(api_client):
    cluster_res = api_client.get("/v1/cluster")
    assert cluster_res.status_code == 200
    state = cluster_res.json()
    assert "workers" in state or "active_jobs" in state or "updated_at" in state
