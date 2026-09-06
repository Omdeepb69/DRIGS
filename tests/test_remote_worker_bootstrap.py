"""Unit and integration tests for remote worker bootstrap and outbound HTTP registration."""

import pytest
from fastapi.testclient import TestClient

from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import ComputeDevice, DeviceState, DeviceType, WorkerInfo
from drigs.workers.bootstrap import RemoteHTTPWorkerRegistryClient, create_remote_worker_agent
from drigs.workers.registry import WorkerRegistry


def test_api_server_worker_endpoints():
    """Test worker registration, heartbeat, and deregistration REST endpoints in APIServer."""
    registry = WorkerRegistry()
    controller = LocalController(auto_register_local_node=False)
    api_server = APIServer(controller=controller, worker_registry=registry)
    client = TestClient(api_server.app)

    # 1. Register worker
    device = ComputeDevice(
        device_id="gpu-remote-0",
        device_type=DeviceType.GPU,
        model_name="Tesla T4 (Kaggle)",
        total_memory_bytes=16 * 1024 * 1024 * 1024,
        available_memory_bytes=16 * 1024 * 1024 * 1024,
    )
    worker = WorkerInfo(
        worker_id="kaggle-worker-01",
        hostname="kaggle-notebook-node",
        ip_address="10.0.0.50",
        devices=[device],
        total_cpus=4,
        total_memory_bytes=16 * 1024 * 1024 * 1024,
        status=DeviceState.HEALTHY,
    )

    reg_res = client.post("/v1/workers/register", json=worker.model_dump(mode="json"))
    assert reg_res.status_code == 201
    assert reg_res.json()["worker_id"] == "kaggle-worker-01"

    # 2. Verify worker appears in list_workers endpoint
    list_res = client.get("/v1/workers")
    assert list_res.status_code == 200
    workers = list_res.json()
    assert len(workers) == 1
    assert workers[0]["worker_id"] == "kaggle-worker-01"
    assert workers[0]["devices"][0]["model_name"] == "Tesla T4 (Kaggle)"

    # 3. Heartbeat
    hb_res = client.post("/v1/workers/kaggle-worker-01/heartbeat", params={"status": "HEALTHY"})
    assert hb_res.status_code == 200
    assert hb_res.json()["status"] == "acknowledged"

    # 4. Deregister worker
    dereg_res = client.post("/v1/workers/kaggle-worker-01/deregister")
    assert dereg_res.status_code == 200
    assert dereg_res.json()["status"] == "deregistered"


def test_create_remote_worker_agent():
    """Test remote worker agent creation helper."""
    agent = create_remote_worker_agent(
        controller_url="http://localhost:8000",
        worker_id="remote-test-worker",
        use_gpu=False,
    )
    assert agent.worker_id == "remote-test-worker"
    info = agent.get_worker_info()
    assert info.worker_id == "remote-test-worker"
    assert len(info.devices) >= 1
