"""Unit and integration tests for REST API security, API Key, and HMAC Bearer Token auth."""

import time
import pytest
from fastapi.testclient import TestClient

from drigs.api.auth import APIKeyAuth, create_hmac_token, verify_bearer_token
from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import ComputeDevice, DeviceState, DeviceType, WorkerInfo, WorkloadSpec
from drigs.workers.bootstrap import RemoteHTTPWorkerRegistryClient
from drigs.workers.registry import WorkerRegistry


def test_verify_bearer_token_logic():
    secret = "my-test-secret-key-123"
    
    # 1. Empty expected key allows everything
    assert verify_bearer_token("anything", "") is True
    assert verify_bearer_token("", "") is True

    # 2. Static key direct comparison
    assert verify_bearer_token(secret, secret) is True
    assert verify_bearer_token("wrong-key", secret) is False
    assert verify_bearer_token("", secret) is False

    # 3. HMAC token verification
    valid_token = create_hmac_token(secret)
    assert verify_bearer_token(valid_token, secret) is True
    assert verify_bearer_token(valid_token, "wrong-secret") is False

    # 4. Tampered HMAC signature
    ts, _ = valid_token.split(".")
    tampered_token = f"{ts}.abcdef0123456789"
    assert verify_bearer_token(tampered_token, secret) is False

    # 5. Expired HMAC token (> 24 hours old)
    old_ts = time.time() - 90000
    expired_token = create_hmac_token(secret, timestamp=old_ts)
    assert verify_bearer_token(expired_token, secret) is False


def test_unauthenticated_dev_mode():
    controller = LocalController(auto_register_local_node=False)
    server = APIServer(controller=controller, api_key=None)
    client = TestClient(server.app)

    # All endpoints return success without auth headers
    res_health = client.get("/v1/health")
    assert res_health.status_code == 200

    res_jobs = client.get("/v1/jobs")
    assert res_jobs.status_code == 200

    res_gpus = client.get("/v1/gpus")
    assert res_gpus.status_code == 200


def test_authenticated_mode_rejections_and_success():
    secret = "secret-cluster-key-999"
    controller = LocalController(auto_register_local_node=False)
    server = APIServer(controller=controller, api_key=secret)
    client = TestClient(server.app)

    # 1. Health check endpoint remains unauthenticated (200 OK)
    res_health = client.get("/v1/health")
    assert res_health.status_code == 200

    # 2. Protected endpoints reject missing auth with 401 Unauthorized
    res_no_auth = client.get("/v1/jobs")
    assert res_no_auth.status_code == 401
    assert "WWW-Authenticate" in res_no_auth.headers

    res_metrics_no_auth = client.get("/metrics")
    assert res_metrics_no_auth.status_code == 401

    # 3. Invalid auth token rejects with 401
    res_invalid = client.get("/v1/jobs", headers={"Authorization": "Bearer invalid-token"})
    assert res_invalid.status_code == 401

    # 4. Valid static Bearer API key succeeds
    res_valid_static = client.get("/v1/jobs", headers={"Authorization": f"Bearer {secret}"})
    assert res_valid_static.status_code == 200

    # 5. Valid X-API-Key header succeeds
    res_valid_x_key = client.get("/v1/jobs", headers={"X-API-Key": secret})
    assert res_valid_x_key.status_code == 200

    # 6. Valid HMAC Bearer Token succeeds
    hmac_token = create_hmac_token(secret)
    res_valid_hmac = client.get("/v1/jobs", headers={"Authorization": f"Bearer {hmac_token}"})
    assert res_valid_hmac.status_code == 200


def test_remote_worker_client_authentication():
    secret = "worker-auth-secret-key"
    registry = WorkerRegistry()
    controller = LocalController(auto_register_local_node=False)
    api_server = APIServer(controller=controller, worker_registry=registry, api_key=secret)
    test_client = TestClient(api_server.app)

    # Monkeypatch RemoteHTTPWorkerRegistryClient to route requests through TestClient
    worker_client_valid = RemoteHTTPWorkerRegistryClient(controller_url="http://testserver", api_key=secret)
    worker_client_invalid = RemoteHTTPWorkerRegistryClient(controller_url="http://testserver", api_key="wrong-key")

    def mock_get_client_valid():
        return test_client

    # 1. Unauthenticated / Wrong key worker registration fails
    worker_client_invalid._get_client = lambda: test_client
    worker = WorkerInfo(
        worker_id="secured-worker-1",
        hostname="secure-host",
        ip_address="10.0.0.1",
        devices=[],
        total_cpus=8,
        total_memory_bytes=16 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    assert worker_client_invalid.register(worker) is False
    assert worker_client_invalid.heartbeat("secured-worker-1") is False

    # 2. Authenticated worker client succeeds
    # Overwrite _get_client to inject headers into test_client
    class AuthenticatedTestClient:
        def __init__(self, client: TestClient, key: str):
            self.client = client
            self.headers = {"Authorization": f"Bearer {key}"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self.headers)
            return self.client.post(url, headers=headers, **kwargs)

        def get(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self.headers)
            return self.client.get(url, headers=headers, **kwargs)

    worker_client_valid._get_client = lambda: AuthenticatedTestClient(test_client, secret)

    assert worker_client_valid.register(worker) is True
    assert worker_client_valid.heartbeat("secured-worker-1") is True
    active_workers = worker_client_valid.get_active_workers()
    assert len(active_workers) == 1
    assert active_workers[0].worker_id == "secured-worker-1"
