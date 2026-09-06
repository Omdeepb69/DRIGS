"""Unit tests for MetricsCollector and Prometheus exposition in drigs.monitoring."""

import pytest
from fastapi.testclient import TestClient

from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    JobStatus,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.monitoring.metrics import MetricValue, MetricsCollector
from drigs.workers.registry import WorkerRegistry


def test_metric_value_set_get_format():
    mv = MetricValue("test_counter", "Help for test counter", "counter")
    mv.set(10.0, {"env": "prod", "node": "node1"})

    assert mv.get({"env": "prod", "node": "node1"}) == 10.0

    mv.inc(5.0, {"env": "prod", "node": "node1"})
    assert mv.get({"env": "prod", "node": "node1"}) == 15.0

    prom_text = mv.to_prometheus_text()
    assert "# HELP test_counter Help for test counter" in prom_text
    assert "# TYPE test_counter counter" in prom_text
    assert 'test_counter{env="prod",node="node1"} 15' in prom_text


def test_metrics_collector_cluster_state():
    job_queue = JobQueue()
    resource_mgr = ResourceManager()
    registry = WorkerRegistry()

    gpu = ComputeDevice(
        device_id="gpu-0",
        model_name="NVIDIA A100",
        device_type=DeviceType.GPU,
        total_memory_bytes=40 * 1024**3,
        available_memory_bytes=30 * 1024**3,
        state=DeviceState.HEALTHY,
    )
    worker = WorkerInfo(
        worker_id="w-1",
        hostname="node1.local",
        ip_address="10.0.0.1",
        devices=[gpu],
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    registry.register(worker)
    resource_mgr.register_worker(worker)

    spec = WorkloadSpec(
        name="job-1",
        resources=ResourceRequirements(cpus=2),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )

    j1 = Job(name="j1", spec=spec)
    j2 = Job(name="j2", spec=spec)
    j3 = Job(name="j3", spec=spec)

    job_queue.enqueue(j1)  # QUEUED

    job_queue.enqueue(j2)
    job_queue.update_job_status(j2.id, JobStatus.SCHEDULED)
    job_queue.update_job_status(j2.id, JobStatus.RUNNING)  # RUNNING

    job_queue.enqueue(j3)
    job_queue.update_job_status(j3.id, JobStatus.SCHEDULED)
    job_queue.update_job_status(j3.id, JobStatus.RUNNING)
    job_queue.update_job_status(j3.id, JobStatus.COMPLETED)  # COMPLETED

    collector = MetricsCollector(
        job_queue=job_queue,
        resource_manager=resource_mgr,
        worker_registry=registry,
    )
    collector.collect()

    assert collector.active_workers.get() == 1.0
    assert collector.total_cpus.get() == 16.0
    assert collector.total_gpus.get() == 1.0
    assert collector.queued_jobs.get() == 1.0
    assert collector.running_jobs.get() == 1.0
    assert collector.completed_jobs.get() == 1.0

    prom_text = collector.generate_prometheus_text()
    assert "drigs_active_workers 1" in prom_text
    assert "drigs_total_cpus 16" in prom_text
    assert "drigs_total_gpus 1" in prom_text
    assert "drigs_queued_jobs 1" in prom_text
    assert "drigs_running_jobs 1" in prom_text
    assert "drigs_completed_jobs_total 1" in prom_text
    assert 'drigs_gpu_memory_total_bytes{device_id="gpu-0",worker_id="w-1"} 42949672960' in prom_text


def test_api_metrics_endpoint():
    controller = LocalController()
    server = APIServer(controller=controller)
    client = TestClient(server.app)

    res = client.get("/metrics")
    assert res.status_code == 200
    assert "# HELP drigs_active_workers" in res.text
    assert "drigs_active_workers" in res.text
