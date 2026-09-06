"""Unit tests for DiagnosticAnalyzer in drigs.diagnostics.analyzer."""

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
from drigs.diagnostics.analyzer import (
    DiagnosticAnalyzer,
    DiagnosticSeverity,
)


def test_diagnose_nonexistent_job():
    analyzer = DiagnosticAnalyzer()
    report = analyzer.analyze_job("nonexistent-999")

    assert report.job_id == "nonexistent-999"
    assert report.health_score == 0.0
    assert len(report.findings) == 1
    assert report.findings[0].category == "EXECUTION"
    assert report.findings[0].severity == DiagnosticSeverity.CRITICAL
    assert report.findings[0].title == "Job Not Found"


def test_diagnose_failed_job_with_vram_error():
    controller = LocalController()

    spec = WorkloadSpec(
        name="failed-vram-job",
        resources=ResourceRequirements(gpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["python3", "train.py"]),
    )
    job = controller.submit_job(spec)
    controller.job_queue.update_job_status(job.id, JobStatus.FAILED, error_message="CUDA Out of Memory (OOM) on GPU 0")

    analyzer = DiagnosticAnalyzer(controller=controller)
    report = analyzer.analyze_job(job.id)

    assert report.job_id == job.id
    assert report.status == JobStatus.FAILED
    assert report.health_score == 50.0
    assert len(report.findings) == 1
    assert report.findings[0].category == "VRAM"
    assert report.findings[0].severity == DiagnosticSeverity.ERROR
    assert "OOM" in report.findings[0].description


def test_diagnose_queued_job_capacity_bottleneck():
    resource_mgr = ResourceManager()
    gpu = ComputeDevice(
        device_id="gpu-0",
        model_name="Tesla T4",
        device_type=DeviceType.GPU,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=16 * 1024**3,
        state=DeviceState.HEALTHY,
    )
    worker = WorkerInfo(
        worker_id="w-1",
        hostname="node1",
        ip_address="127.0.0.1",
        devices=[gpu],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    resource_mgr.register_worker(worker)

    job_queue = JobQueue()
    # Job requesting 80GB VRAM (exceeding 16GB max available)
    spec = WorkloadSpec(
        name="oversized-vram-job",
        resources=ResourceRequirements(gpus=1, gpu_memory_bytes=80 * 1024**3),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "hi"]),
    )
    job = Job(name="oversized-vram-job", spec=spec)
    job_queue.enqueue(job)

    analyzer = DiagnosticAnalyzer(job_queue=job_queue, resource_manager=resource_mgr)
    report = analyzer.analyze_job(job.id)

    assert report.job_id == job.id
    assert report.status == JobStatus.QUEUED
    assert report.health_score < 100.0

    vram_findings = [f for f in report.findings if f.category == "VRAM"]
    assert len(vram_findings) == 1
    assert vram_findings[0].severity == DiagnosticSeverity.CRITICAL
    assert "Capacity Exceeded" in vram_findings[0].title


def test_diagnose_healthy_running_job():
    controller = LocalController()
    spec = WorkloadSpec(
        name="healthy-job",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "ok"]),
    )
    job = controller.submit_job(spec)
    controller.job_queue.update_job_status(job.id, JobStatus.SCHEDULED)
    controller.job_queue.update_job_status(job.id, JobStatus.RUNNING)

    analyzer = DiagnosticAnalyzer(controller=controller)
    report = analyzer.analyze_job(job.id)

    assert report.health_score == 100.0
    assert len(report.findings) == 1
    assert report.findings[0].severity == DiagnosticSeverity.INFO
    assert report.findings[0].title == "Healthy Execution State"


def test_api_diagnose_endpoint():
    controller = LocalController()
    spec = WorkloadSpec(
        name="api-diag-job",
        resources=ResourceRequirements(cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "diag"]),
    )
    job = controller.submit_job(spec)

    server = APIServer(controller=controller)
    client = TestClient(server.app)

    res = client.get(f"/v1/jobs/{job.id}/diagnose")
    assert res.status_code == 200
    data = res.json()

    assert data["job_id"] == job.id
    assert data["health_score"] == 100.0
    assert len(data["findings"]) == 1
    assert data["findings"][0]["severity"] == "INFO"
