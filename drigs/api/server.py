"""FastAPI REST Control Plane Server for DRIGS."""

from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from drigs.api.auth import APIKeyAuth
from drigs.core.controller import LocalController
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    Job,
    JobStatus,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.core.queue import JobQueue, AdmissionError
from drigs.core.resource_manager import ResourceManager
from drigs.diagnostics.analyzer import DiagnosticAnalyzer
from drigs.monitoring.metrics import MetricsCollector
from drigs.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class JobSubmitRequest(BaseModel):
    """Request payload for submitting a new job."""

    name: Optional[str] = None
    spec: WorkloadSpec
    priority: int = Field(default=0, ge=0)


class JobSubmitResponse(BaseModel):
    """Response payload returned upon job submission."""

    job_id: str
    status: JobStatus
    message: str


class APIServer:
    """FastAPI Control Plane Server encapsulating DRIGS management endpoints."""

    def __init__(
        self,
        controller: Optional[LocalController] = None,
        job_queue: Optional[JobQueue] = None,
        worker_registry: Optional[WorkerRegistry] = None,
        resource_manager: Optional[ResourceManager] = None,
        api_key: Optional[str] = None,
    ):
        self.controller = controller or LocalController()
        self.job_queue = job_queue or self.controller.job_queue
        self.worker_registry = worker_registry
        self.resource_manager = resource_manager or self.controller.resource_manager
        self.api_key = api_key if api_key is not None else os.getenv("DRIGS_API_KEY")
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        auth_dep = APIKeyAuth(api_key=self.api_key)
        app = FastAPI(
            title="DRIGS Control Plane API",
            description="Distributed Resource & Intelligent GPU Scheduling REST API",
            version="0.1.0",
            dependencies=[Depends(auth_dep)],
        )

        @app.get("/v1/health")
        def health_check() -> Dict[str, str]:
            return {"status": "ok", "service": "drigs-control-plane"}

        @app.get("/metrics", response_class=PlainTextResponse)
        def get_metrics() -> str:
            collector = MetricsCollector(
                job_queue=self.job_queue,
                resource_manager=self.resource_manager,
                worker_registry=self.worker_registry,
            )
            return collector.generate_prometheus_text()

        @app.post("/v1/jobs", response_model=JobSubmitResponse, status_code=status.HTTP_201_CREATED)
        def submit_job(req: JobSubmitRequest) -> JobSubmitResponse:
            job_name = req.name or req.spec.name
            try:
                submitted_job = self.controller.submit_job(
                    spec_or_content=req.spec,
                    priority=req.priority,
                    job_name=job_name,
                )
                return JobSubmitResponse(
                    job_id=submitted_job.id,
                    status=submitted_job.status,
                    message=f"Job {submitted_job.id} submitted successfully.",
                )
            except AdmissionError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                logger.error("Error submitting job: %s", e)
                raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e

        @app.get("/v1/jobs", response_model=List[Dict[str, Any]])
        def list_jobs(status: Optional[JobStatus] = None) -> List[Dict[str, Any]]:
            jobs = list(self.job_queue._jobs.values())
            if status is not None:
                jobs = [j for j in jobs if j.status == status]
            return [j.model_dump(mode="json") for j in jobs]

        @app.get("/v1/jobs/{job_id}", response_model=Dict[str, Any])
        def get_job(job_id: str) -> Dict[str, Any]:
            job = self.controller.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
            return job.model_dump(mode="json")

        @app.get("/v1/jobs/{job_id}/diagnose", response_model=Dict[str, Any])
        def diagnose_job(job_id: str) -> Dict[str, Any]:
            analyzer = DiagnosticAnalyzer(
                controller=self.controller,
                job_queue=self.job_queue,
                resource_manager=self.resource_manager,
                worker_registry=self.worker_registry,
            )
            report = analyzer.analyze_job(job_id)
            return report.model_dump(mode="json")

        @app.post("/v1/jobs/{job_id}/cancel")
        def cancel_job(job_id: str) -> Dict[str, Any]:
            success = self.controller.cancel_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail=f"Failed to cancel job {job_id}.")
            return {"job_id": job_id, "status": "CANCELLED", "message": "Job cancelled successfully."}

        @app.get("/v1/workers", response_model=List[Dict[str, Any]])
        def list_workers() -> List[Dict[str, Any]]:
            if self.worker_registry:
                workers = self.worker_registry.get_active_workers()
                return [w.model_dump(mode="json") for w in workers]
            workers_dict = getattr(self.resource_manager, "_workers", {})
            return [w.model_dump(mode="json") for w in workers_dict.values()]

        @app.post("/v1/workers/register", status_code=status.HTTP_201_CREATED)
        def register_worker(worker: WorkerInfo) -> Dict[str, Any]:
            self.resource_manager.register_worker(worker)
            if self.worker_registry:
                self.worker_registry.register(worker)
            logger.info("Registered worker %s (%s)", worker.worker_id, worker.hostname)
            return {"status": "registered", "worker_id": worker.worker_id}

        @app.post("/v1/workers/{worker_id}/heartbeat")
        def worker_heartbeat(worker_id: str, status: Optional[str] = None) -> Dict[str, Any]:
            dev_state = DeviceState.HEALTHY
            if status:
                try:
                    dev_state = DeviceState(status.upper())
                except ValueError:
                    pass

            if self.worker_registry:
                self.worker_registry.heartbeat(worker_id, status=dev_state)

            workers_dict = getattr(self.resource_manager, "_workers", {})
            if worker_id in workers_dict:
                w = workers_dict[worker_id]
                updated_worker = WorkerInfo(
                    worker_id=w.worker_id,
                    hostname=w.hostname,
                    ip_address=w.ip_address,
                    devices=w.devices,
                    total_cpus=w.total_cpus,
                    total_memory_bytes=w.total_memory_bytes,
                    status=dev_state,
                    last_heartbeat=datetime.now(timezone.utc),
                )
                self.resource_manager.register_worker(updated_worker)

            return {"status": "acknowledged", "worker_id": worker_id}

        @app.post("/v1/workers/{worker_id}/deregister")
        def deregister_worker(worker_id: str) -> Dict[str, Any]:
            self.resource_manager.unregister_worker(worker_id)
            if self.worker_registry:
                self.worker_registry.deregister(worker_id)
            return {"status": "deregistered", "worker_id": worker_id}

        @app.get("/v1/gpus", response_model=List[Dict[str, Any]])
        def list_gpus() -> List[Dict[str, Any]]:
            available_devices = self.resource_manager.get_available_devices()
            gpus = [
                d for d in available_devices
                if str(getattr(d.device_type, "value", d.device_type)).upper() == "GPU"
            ]
            return [g.model_dump(mode="json") for g in gpus]

        @app.get("/v1/cluster", response_model=Dict[str, Any])
        def get_cluster_state() -> Dict[str, Any]:
            state = self.resource_manager.get_cluster_state()
            return state.model_dump(mode="json")

        return app
