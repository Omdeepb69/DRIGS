"""Systems Diagnostic Subsystem for DRIGS workload analysis."""

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from drigs.core.controller import LocalController
from drigs.core.models import DeviceState, Job, JobStatus
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class DiagnosticSeverity(str, Enum):
    """Severity level of a diagnostic finding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DiagnosticFinding(BaseModel):
    """Single diagnostic finding detailing an observed issue or status."""

    category: str  # e.g., "VRAM", "CPU", "QUEUE", "WORKER", "EXECUTION"
    severity: DiagnosticSeverity
    title: str
    description: str
    recommendation: Optional[str] = None


class DiagnosticReport(BaseModel):
    """Comprehensive diagnostic analysis report for a workload."""

    job_id: str
    job_name: str
    status: JobStatus
    health_score: float = Field(default=100.0, ge=0.0, le=100.0)
    findings: List[DiagnosticFinding] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DiagnosticAnalyzer:
    """Analyzes workload execution failures, VRAM pressure, queue starvation, and worker health."""

    def __init__(
        self,
        controller: Optional[LocalController] = None,
        job_queue: Optional[JobQueue] = None,
        resource_manager: Optional[ResourceManager] = None,
        worker_registry: Optional[WorkerRegistry] = None,
    ):
        self.controller = controller
        self.job_queue = job_queue or (controller.job_queue if controller else None)
        self.resource_manager = resource_manager or (controller.resource_manager if controller else None)
        self.worker_registry = worker_registry

    def analyze_job(self, job_id: str) -> DiagnosticReport:
        """Perform full diagnostic analysis on a specific job ID."""
        job: Optional[Job] = None
        if self.controller:
            job = self.controller.get_job(job_id)
        elif self.job_queue:
            job = self.job_queue.get_job(job_id)

        if not job:
            finding = DiagnosticFinding(
                category="EXECUTION",
                severity=DiagnosticSeverity.CRITICAL,
                title="Job Not Found",
                description=f"Job ID '{job_id}' could not be located in the control plane queue or history.",
                recommendation="Verify the job ID or submit the workload spec again.",
            )
            return DiagnosticReport(
                job_id=job_id,
                job_name="Unknown",
                status=JobStatus.FAILED,
                health_score=0.0,
                findings=[finding],
            )

        findings: List[DiagnosticFinding] = []
        health_score = 100.0

        # 1. Analyze status & execution errors
        if job.status == JobStatus.FAILED:
            health_score -= 50.0
            err_msg = job.error_message or "Unknown execution error"

            category = "EXECUTION"
            rec = "Inspect entrypoint script, logs, and dependencies."

            if "vram" in err_msg.lower() or "memory" in err_msg.lower() or "oom" in err_msg.lower():
                category = "VRAM"
                rec = "Reduce workload batch size or request GPUs with larger VRAM capacity."
            elif "worker" in err_msg.lower() or "timeout" in err_msg.lower():
                category = "WORKER"
                rec = "Check worker node network connection and process group responsiveness."

            findings.append(
                DiagnosticFinding(
                    category=category,
                    severity=DiagnosticSeverity.ERROR,
                    title=f"Job Execution Failed",
                    description=f"Job failed with error: {err_msg}",
                    recommendation=rec,
                )
            )

        # 2. Analyze resource requirements & queue bottlenecks if QUEUED
        if job.status == JobStatus.QUEUED:
            req = job.spec.resources

            # Check VRAM requirements against available GPUs
            available_devices = []
            if self.resource_manager:
                available_devices = self.resource_manager.get_available_devices()

            gpus = [d for d in available_devices if str(getattr(d.device_type, "value", d.device_type)).upper() == "GPU"]

            if req.gpus > 0 and not gpus:
                health_score -= 30.0
                findings.append(
                    DiagnosticFinding(
                        category="VRAM",
                        severity=DiagnosticSeverity.WARNING,
                        title="GPU Bottleneck - No Available GPUs",
                        description=f"Job requested {req.gpus} GPU(s), but no GPUs are currently available in healthy state.",
                        recommendation="Register additional GPU worker nodes or wait for active jobs to release GPUs.",
                    )
                )

            if req.gpu_memory_bytes > 0 and gpus:
                max_vram = max(g.total_memory_bytes for g in gpus)
                if req.gpu_memory_bytes > max_vram:
                    health_score -= 40.0
                    findings.append(
                        DiagnosticFinding(
                            category="VRAM",
                            severity=DiagnosticSeverity.CRITICAL,
                            title="VRAM Pressure - Capacity Exceeded",
                            description=(
                                f"Requested GPU VRAM ({req.gpu_memory_bytes / (1024**3):.2f} GB) "
                                f"exceeds maximum available GPU VRAM capacity ({max_vram / (1024**3):.2f} GB)."
                            ),
                            recommendation="Scale down VRAM request or add higher capacity GPU nodes (e.g. A100 80GB).",
                        )
                    )

            # Check CPU requirements
            total_cpus = 0
            if self.resource_manager:
                workers_dict = getattr(self.resource_manager, "_workers", {})
                total_cpus = sum(w.total_cpus for w in workers_dict.values())

            if req.cpus > total_cpus and total_cpus > 0:
                health_score -= 30.0
                findings.append(
                    DiagnosticFinding(
                        category="CPU",
                        severity=DiagnosticSeverity.WARNING,
                        title="CPU Capacity Bottleneck",
                        description=f"Job requested {req.cpus} CPUs, exceeding cluster total of {total_cpus} CPUs.",
                        recommendation="Reduce CPU core request or scale cluster workers.",
                    )
                )

        # 3. Analyze assigned worker node health if RUNNING or SCHEDULED
        if job.allocation and self.resource_manager:
            worker_id = job.allocation.worker_id
            workers_dict = getattr(self.resource_manager, "_workers", {})
            worker = workers_dict.get(worker_id)
            if worker and worker.status != DeviceState.HEALTHY:
                health_score -= 30.0
                findings.append(
                    DiagnosticFinding(
                        category="WORKER",
                        severity=DiagnosticSeverity.WARNING,
                        title=f"Worker Node Degraded ({worker.status.value})",
                        description=f"Assigned worker '{worker_id}' is currently in state {worker.status.value}.",
                        recommendation="Monitor worker node heartbeats and health metrics.",
                    )
                )

        # 4. If healthy, add informational finding
        if not findings:
            findings.append(
                DiagnosticFinding(
                    category="EXECUTION",
                    severity=DiagnosticSeverity.INFO,
                    title="Healthy Execution State",
                    description=f"Job '{job_id}' is operating normally in status {job.status.value}.",
                    recommendation="No corrective action required.",
                )
            )

        health_score = max(0.0, min(100.0, health_score))

        return DiagnosticReport(
            job_id=job.id,
            job_name=job.name,
            status=job.status,
            health_score=health_score,
            findings=findings,
        )
