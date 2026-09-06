"""Runtime metrics collector and Prometheus format exporter for DRIGS."""

import logging
from threading import RLock
from typing import Dict, List, Optional

from drigs.core.models import JobStatus
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.workers.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class MetricValue:
    """Represents a single Prometheus metric gauge or counter value."""

    def __init__(self, name: str, help_text: str, metric_type: str = "gauge"):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self._values: Dict[str, float] = {}  # label_str -> value

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set metric value for specified labels."""
        key = self._format_labels(labels)
        self._values[key] = float(value)

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment metric counter/gauge value by amount."""
        key = self._format_labels(labels)
        self._values[key] = self._values.get(key, 0.0) + float(value)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Retrieve current metric value for specified labels."""
        key = self._format_labels(labels)
        return self._values.get(key, 0.0)

    def _format_labels(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        items = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(items) + "}"

    def to_prometheus_text(self) -> str:
        """Render metric into standard Prometheus text exposition format."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.metric_type}",
        ]
        if not self._values:
            lines.append(f"{self.name} 0")
        else:
            for label_str, val in self._values.items():
                if val.is_integer():
                    val_str = str(int(val))
                else:
                    val_str = f"{val:.6f}"
                lines.append(f"{self.name}{label_str} {val_str}")
        return "\n".join(lines)


class MetricsCollector:
    """Collects cluster, device, and workload operational metrics and exports in Prometheus format."""

    def __init__(
        self,
        job_queue: Optional[JobQueue] = None,
        resource_manager: Optional[ResourceManager] = None,
        worker_registry: Optional[WorkerRegistry] = None,
    ):
        self.job_queue = job_queue
        self.resource_manager = resource_manager
        self.worker_registry = worker_registry
        self._lock = RLock()

        self.active_workers = MetricValue("drigs_active_workers", "Total active worker nodes", "gauge")
        self.total_cpus = MetricValue("drigs_total_cpus", "Total CPU cores across active workers", "gauge")
        self.total_gpus = MetricValue("drigs_total_gpus", "Total GPU devices across active workers", "gauge")

        self.queued_jobs = MetricValue("drigs_queued_jobs", "Current number of queued jobs", "gauge")
        self.running_jobs = MetricValue("drigs_running_jobs", "Current number of running jobs", "gauge")
        self.completed_jobs = MetricValue("drigs_completed_jobs_total", "Total completed jobs count", "counter")
        self.failed_jobs = MetricValue("drigs_failed_jobs_total", "Total failed jobs count", "counter")

        self.gpu_memory_used = MetricValue("drigs_gpu_memory_used_bytes", "VRAM memory used in bytes", "gauge")
        self.gpu_memory_total = MetricValue("drigs_gpu_memory_total_bytes", "VRAM total memory in bytes", "gauge")

        self.scheduling_latency = MetricValue(
            "drigs_scheduling_latency_seconds", "Last scheduling pass duration in seconds", "gauge"
        )

    def record_scheduling_latency(self, seconds: float) -> None:
        """Record the duration of a scheduling orchestration pass."""
        with self._lock:
            self.scheduling_latency.set(seconds)

    def collect(self) -> None:
        """Poll state from job queue, resource manager, and worker registry to update metric gauges."""
        with self._lock:
            # 1. Collect worker & device metrics
            workers = []
            if self.worker_registry:
                workers = self.worker_registry.get_active_workers()
            elif self.resource_manager:
                workers = list(getattr(self.resource_manager, "_workers", {}).values())

            self.active_workers.set(len(workers))
            total_c = sum(w.total_cpus for w in workers)
            self.total_cpus.set(total_c)

            total_g = 0
            for w in workers:
                for dev in w.devices:
                    if str(getattr(dev.device_type, "value", dev.device_type)).upper() == "GPU":
                        total_g += 1
                        labels = {"worker_id": w.worker_id, "device_id": dev.device_id}
                        self.gpu_memory_total.set(dev.total_memory_bytes, labels)
                        used_mem = getattr(dev, "total_memory_bytes", 0) - getattr(dev, "available_memory_bytes", 0)
                        self.gpu_memory_used.set(max(0, used_mem), labels)

            self.total_gpus.set(total_g)

            # 2. Collect job queue metrics
            if self.job_queue:
                jobs = list(getattr(self.job_queue, "_jobs", {}).values())
                q_count = sum(1 for j in jobs if j.status == JobStatus.QUEUED)
                r_count = sum(1 for j in jobs if j.status == JobStatus.RUNNING)
                c_count = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
                f_count = sum(1 for j in jobs if j.status == JobStatus.FAILED)

                self.queued_jobs.set(q_count)
                self.running_jobs.set(r_count)
                self.completed_jobs.set(c_count)
                self.failed_jobs.set(f_count)

    def generate_prometheus_text(self) -> str:
        """Generate full Prometheus text exposition string."""
        self.collect()
        metrics = [
            self.active_workers,
            self.total_cpus,
            self.total_gpus,
            self.queued_jobs,
            self.running_jobs,
            self.completed_jobs,
            self.failed_jobs,
            self.gpu_memory_used,
            self.gpu_memory_total,
            self.scheduling_latency,
        ]
        return "\n\n".join(m.to_prometheus_text() for m in metrics) + "\n"
