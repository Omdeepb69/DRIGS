"""Priority Scheduler Policy with Starvation Prevention for DRIGS."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Set

from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    DeviceState,
    DeviceType,
    Job,
    ResourceAllocation,
)

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class PriorityScheduler:
    """Priority scheduler: orders pending jobs by job priority and aging factor to prevent starvation."""

    def __init__(self, aging_factor: float = 0.01):
        self.aging_factor = aging_factor

    def compute_effective_priority(self, job: Job, now: Optional[datetime] = None) -> float:
        """Compute effective job priority incorporating wait time aging boost."""
        current_time = now or _now_utc()
        wait_seconds = max(0.0, (current_time - job.submitted_at).total_seconds())
        return job.priority + (self.aging_factor * wait_seconds)

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        allocations: List[ResourceAllocation] = []
        if not pending_jobs or not cluster_state.workers:
            return allocations

        now = _now_utc()
        # Sort jobs by effective priority (highest priority first), then submission time
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (-self.compute_effective_priority(j, now), j.submitted_at),
        )

        busy_devices: Dict[str, Set[str]] = {w_id: set() for w_id in cluster_state.workers}
        allocated_cpus: Dict[str, int] = {w_id: 0 for w_id in cluster_state.workers}
        allocated_memory: Dict[str, int] = {w_id: 0 for w_id in cluster_state.workers}

        sorted_worker_ids = sorted(cluster_state.workers.keys())

        for job in sorted_jobs:
            req = job.spec.resources
            req_gpus = req.gpus
            req_vram = req.gpu_memory_bytes
            req_cpus = req.cpus
            req_ram = req.memory_bytes

            for w_id in sorted_worker_ids:
                worker = cluster_state.workers[w_id]
                if worker.status != DeviceState.HEALTHY:
                    continue

                avail_cpus = max(0, worker.total_cpus - allocated_cpus[w_id])
                avail_ram = max(0, worker.total_memory_bytes - allocated_memory[w_id])

                if avail_cpus < req_cpus or avail_ram < req_ram:
                    continue

                candidate_gpus: List[str] = []
                if req_gpus > 0:
                    for dev in worker.devices:
                        if dev.device_type != DeviceType.GPU:
                            continue
                        if dev.device_id in busy_devices[w_id]:
                            continue
                        if req_vram > 0 and dev.available_memory_bytes < req_vram:
                            continue
                        candidate_gpus.append(dev.device_id)

                    if len(candidate_gpus) < req_gpus:
                        continue

                assigned_gpus = candidate_gpus[:req_gpus] if req_gpus > 0 else []
                busy_devices[w_id].update(assigned_gpus)
                allocated_cpus[w_id] += req_cpus
                allocated_memory[w_id] += req_ram

                assigned_cores = list(range(allocated_cpus[w_id] - req_cpus, allocated_cpus[w_id]))

                alloc = ResourceAllocation(
                    job_id=job.id,
                    worker_id=w_id,
                    assigned_device_ids=assigned_gpus,
                    assigned_cpu_cores=assigned_cores,
                    memory_bytes=req_ram,
                )
                allocations.append(alloc)
                logger.info(
                    "PriorityScheduler scheduled job %s (priority=%d) on worker %s",
                    job.id,
                    job.priority,
                    w_id,
                )
                break

        return allocations
