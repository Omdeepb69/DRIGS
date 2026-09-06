"""FIFO (First-In, First-Out) Scheduling Policy for DRIGS."""

import logging
from typing import Dict, List, Set
from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    DeviceState,
    DeviceType,
    Job,
    ResourceAllocation,
)

logger = logging.getLogger(__name__)


class FIFOScheduler:
    """Schedules jobs in strictly FIFO order (by submission time)."""

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        allocations: List[ResourceAllocation] = []
        if not pending_jobs or not cluster_state.workers:
            return allocations

        # Sort jobs by submission time (FIFO)
        sorted_jobs = sorted(pending_jobs, key=lambda j: j.submitted_at)

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
                break

        return allocations
