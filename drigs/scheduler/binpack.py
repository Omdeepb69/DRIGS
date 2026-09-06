"""Bin-Packing Scheduler Policy for DRIGS."""

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


class BinPackScheduler:
    """Bin-Packing scheduler: packs workloads onto worker nodes with the smallest remaining surplus capacity to minimize VRAM fragmentation."""

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        allocations: List[ResourceAllocation] = []
        if not pending_jobs or not cluster_state.workers:
            return allocations

        sorted_jobs = sorted(pending_jobs, key=lambda j: (-j.priority, j.submitted_at))

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

            best_worker_id: Optional[str] = None
            best_candidate_gpus: List[str] = []
            best_surplus_score: Optional[float] = None

            for w_id in sorted_worker_ids:
                worker = cluster_state.workers[w_id]
                if worker.status != DeviceState.HEALTHY:
                    continue

                avail_cpus = max(0, worker.total_cpus - allocated_cpus[w_id])
                avail_ram = max(0, worker.total_memory_bytes - allocated_memory[w_id])

                if avail_cpus < req_cpus or avail_ram < req_ram:
                    continue

                candidate_gpus: List[str] = []
                total_avail_vram = 0
                if req_gpus > 0:
                    for dev in worker.devices:
                        if dev.device_type != DeviceType.GPU:
                            continue
                        if dev.device_id in busy_devices[w_id]:
                            continue
                        if req_vram > 0 and dev.available_memory_bytes < req_vram:
                            continue
                        candidate_gpus.append(dev.device_id)
                        total_avail_vram += dev.available_memory_bytes

                    if len(candidate_gpus) < req_gpus:
                        continue

                assigned_gpus = candidate_gpus[:req_gpus] if req_gpus > 0 else []

                # Calculate remaining surplus score (smaller score means better packed node)
                rem_cpus = avail_cpus - req_cpus
                rem_ram = avail_ram - req_ram
                rem_vram = max(0, total_avail_vram - (req_vram * req_gpus if req_gpus > 0 else 0))

                surplus_score = rem_cpus + (rem_ram / 1e9) + (rem_vram / 1e9)

                if best_surplus_score is None or surplus_score < best_surplus_score:
                    best_surplus_score = surplus_score
                    best_worker_id = w_id
                    best_candidate_gpus = assigned_gpus

            if best_worker_id is not None:
                w_id = best_worker_id
                busy_devices[w_id].update(best_candidate_gpus)
                allocated_cpus[w_id] += req_cpus
                allocated_memory[w_id] += req_ram

                assigned_cores = list(range(allocated_cpus[w_id] - req_cpus, allocated_cpus[w_id]))

                alloc = ResourceAllocation(
                    job_id=job.id,
                    worker_id=w_id,
                    assigned_device_ids=best_candidate_gpus,
                    assigned_cpu_cores=assigned_cores,
                    memory_bytes=req_ram,
                )
                allocations.append(alloc)
                logger.info(
                    "BinPackScheduler packed job %s onto worker %s (surplus_score=%.2f)",
                    job.id,
                    w_id,
                    best_surplus_score or 0.0,
                )

        return allocations
