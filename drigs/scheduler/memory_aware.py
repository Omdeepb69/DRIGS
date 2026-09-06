"""Memory-Aware Scheduling Policy for DRIGS."""

import logging
from typing import Dict, List, Optional, Set
from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    ResourceAllocation,
)

logger = logging.getLogger(__name__)


class MemoryAwareScheduler:
    """Memory-aware scheduler evaluating real-time VRAM and RAM availability to prevent OOMs."""

    def schedule(
        self,
        pending_jobs: List[Job],
        cluster_state: ClusterState,
    ) -> List[ResourceAllocation]:
        allocations: List[ResourceAllocation] = []
        if not pending_jobs or not cluster_state.workers:
            return allocations

        # Sort jobs by GPU VRAM requirement descending (largest VRAM request first)
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (-j.spec.resources.gpu_memory_bytes, -j.priority, j.submitted_at),
        )

        busy_devices: Dict[str, Set[str]] = {w_id: set() for w_id in cluster_state.workers}
        allocated_cpus: Dict[str, int] = {w_id: 0 for w_id in cluster_state.workers}
        allocated_memory: Dict[str, int] = {w_id: 0 for w_id in cluster_state.workers}

        for job in sorted_jobs:
            req = job.spec.resources
            req_gpus = req.gpus
            req_vram = req.gpu_memory_bytes
            req_cpus = req.cpus
            req_ram = req.memory_bytes

            best_worker_id: Optional[str] = None
            best_candidate_gpus: List[str] = []
            best_vram_fit_score: Optional[float] = None

            for w_id, worker in cluster_state.workers.items():
                if worker.status != DeviceState.HEALTHY:
                    continue

                avail_cpus = max(0, worker.total_cpus - allocated_cpus[w_id])
                avail_ram = max(0, worker.total_memory_bytes - allocated_memory[w_id])

                if avail_cpus < req_cpus or avail_ram < req_ram:
                    continue

                if req_gpus > 0:
                    gpu_candidates: List[ComputeDevice] = []
                    for dev in worker.devices:
                        if dev.device_type != DeviceType.GPU:
                            continue
                        if dev.device_id in busy_devices[w_id]:
                            continue
                        if req_vram > 0 and dev.available_memory_bytes < req_vram:
                            continue
                        gpu_candidates.append(dev)

                    if len(gpu_candidates) < req_gpus:
                        continue

                    # Sort GPUs by available VRAM descending
                    gpu_candidates.sort(key=lambda d: d.available_memory_bytes, reverse=True)
                    selected_devs = gpu_candidates[:req_gpus]
                    selected_ids = [d.device_id for d in selected_devs]

                    avg_vram = sum(d.available_memory_bytes for d in selected_devs) / req_gpus
                    if best_vram_fit_score is None or avg_vram > best_vram_fit_score:
                        best_vram_fit_score = avg_vram
                        best_worker_id = w_id
                        best_candidate_gpus = selected_ids
                else:
                    if best_vram_fit_score is None or avail_ram > best_vram_fit_score:
                        best_vram_fit_score = float(avail_ram)
                        best_worker_id = w_id
                        best_candidate_gpus = []

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

        return allocations
