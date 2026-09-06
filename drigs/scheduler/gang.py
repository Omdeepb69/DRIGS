"""Gang Scheduling Policy Wrapper for DRIGS."""

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
from drigs.scheduler.fifo import FIFOScheduler

logger = logging.getLogger(__name__)


class GangScheduler:
    """Gang scheduler guaranteeing atomic all-or-nothing resource reservations for workloads."""

    def __init__(self, inner_scheduler: Optional[Scheduler] = None):
        self.inner_scheduler = inner_scheduler or FIFOScheduler()

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

            # 1. Try single-node placement first
            single_alloc = self._try_single_node(
                job, req_gpus, req_vram, req_cpus, req_ram, cluster_state, sorted_worker_ids, busy_devices, allocated_cpus, allocated_memory
            )
            if single_alloc:
                allocations.append(single_alloc)
                continue

            # 2. Try multi-node gang placement if single-node placement failed
            multi_allocs = self._try_multi_node_gang(
                job, req_gpus, req_vram, req_cpus, req_ram, cluster_state, sorted_worker_ids, busy_devices, allocated_cpus, allocated_memory
            )
            if multi_allocs:
                allocations.extend(multi_allocs)

        return allocations

    def _try_single_node(
        self,
        job: Job,
        req_gpus: int,
        req_vram: int,
        req_cpus: int,
        req_ram: int,
        cluster_state: ClusterState,
        worker_ids: List[str],
        busy_devices: Dict[str, Set[str]],
        allocated_cpus: Dict[str, int],
        allocated_memory: Dict[str, int],
    ) -> Optional[ResourceAllocation]:
        for w_id in worker_ids:
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

            return ResourceAllocation(
                job_id=job.id,
                worker_id=w_id,
                assigned_device_ids=assigned_gpus,
                assigned_cpu_cores=assigned_cores,
                memory_bytes=req_ram,
            )
        return None

    def _try_multi_node_gang(
        self,
        job: Job,
        req_gpus: int,
        req_vram: int,
        req_cpus: int,
        req_ram: int,
        cluster_state: ClusterState,
        worker_ids: List[str],
        busy_devices: Dict[str, Set[str]],
        allocated_cpus: Dict[str, int],
        allocated_memory: Dict[str, int],
    ) -> List[ResourceAllocation]:
        if req_gpus <= 0:
            return []

        # Find candidate GPUs across all healthy workers
        worker_gpu_map: Dict[str, List[str]] = {}
        total_available_gpus = 0

        for w_id in worker_ids:
            worker = cluster_state.workers[w_id]
            if worker.status != DeviceState.HEALTHY:
                continue

            avail_cpus = max(0, worker.total_cpus - allocated_cpus[w_id])
            avail_ram = max(0, worker.total_memory_bytes - allocated_memory[w_id])

            if avail_cpus < 1 or avail_ram < 1024**2:
                continue

            candidates: List[str] = []
            for dev in worker.devices:
                if dev.device_type != DeviceType.GPU:
                    continue
                if dev.device_id in busy_devices[w_id]:
                    continue
                if req_vram > 0 and dev.available_memory_bytes < req_vram:
                    continue
                candidates.append(dev.device_id)

            if candidates:
                worker_gpu_map[w_id] = candidates
                total_available_gpus += len(candidates)

        # Atomic Gang Check: MUST satisfy full req_gpus count across cluster
        if total_available_gpus < req_gpus:
            logger.debug(
                "Gang scheduling deferred job %s: requested %d GPUs, available %d",
                job.id,
                req_gpus,
                total_available_gpus,
            )
            return []

        # Satisfy gang allocation across workers
        gang_allocations: List[ResourceAllocation] = []
        remaining_needed = req_gpus

        for w_id, candidates in worker_gpu_map.items():
            if remaining_needed <= 0:
                break

            take_count = min(len(candidates), remaining_needed)
            assigned_gpus = candidates[:take_count]
            remaining_needed -= take_count

            # Proportional CPU & RAM per worker
            w_cpus = max(1, req_cpus // len(worker_gpu_map))
            w_ram = max(1024**2, req_ram // len(worker_gpu_map))

            busy_devices[w_id].update(assigned_gpus)
            allocated_cpus[w_id] += w_cpus
            allocated_memory[w_id] += w_ram

            assigned_cores = list(range(allocated_cpus[w_id] - w_cpus, allocated_cpus[w_id]))

            alloc = ResourceAllocation(
                job_id=job.id,
                worker_id=w_id,
                assigned_device_ids=assigned_gpus,
                assigned_cpu_cores=assigned_cores,
                memory_bytes=w_ram,
            )
            gang_allocations.append(alloc)

        return gang_allocations
