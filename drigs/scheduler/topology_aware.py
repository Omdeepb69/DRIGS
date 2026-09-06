"""Topology-Aware Scheduling Policy for DRIGS."""

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
from drigs.hardware.topology import TopologyGraph

logger = logging.getLogger(__name__)


class TopologyAwareScheduler:
    """Topology-Aware scheduler selecting GPU placements to maximize interconnect bandwidth scores (NVLink > PCIe Switch > PCIe > NUMA)."""

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
            best_topo_score: float = -1.0

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

                # Build topology graph for worker devices
                graph = TopologyGraph(devices=worker.devices)

                if req_gpus > 0:
                    assigned_gpus = graph.find_best_clique(req_gpus, candidate_gpus)
                    topo_score = graph.get_matrix().get_group_topology_score(assigned_gpus)
                else:
                    assigned_gpus = []
                    topo_score = 100.0

                if topo_score > best_topo_score:
                    best_topo_score = topo_score
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
                    "TopologyAwareScheduler allocated job %s on worker %s (gpus=%s, topo_score=%.1f)",
                    job.id,
                    w_id,
                    best_candidate_gpus,
                    best_topo_score,
                )

        return allocations
