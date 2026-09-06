"""Synthetic AI Workload Generator and Reproducible Benchmark Framework for DRIGS."""

import copy
from datetime import datetime, timezone
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict

from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.hardware.simulated import SimulatedBackend
from drigs.hardware.topology import TopologyGraph


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class SyntheticWorkloadItem(BaseModel):
    """Specification of a synthetic workload job in a benchmark trace dataset."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    name: str
    priority: int = 0
    spec: WorkloadSpec
    arrival_delay: float = Field(ge=0.0, description="Arrival delay in seconds relative to trace start")
    duration_seconds: float = Field(gt=0.0, description="Simulated execution duration in seconds")


class WorkloadGenerator:
    """Generates synthetic AI workload traces with varying GPU/VRAM requests, durations, and arrival patterns."""

    WORKLOAD_TEMPLATES = [
        {"name": "resnet50-training", "gpus": 1, "vram_gb": 8, "cpus": 4, "ram_gb": 16},
        {"name": "bert-large-finetune", "gpus": 2, "vram_gb": 16, "cpus": 8, "ram_gb": 32},
        {"name": "llama-7b-inference", "gpus": 1, "vram_gb": 14, "cpus": 4, "ram_gb": 24},
        {"name": "llama-70b-distributed", "gpus": 4, "vram_gb": 24, "cpus": 16, "ram_gb": 64},
        {"name": "diffusion-xl-gen", "gpus": 2, "vram_gb": 12, "cpus": 6, "ram_gb": 32},
        {"name": "gnn-graph-embed", "gpus": 1, "vram_gb": 6, "cpus": 4, "ram_gb": 16},
    ]

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """Reset internal random generator seed for exact reproducibility."""
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_trace(
        self,
        count: int = 20,
        arrival_pattern: str = "uniform",
        min_arrival_interval: float = 0.5,
        max_arrival_interval: float = 2.0,
        poisson_rate: float = 1.0,
        burst_size: int = 5,
        gpu_counts: Optional[List[int]] = None,
        gpu_memory_options_bytes: Optional[List[int]] = None,
        duration_range_seconds: Tuple[float, float] = (1.0, 10.0),
        priority_range: Tuple[int, int] = (0, 10),
    ) -> List[SyntheticWorkloadItem]:
        """Generate a synthetic workload trace dataset of specified count and arrival characteristics."""
        trace: List[SyntheticWorkloadItem] = []
        current_time = 0.0

        gpu_counts = gpu_counts or [1, 2, 4]
        vram_options = gpu_memory_options_bytes or [
            4 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            12 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            24 * 1024 * 1024 * 1024,
        ]

        for i in range(count):
            # 1. Compute arrival delay relative to trace start
            if i == 0:
                arrival_delay = 0.0
            else:
                if arrival_pattern == "uniform":
                    interval = self._rng.uniform(min_arrival_interval, max_arrival_interval)
                elif arrival_pattern == "poisson":
                    interval = self._rng.expovariate(poisson_rate)
                elif arrival_pattern == "burst":
                    interval = 0.0 if (i % burst_size != 0) else self._rng.uniform(min_arrival_interval, max_arrival_interval)
                elif arrival_pattern == "constant":
                    interval = min_arrival_interval
                else:
                    interval = self._rng.uniform(min_arrival_interval, max_arrival_interval)

                current_time += interval
                arrival_delay = current_time

            # 2. Select workload characteristics
            tmpl = self._rng.choice(self.WORKLOAD_TEMPLATES)
            gpus = self._rng.choice(gpu_counts)
            vram_bytes = self._rng.choice(vram_options)
            priority = self._rng.randint(priority_range[0], priority_range[1])
            duration = self._rng.uniform(duration_range_seconds[0], duration_range_seconds[1])

            req = ResourceRequirements(
                gpus=gpus,
                gpu_memory_bytes=vram_bytes,
                cpus=tmpl["cpus"],
                memory_bytes=tmpl["ram_gb"] * 1024 * 1024 * 1024,
            )

            spec = WorkloadSpec(
                name=f"{tmpl['name']}-{i+1}",
                resources=req,
                execution=WorkloadExecutionConfig(
                    entrypoint="python",
                    args=["-c", f"import time; time.sleep({duration:.2f})"],
                ),
            )

            item = SyntheticWorkloadItem(
                job_id=f"synth-job-{i+1:04d}",
                name=f"{tmpl['name']}-{i+1}",
                priority=priority,
                spec=spec,
                arrival_delay=round(arrival_delay, 3),
                duration_seconds=round(duration, 3),
            )
            trace.append(item)

        return trace


class TrialMetrics(BaseModel):
    """Execution performance metrics captured during a single benchmark trial."""

    makespan_seconds: float
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    scheduling_throughput: float
    queue_wait_times: List[float]
    mean_queue_wait_seconds: float
    p50_queue_wait_seconds: float
    p95_queue_wait_seconds: float
    vram_fragmentation_avg: float
    placement_efficiency: float
    avg_topology_score: float


class BenchmarkResult(BaseModel):
    """Statistical summary across multi-trial benchmark runs."""

    scheduler_name: str
    num_trials: int
    trials: List[TrialMetrics]

    makespan_mean: float
    makespan_std: float
    throughput_mean: float
    throughput_std: float
    mean_wait_time_mean: float
    mean_wait_time_std: float
    p95_wait_time_mean: float
    vram_fragmentation_mean: float
    vram_fragmentation_std: float
    placement_efficiency_mean: float
    topology_score_mean: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to serializable dictionary format."""
        return {
            "scheduler_name": self.scheduler_name,
            "num_trials": self.num_trials,
            "makespan_mean_sec": round(self.makespan_mean, 3),
            "makespan_std_sec": round(self.makespan_std, 3),
            "throughput_mean_jobs_per_sec": round(self.throughput_mean, 4),
            "throughput_std_jobs_per_sec": round(self.throughput_std, 4),
            "mean_wait_time_sec": round(self.mean_wait_time_mean, 3),
            "mean_wait_time_std_sec": round(self.mean_wait_time_std, 3),
            "p95_wait_time_sec": round(self.p95_wait_time_mean, 3),
            "vram_fragmentation_pct": round(self.vram_fragmentation_mean * 100, 2),
            "placement_efficiency_pct": round(self.placement_efficiency_mean * 100, 2),
            "avg_topology_score": round(self.topology_score_mean, 2),
        }


class BenchmarkHarness:
    """Multi-trial statistical evaluation harness for DRIGS scheduling policies."""

    def __init__(self, cluster_workers: Optional[List[WorkerInfo]] = None):
        self.cluster_workers = cluster_workers or self._create_default_simulated_cluster()

    def _create_default_simulated_cluster(self) -> List[WorkerInfo]:
        """Create a default synthetic 2-node cluster with 4x GPUs (24GB VRAM) per node."""
        workers: List[WorkerInfo] = []
        for w_idx in range(2):
            sim_backend = SimulatedBackend(
                num_gpus=4,
                vram_per_gpu_bytes=24 * 1024 * 1024 * 1024,
                model_name=f"NVIDIA RTX 4090 (Worker {w_idx})",
            )
            devices = sim_backend.discover_devices()
            worker = WorkerInfo(
                worker_id=f"worker-{w_idx}",
                hostname=f"node-{w_idx}.drigs.local",
                ip_address=f"192.168.1.{10 + w_idx}",
                devices=devices,
                total_cpus=16,
                total_memory_bytes=64 * 1024 * 1024 * 1024,
            )
            workers.append(worker)
        return workers

    def _build_cluster_state(self, workers: List[WorkerInfo]) -> ClusterState:
        """Construct a fresh ClusterState dictionary from worker list."""
        worker_dict = {w.worker_id: copy.deepcopy(w) for w in workers}
        return ClusterState(workers=worker_dict)

    def run_single_trial(
        self,
        scheduler: Scheduler,
        trace: List[SyntheticWorkloadItem],
        time_step: float = 0.5,
        max_simulation_steps: int = 10000,
    ) -> TrialMetrics:
        """Run a discrete-event simulated benchmark trial for a trace dataset and scheduler."""
        cluster_state = self._build_cluster_state(self.cluster_workers)

        # Track device state dynamically during simulation
        # worker_id -> device_id -> available_vram
        worker_gpu_vram: Dict[str, Dict[str, int]] = {}
        worker_gpu_total_vram: Dict[str, Dict[str, int]] = {}
        worker_cpu_alloc: Dict[str, int] = {w.worker_id: 0 for w in self.cluster_workers}
        worker_mem_alloc: Dict[str, int] = {w.worker_id: 0 for w in self.cluster_workers}

        for w in self.cluster_workers:
            worker_gpu_vram[w.worker_id] = {}
            worker_gpu_total_vram[w.worker_id] = {}
            for dev in w.devices:
                if dev.device_type == DeviceType.GPU:
                    worker_gpu_vram[w.worker_id][dev.device_id] = dev.available_memory_bytes
                    worker_gpu_total_vram[w.worker_id][dev.device_id] = dev.total_memory_bytes

        # Pending trace queue ordered by arrival_delay
        sorted_trace = sorted(trace, key=lambda item: item.arrival_delay)
        trace_idx = 0

        pending_jobs: Dict[str, Job] = {}  # job_id -> Job
        job_arrival_time: Dict[str, float] = {}  # job_id -> arrival_time
        job_duration: Dict[str, float] = {}  # job_id -> duration
        job_wait_time: Dict[str, float] = {}  # job_id -> queue_wait_seconds

        # Active executions: list of dict(finish_time, job_id, allocation, vram_req)
        active_running_jobs: List[Dict[str, Any]] = []

        completed_job_ids: List[str] = []
        failed_job_ids: List[str] = []
        topo_scores: List[float] = []

        sim_time = 0.0
        vram_frag_samples: List[float] = []

        step_count = 0

        while step_count < max_simulation_steps:
            step_count += 1

            # 1. Enqueue newly arrived trace items
            while trace_idx < len(sorted_trace) and sorted_trace[trace_idx].arrival_delay <= sim_time:
                item = sorted_trace[trace_idx]
                job = Job(
                    job_id=item.job_id,
                    name=item.name,
                    priority=item.priority,
                    spec=item.spec,
                    submitted_at=_now_utc(),
                )
                pending_jobs[job.id] = job
                job_arrival_time[job.id] = item.arrival_delay
                job_duration[job.id] = item.duration_seconds
                trace_idx += 1

            # 2. Check completed running jobs and release resources
            still_running: List[Dict[str, Any]] = []
            for run_info in active_running_jobs:
                if run_info["finish_time"] <= sim_time:
                    # Release resources
                    alloc: ResourceAllocation = run_info["allocation"]
                    w_id = alloc.worker_id
                    vram_req = run_info["vram_req"]
                    req_cpus = run_info["cpus_req"]
                    req_ram = run_info["ram_req"]

                    for dev_id in alloc.assigned_device_ids:
                        worker_gpu_vram[w_id][dev_id] += vram_req

                    worker_cpu_alloc[w_id] = max(0, worker_cpu_alloc[w_id] - req_cpus)
                    worker_mem_alloc[w_id] = max(0, worker_mem_alloc[w_id] - req_ram)

                    completed_job_ids.append(run_info["job_id"])
                else:
                    still_running.append(run_info)

            active_running_jobs = still_running

            # 3. Construct updated ClusterState view reflecting active allocations
            current_workers: Dict[str, WorkerInfo] = {}
            for w in self.cluster_workers:
                w_id = w.worker_id
                updated_devices: List[ComputeDevice] = []
                for dev in w.devices:
                    if dev.device_type == DeviceType.GPU:
                        avail = worker_gpu_vram[w_id][dev.device_id]
                        total = worker_gpu_total_vram[w_id][dev.device_id]
                        util = ((total - avail) / total) * 100.0 if total > 0 else 0.0
                        updated_dev = ComputeDevice(
                            device_id=dev.device_id,
                            device_type=dev.device_type,
                            vendor=dev.vendor,
                            model_name=dev.model_name,
                            total_memory_bytes=total,
                            available_memory_bytes=avail,
                            utilization_pct=util,
                            pcie_bus_id=dev.pcie_bus_id,
                            numa_node=dev.numa_node,
                            topology_tags=dev.topology_tags,
                        )
                    else:
                        updated_dev = dev
                    updated_devices.append(updated_dev)

                avail_cpus = w.total_cpus
                avail_mem = w.total_memory_bytes

                current_workers[w_id] = WorkerInfo(
                    worker_id=w_id,
                    hostname=w.hostname,
                    ip_address=w.ip_address,
                    devices=updated_devices,
                    total_cpus=avail_cpus,
                    total_memory_bytes=avail_mem,
                    status=DeviceState.HEALTHY,
                )

            current_cluster_state = ClusterState(workers=current_workers)

            # 4. Schedule pending jobs
            if pending_jobs:
                pending_list = list(pending_jobs.values())
                scheduled_allocations = scheduler.schedule(pending_list, current_cluster_state)

                for alloc in scheduled_allocations:
                    j_id = alloc.job_id
                    if j_id not in pending_jobs:
                        continue

                    job = pending_jobs.pop(j_id)
                    wait_sec = max(0.0, sim_time - job_arrival_time[j_id])
                    job_wait_time[j_id] = wait_sec

                    vram_req = job.spec.resources.gpu_memory_bytes
                    cpus_req = job.spec.resources.cpus
                    ram_req = job.spec.resources.memory_bytes
                    w_id = alloc.worker_id

                    # Deduct resources
                    for dev_id in alloc.assigned_device_ids:
                        worker_gpu_vram[w_id][dev_id] = max(0, worker_gpu_vram[w_id][dev_id] - vram_req)

                    worker_cpu_alloc[w_id] += cpus_req
                    worker_mem_alloc[w_id] += ram_req

                    # Calculate topology score for allocation if multi-GPU
                    if len(alloc.assigned_device_ids) > 1:
                        w_info = current_workers[w_id]
                        graph = TopologyGraph(devices=w_info.devices)
                        t_score = graph.get_matrix().get_group_topology_score(alloc.assigned_device_ids)
                        topo_scores.append(t_score)

                    finish_time = sim_time + job_duration[j_id]
                    active_running_jobs.append(
                        {
                            "job_id": j_id,
                            "finish_time": finish_time,
                            "allocation": alloc,
                            "vram_req": vram_req,
                            "cpus_req": cpus_req,
                            "ram_req": ram_req,
                        }
                    )

            # 5. Measure VRAM fragmentation sample at time step
            total_vram_capacity = sum(
                sum(worker_gpu_total_vram[w_id].values()) for w_id in worker_gpu_total_vram
            )
            total_vram_available = sum(
                sum(worker_gpu_vram[w_id].values()) for w_id in worker_gpu_vram
            )
            total_vram_used = total_vram_capacity - total_vram_available

            if total_vram_capacity > 0:
                vram_unused_ratio = (total_vram_capacity - total_vram_used) / total_vram_capacity
                vram_frag_samples.append(vram_unused_ratio)

            # Check loop termination: all trace jobs processed and active jobs finished
            if trace_idx >= len(sorted_trace) and not pending_jobs and not active_running_jobs:
                break

            sim_time += time_step

        makespan = sim_time
        total_jobs = len(trace)
        completed_count = len(completed_job_ids)
        failed_count = total_jobs - completed_count

        throughput = (completed_count / makespan) if makespan > 0 else 0.0

        wait_times = list(job_wait_time.values())
        if wait_times:
            mean_wait = statistics.mean(wait_times)
            sorted_waits = sorted(wait_times)
            p50_idx = int(len(sorted_waits) * 0.50)
            p95_idx = min(int(len(sorted_waits) * 0.95), len(sorted_waits) - 1)
            p50_wait = sorted_waits[p50_idx]
            p95_wait = sorted_waits[p95_idx]
        else:
            mean_wait = p50_wait = p95_wait = 0.0

        vram_frag_avg = statistics.mean(vram_frag_samples) if vram_frag_samples else 0.0
        placement_eff = (completed_count / total_jobs) if total_jobs > 0 else 0.0
        avg_topo_score = statistics.mean(topo_scores) if topo_scores else 100.0

        return TrialMetrics(
            makespan_seconds=round(makespan, 3),
            total_jobs=total_jobs,
            completed_jobs=completed_count,
            failed_jobs=failed_count,
            scheduling_throughput=round(throughput, 4),
            queue_wait_times=[round(w, 3) for w in wait_times],
            mean_queue_wait_seconds=round(mean_wait, 3),
            p50_queue_wait_seconds=round(p50_wait, 3),
            p95_queue_wait_seconds=round(p95_wait, 3),
            vram_fragmentation_avg=round(vram_frag_avg, 4),
            placement_efficiency=round(placement_eff, 4),
            avg_topology_score=round(avg_topo_score, 2),
        )

    def run_multi_trial(
        self,
        scheduler: Scheduler,
        scheduler_name: str,
        generator: WorkloadGenerator,
        num_trials: int = 5,
        trace_count: int = 20,
        time_step: float = 0.5,
        **generator_kwargs,
    ) -> BenchmarkResult:
        """Execute multi-trial statistical benchmark runs across reproducible synthetic workload traces."""
        trials: List[TrialMetrics] = []
        base_seed = generator.seed or 42

        for t_idx in range(num_trials):
            generator.set_seed(base_seed + t_idx * 1000)
            trace = generator.generate_trace(count=trace_count, **generator_kwargs)
            trial_metric = self.run_single_trial(scheduler, trace, time_step=time_step)
            trials.append(trial_metric)

        makespans = [tr.makespan_seconds for tr in trials]
        throughputs = [tr.scheduling_throughput for tr in trials]
        mean_waits = [tr.mean_queue_wait_seconds for tr in trials]
        p95_waits = [tr.p95_queue_wait_seconds for tr in trials]
        vram_frags = [tr.vram_fragmentation_avg for tr in trials]
        effs = [tr.placement_efficiency for tr in trials]
        topos = [tr.avg_topology_score for tr in trials]

        return BenchmarkResult(
            scheduler_name=scheduler_name,
            num_trials=num_trials,
            trials=trials,
            makespan_mean=statistics.mean(makespans),
            makespan_std=statistics.stdev(makespans) if num_trials > 1 else 0.0,
            throughput_mean=statistics.mean(throughputs),
            throughput_std=statistics.stdev(throughputs) if num_trials > 1 else 0.0,
            mean_wait_time_mean=statistics.mean(mean_waits),
            mean_wait_time_std=statistics.stdev(mean_waits) if num_trials > 1 else 0.0,
            p95_wait_time_mean=statistics.mean(p95_waits),
            vram_fragmentation_mean=statistics.mean(vram_frags),
            vram_fragmentation_std=statistics.stdev(vram_frags) if num_trials > 1 else 0.0,
            placement_efficiency_mean=statistics.mean(effs),
            topology_score_mean=statistics.mean(topos),
        )

    def compare_schedulers(
        self,
        schedulers: Dict[str, Scheduler],
        generator: WorkloadGenerator,
        num_trials: int = 5,
        trace_count: int = 20,
        time_step: float = 0.5,
        **generator_kwargs,
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple scheduling policies on exact identical reproducible synthetic workload traces."""
        results: Dict[str, BenchmarkResult] = {}
        for name, scheduler in schedulers.items():
            results[name] = self.run_multi_trial(
                scheduler=scheduler,
                scheduler_name=name,
                generator=generator,
                num_trials=num_trials,
                trace_count=trace_count,
                time_step=time_step,
                **generator_kwargs,
            )
        return results
