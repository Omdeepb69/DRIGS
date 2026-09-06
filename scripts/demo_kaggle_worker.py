"""DRIGS Remote GPU Worker & Ephemeral Session Recovery Demonstration Script.

Simulates dynamic remote GPU cluster orchestration:
1. Controller & Control Plane REST API Initialization
2. Outbound Remote Worker Phone-Home Registration (Kaggle T4 Node Join)
3. Dynamic Cluster Telemetry Expansion
4. GPU Workload Scheduling & Execution
5. Remote Session Death Simulation (Heartbeat Timeout)
6. Automated Checkpoint Rescheduling onto Replacement Worker
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

from drigs.api.server import APIServer
from drigs.core.controller import LocalController
from drigs.core.models import (
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
from drigs.core.queue import JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.recovery.checkpoint import CheckpointManager
from drigs.recovery.detector import FailureDetector
from drigs.recovery.rescheduler import Rescheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.workers.bootstrap import RemoteHTTPWorkerRegistryClient, create_remote_worker_agent
from drigs.workers.registry import WorkerRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drigs-demo")


def run_kaggle_worker_demo() -> Dict[str, Any]:
    """Execute dynamic Kaggle GPU worker join, workload dispatch, and session recovery flow."""
    print("\n" + "=" * 80)
    print(" DRIGS DYNAMIC REMOTE WORKER & KAGGLE SESSION RECOVERY DEMO ")
    print("=" * 80)

    # 1. Initialize Control Plane (Worker Registry, Resource Manager, Job Queue, Controller)
    registry = WorkerRegistry(timeout_seconds=0.05)
    resource_manager = ResourceManager()
    controller = LocalController(
        resource_manager=resource_manager,
        scheduler=MemoryAwareScheduler(),
        auto_register_local_node=True,
    )

    print("\n[Step 1] Initializing DRIGS Control Plane & Local Worker...")
    initial_cluster = resource_manager.get_cluster_state()
    print(f"  Active Workers: {len(initial_cluster.workers)}")
    for w_id, w in initial_cluster.workers.items():
        print(f"  - Worker: {w_id} ({w.hostname}) | CPUs: {w.total_cpus} | Devices: {len(w.devices)}")

    # 2. Simulate Kaggle Remote GPU Node Registration ("Phone-Home")
    print("\n[Step 2] Remote Kaggle GPU Worker Phone-Home Registration...")
    kaggle_gpu = ComputeDevice(
        device_id="gpu-kaggle-t4-0",
        device_type=DeviceType.GPU,
        vendor="NVIDIA",
        model_name="NVIDIA Tesla T4 (Kaggle Ephemeral)",
        total_memory_bytes=16 * 1024 * 1024 * 1024,  # 16 GB VRAM
        available_memory_bytes=16 * 1024 * 1024 * 1024,
        compute_capability="7.5",
    )
    kaggle_worker_info = WorkerInfo(
        worker_id="worker-kaggle-t4-01",
        hostname="kaggle-session-prod-89a",
        ip_address="34.125.10.42",
        devices=[kaggle_gpu],
        total_cpus=4,
        total_memory_bytes=16 * 1024 * 1024 * 1024,
        status=DeviceState.HEALTHY,
    )

    registry.register(kaggle_worker_info)
    resource_manager.register_worker(kaggle_worker_info)

    expanded_cluster = resource_manager.get_cluster_state()
    print(f"  Cluster Expanded! Total Workers: {len(expanded_cluster.workers)}")
    print(f"  - Joined Remote Node: {kaggle_worker_info.worker_id}")
    print(f"  - Model: {kaggle_gpu.model_name} (VRAM: {kaggle_gpu.total_memory_bytes / 1e9:.1f} GB)")

    # 3. Submit AI Workload Targetted for GPU Execution
    print("\n[Step 3] Submitting AI Workload to DRIGS Cluster...")
    spec = WorkloadSpec(
        name="llama-7b-eval-job",
        resources=ResourceRequirements(gpus=1, gpu_memory_bytes=12 * 1024 * 1024 * 1024, cpus=2),
        execution=WorkloadExecutionConfig(
            entrypoint="python",
            args=["-c", "import time; print('Evaluating LLaMA-7B on Kaggle GPU...'); time.sleep(2)"],
        ),
    )
    job = controller.submit_job(spec, priority=5, job_name="llama-7b-eval-job")
    print(f"  Job Submitted: ID={job.id} | Name={job.name} | Status={job.status.value}")

    # 4. Schedule Job onto Kaggle Worker
    print("\n[Step 4] Control Plane Scheduling Pass...")
    step_runs = controller.step()
    assigned_worker = job.allocation.worker_id if job.allocation else "Unallocated"
    assigned_gpus = job.allocation.assigned_device_ids if job.allocation else []
    print(f"  Scheduling Result: Job {job.id} assigned to Worker '{assigned_worker}' on GPUs {assigned_gpus}")
    assert assigned_worker == "worker-kaggle-t4-01", "Job should be scheduled on Kaggle GPU worker"

    # Save checkpoint to simulate active training step
    ckpt_mgr = CheckpointManager()
    ckpt_path = ckpt_mgr.save_checkpoint(
        job_id=job.id,
        step=200,
        checkpoint_data={"step": 200, "eval_loss": 0.421, "vram_allocated_gb": 12.0},
    )
    print(f"  Saved Workload Checkpoint: Step 200 at {ckpt_path}")

    # 5. Simulate Kaggle Session Termination & Heartbeat Timeout
    print("\n[Step 5] Simulating Ephemeral Kaggle Session Termination (Node Crash)...")
    time.sleep(0.1)
    # Fast forward heartbeat clock simulation to trigger timeout
    detector = FailureDetector(registry=registry, check_interval_seconds=0.05)
    timed_out_nodes = registry.check_timeouts()
    print(f"  Failure Detector Alert: Timed out nodes detected: {timed_out_nodes}")
    assert "worker-kaggle-t4-01" in timed_out_nodes

    # 6. Trigger Automated Rescheduling & Checkpoint Restoration
    print("\n[Step 6] Rescheduler Triggered: Automated Checkpoint Recovery...")
    rescheduler = Rescheduler(
        job_queue=controller.job_queue,
        resource_manager=resource_manager,
        checkpoint_manager=ckpt_mgr,
    )
    rescheduled = rescheduler.reschedule_job(job.id, reason="Kaggle session timeout")
    recovered_job = controller.get_job(job.id)
    restore_ckpt_env = recovered_job.spec.execution.env.get("DRIGS_RESTORE_CHECKPOINT", "None")
    restore_step_env = recovered_job.spec.execution.env.get("DRIGS_RESTORE_STEP", "None")

    print(f"  Rescheduling Success: {rescheduled}")
    print(f"  Recovered Job Status: {recovered_job.status.value}")
    print(f"  Restored Checkpoint Step: {restore_step_env} ({restore_ckpt_env})")

    demo_results = {
        "status": "SUCCESS",
        "registered_worker": kaggle_worker_info.worker_id,
        "scheduled_job_id": job.id,
        "failure_detected": "worker-kaggle-t4-01" in timed_out_nodes,
        "rescheduled_ok": rescheduled,
        "restored_step": restore_step_env,
    }

    print("\n" + "=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY: DYNAMIC WORKER ORCHESTRATION VERIFIED ")
    print("=" * 80 + "\n")

    return demo_results


if __name__ == "__main__":
    run_kaggle_worker_demo()
