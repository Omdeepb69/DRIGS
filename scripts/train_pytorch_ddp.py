"""Real Multi-Process PyTorch DistributedDataParallel (DDP) Benchmark for DRIGS.

Executes real PyTorch DDP multi-process training across 2 ranks (dual Tesla T4 GPUs or CPU Gloo fallback)
measuring rank synchronization latency, step execution time (ms/step), and all-reduce throughput.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

# Ensure DRIGS repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Rank %(rank)s] %(message)s")
logger = logging.getLogger("drigs-ddp")


def run_ddp_training_rank(
    epochs: int = 5,
    batch_size: int = 32,
    hidden_dim: int = 512,
    backend_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute PyTorch DistributedDataParallel (DDP) training loop on the local rank."""
    try:
        import torch
        import torch.distributed as dist
        import torch.nn as nn
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError as err:
        print(f"⚠️ PyTorch distributed unavailable ({err}). Exiting.")
        return {"status": "UNAVAILABLE", "error": str(err)}

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Select PyTorch distributed backend: NCCL for CUDA GPUs, Gloo for CPU
    use_cuda = torch.cuda.is_available()
    if backend_name:
        dist_backend = backend_name
    else:
        dist_backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.device_count() > local_rank else "cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"🚀 Initializing PyTorch DDP Process Group...")
    print(f"  - Rank: {rank} / World Size: {world_size}")
    print(f"  - Master Rendezvous: {master_addr}:{master_port}")
    print(f"  - Backend: {dist_backend.upper()}")
    print(f"  - Active Compute Device: {device}")

    # Initialize PyTorch DDP process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=dist_backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
        )

    # Define deep neural network model
    model = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10),
    ).to(device)

    # Wrap model with DistributedDataParallel
    if use_cuda:
        ddp_model = DDP(model, device_ids=[local_rank] if torch.cuda.device_count() > local_rank else None)
    else:
        ddp_model = DDP(model)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    step_times_ms = []
    allreduce_latencies_ms = []

    print(f"  - Starting DDP Training Epochs ({epochs} epochs, Batch Size {batch_size})...")

    for epoch in range(1, epochs + 1):
        step_start = time.monotonic()

        inputs = torch.randn(batch_size, hidden_dim, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        step_elapsed_ms = (time.monotonic() - step_start) * 1000.0
        step_times_ms.append(step_elapsed_ms)

        # Measure explicit PyTorch Dist All-Reduce tensor synchronization latency
        sync_tensor = torch.tensor([loss.item()], device=device)
        sync_start = time.monotonic()
        dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
        sync_elapsed_ms = (time.monotonic() - sync_start) * 1000.0
        allreduce_latencies_ms.append(sync_elapsed_ms)

        avg_loss = (sync_tensor / world_size).item()
        print(f"Epoch [{epoch:2d}/{epochs}] | Rank {rank} Loss: {loss.item():.4f} | Synced Loss: {avg_loss:.4f} | Step Time: {step_elapsed_ms:.2f}ms | AllReduce: {sync_elapsed_ms:.2f}ms")

        time.sleep(0.05)

    mean_step_time_ms = sum(step_times_ms) / len(step_times_ms) if step_times_ms else 0.0
    mean_allreduce_ms = sum(allreduce_latencies_ms) / len(allreduce_latencies_ms) if allreduce_latencies_ms else 0.0

    print(f"✅ Rank {rank} DDP Training Completed Successfully!")
    print(f"  - Mean Step Latency: {mean_step_time_ms:.2f} ms/step")
    print(f"  - Mean AllReduce Latency: {mean_allreduce_ms:.2f} ms\n")

    results = {
        "rank": rank,
        "world_size": world_size,
        "backend": dist_backend,
        "epochs": epochs,
        "mean_step_time_ms": round(mean_step_time_ms, 2),
        "mean_allreduce_latency_ms": round(mean_allreduce_ms, 2),
        "device": str(device),
        "status": "SUCCESS",
    }

    if dist.is_initialized():
        dist.destroy_process_group()

    return results


def run_ddp_benchmark_suite(
    epochs: int = 5,
    output_path: str = "kaggle_results/ddp_benchmark_results.json",
) -> Dict[str, Any]:
    """Launch 2-process PyTorch DDP benchmark via DistributedBackend or in-process rank simulation."""
    from drigs.core.models import (
        ComputeDevice,
        DeviceType,
        Job,
        JobStatus,
        ResourceAllocation,
        WorkloadDistributionConfig,
        WorkloadExecutionConfig,
        WorkloadSpec,
    )
    from drigs.execution.distributed import DistributedBackend

    print("\n" + "=" * 80)
    print(" DRIGS PYTORCH DISTRIBUTED DATA PARALLEL (DDP) BENCHMARK ")
    print("=" * 80)

    # 1. Construct multi-rank DDP Job spec
    spec = WorkloadSpec(
        name="pytorch-ddp-benchmark",
        execution=WorkloadExecutionConfig(
            entrypoint="python3",
            args=[
                __file__,
                "--worker-rank",
                "--epochs",
                str(epochs),
            ],
        ),
        distribution=WorkloadDistributionConfig(
            framework="pytorch",
            world_size=2,
            master_addr="127.0.0.1",
        ),
    )

    job = Job(job_id="job-ddp-bench-001", name="pytorch-ddp-benchmark", spec=spec)

    alloc = ResourceAllocation(
        job_id=job.id,
        worker_id="worker-0",
        assigned_device_ids=["gpu-0", "gpu-1"],
        assigned_cpu_cores=[0, 1, 2, 3],
        memory_bytes=4 * 1024**3,
    )

    backend = DistributedBackend()
    print("Launching 2-rank PyTorch DDP processes via DRIGS DistributedBackend...")
    handle = backend.launch(job, alloc)

    start_wait = time.monotonic()
    while time.monotonic() - start_wait < 60.0:
        status = backend.get_status(handle)
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.5)

    final_status = backend.get_status(handle)
    logs = backend.read_logs(handle)
    print("\n--- DDP PROCESS LOG OUTPUT ---")
    print(logs[:1500] if len(logs) > 1500 else logs)

    benchmark_summary = {
        "benchmark": "PyTorch DistributedDataParallel (DDP) Benchmark",
        "job_id": job.id,
        "world_size": 2,
        "final_status": final_status.value,
        "logs": logs,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print(f"\nSaved DDP benchmark results to {out_file.resolve()}")
    print("=" * 80 + "\n")
    return benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP Multi-Process Benchmark for DRIGS")
    parser.add_argument("--worker-rank", action="store_true", help="Execute single DDP worker rank mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--backend", type=str, default=None, help="PyTorch dist backend (nccl or gloo)")
    parser.add_argument("--output", type=str, default="kaggle_results/ddp_benchmark_results.json", help="Output path")

    args = parser.parse_args()

    if args.worker_rank:
        run_ddp_training_rank(epochs=args.epochs, backend_name=args.backend)
    else:
        run_ddp_benchmark_suite(epochs=args.epochs, output_path=args.output)


if __name__ == "__main__":
    main()
