"""High-Scale Job Queue Stress and Bottleneck Benchmark for DRIGS.

Evaluates JobQueue admission throughput, scheduler decision latency, state transition overhead,
and SQLite database lock acquisition latency under high-concurrency queue depths (N = 500, 1000, 2500 jobs).
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Ensure DRIGS repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from drigs.core.models import Job, JobStatus
from drigs.core.queue import JobQueue
from drigs.core.storage import SQLiteStore
from drigs.scheduler.binpack import BinPackScheduler
from drigs.scheduler.fifo import FIFOScheduler
from drigs.scheduler.fit import BestFitScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.scheduler.priority import PriorityScheduler
from drigs.scheduler.topology_aware import TopologyAwareScheduler
from experiments.harness import BenchmarkHarness, WorkloadGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drigs-stress")


def benchmark_queue_scaling(
    job_counts: List[int] = [500, 1000, 2500],
    use_sqlite: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute high-scale queue throughput and latency benchmark across increasing job counts."""
    generator = WorkloadGenerator(seed=seed)
    harness = BenchmarkHarness()
    cluster_state = harness._build_cluster_state(harness.cluster_workers)

    schedulers = {
        "FIFO": FIFOScheduler(),
        "Priority": PriorityScheduler(),
        "BestFit": BestFitScheduler(),
        "MemoryAware": MemoryAwareScheduler(),
        "TopologyAware": TopologyAwareScheduler(),
    }

    scaling_results: Dict[str, Any] = {
        "benchmark": "High-Scale Job Queue Stress & Bottleneck Benchmark",
        "job_counts": job_counts,
        "use_sqlite": use_sqlite,
        "results_by_count": {},
    }

    print("\n" + "=" * 80)
    print(" DRIGS HIGH-SCALE JOB QUEUE STRESS BENCHMARK ")
    print("=" * 80)

    for N in job_counts:
        print(f"\n--- TESTING QUEUE DEPTH N = {N} JOBS ---")
        generator.set_seed(seed + N)
        trace_items = generator.generate_trace(count=N, arrival_pattern="uniform")

        # Convert trace items to Jobs
        jobs: List[Job] = [
            Job(
                job_id=item.job_id,
                name=item.name,
                priority=item.priority,
                spec=item.spec,
            )
            for item in trace_items
        ]

        # Setup storage backend if enabled
        temp_db_file = None
        storage = None
        if use_sqlite:
            temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db_file.close()
            storage = SQLiteStore(db_path=temp_db_file.name)

        jq = JobQueue(storage_backend=storage)

        # 1. Benchmark Enqueue Throughput
        start_enqueue = time.monotonic()
        for j in jobs:
            jq.enqueue(j)
        enqueue_elapsed = time.monotonic() - start_enqueue
        enqueue_throughput = N / enqueue_elapsed if enqueue_elapsed > 0 else 0.0
        print(f"  [Enqueue] Admitted {N} jobs in {enqueue_elapsed:.4f}s ({enqueue_throughput:.1f} jobs/sec)")

        # 2. Benchmark Scheduler Decision Scan Latency across Policy Engine
        sched_metrics: Dict[str, Any] = {}
        pending_jobs = jq.get_pending_jobs()

        for s_name, scheduler in schedulers.items():
            start_sched = time.monotonic()
            allocations = scheduler.schedule(pending_jobs, cluster_state)
            sched_elapsed = (time.monotonic() - start_sched) * 1000.0  # ms
            alloc_count = len(allocations)
            sched_metrics[s_name] = {
                "decision_latency_ms": round(sched_elapsed, 3),
                "allocations_made": alloc_count,
            }
            print(f"  [{s_name:13s}] Decision Latency: {sched_elapsed:7.2f} ms | Allocations: {alloc_count}")

        # 3. Benchmark State Transition & Storage Lock Latency
        start_status = time.monotonic()
        # Transition first 100 jobs to SCHEDULED then RUNNING
        sample_size = min(100, N)
        for j in jobs[:sample_size]:
            jq.update_job_status(j.id, JobStatus.SCHEDULED)
            jq.update_job_status(j.id, JobStatus.RUNNING)
        status_elapsed = (time.monotonic() - start_status) * 1000.0  # ms
        avg_transition_ms = status_elapsed / (sample_size * 2) if sample_size > 0 else 0.0

        print(f"  [State Locking] {sample_size * 2} Transitions in {status_elapsed:.2f} ms ({avg_transition_ms:.3f} ms/transition)")

        # Record dataset metrics for count N
        scaling_results["results_by_count"][str(N)] = {
            "num_jobs": N,
            "enqueue_elapsed_sec": round(enqueue_elapsed, 4),
            "enqueue_throughput_jobs_per_sec": round(enqueue_throughput, 1),
            "scheduler_metrics": sched_metrics,
            "avg_status_transition_ms": round(avg_transition_ms, 3),
        }

        # Cleanup SQLite DB file
        if temp_db_file and os.path.exists(temp_db_file.name):
            try:
                os.remove(temp_db_file.name)
            except OSError:
                pass

    print("\n" + "=" * 80 + "\n")
    return scaling_results


def main():
    parser = argparse.ArgumentParser(description="DRIGS High-Scale Job Queue Stress Benchmark")
    parser.add_argument(
        "--job-counts",
        nargs="+",
        type=int,
        default=[500, 1000, 2500],
        help="Job count scale levels to benchmark (default: 500 1000 2500)",
    )
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Disable SQLite persistence backend (pure in-memory benchmark)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kaggle_results/stress_test_results.json",
        help="Output JSON master dataset file path",
    )

    args = parser.parse_args()
    results = benchmark_queue_scaling(
        job_counts=args.job_counts,
        use_sqlite=not args.no_sqlite,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved queue stress test results to %s", out_path.resolve())


if __name__ == "__main__":
    main()
