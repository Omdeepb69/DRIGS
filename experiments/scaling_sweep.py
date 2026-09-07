"""Control-Plane Scaling Sweep Benchmark — Task 18.6.

Extends the stress test to N = 50, 100, 250, 500, 750, 1000 jobs,
measuring scheduler decision latency and enqueue throughput at each N
for all five scheduling policies.

Outputs:
  kaggle_results/scaling_sweep.json
  docs/paper/figures/scaling_sweep.pdf

Run from repo root:
    python3 experiments/scaling_sweep.py
"""

import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.harness import BenchmarkHarness, WorkloadGenerator
from drigs.scheduler.fifo import FIFOScheduler
from drigs.scheduler.priority import PriorityScheduler
from drigs.scheduler.fit import BestFitScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.scheduler.topology_aware import TopologyAwareScheduler

N_VALUES = [50, 100, 250, 500, 750, 1000]
SEED = 42
TIME_STEP = 0.2

SCHEDULERS: dict[str, object] = {
    "FIFO": FIFOScheduler(),
    "Priority": PriorityScheduler(aging_factor=0.05),
    "BestFit": BestFitScheduler(),
    "MemoryAware": MemoryAwareScheduler(),
    "TopologyAware": TopologyAwareScheduler(),
}

# Line styles — black/gray only
STYLES: dict[str, dict] = {
    "FIFO":         {"color": "#000000", "linestyle": "-",             "linewidth": 1.4, "marker": "o", "markersize": 4},
    "Priority":     {"color": "#333333", "linestyle": "--",            "linewidth": 1.4, "marker": "s", "markersize": 4},
    "BestFit":      {"color": "#666666", "linestyle": "-.",            "linewidth": 1.3, "marker": "^", "markersize": 4},
    "MemoryAware":  {"color": "#000000", "linestyle": ":",             "linewidth": 1.3, "marker": "D", "markersize": 4},
    "TopologyAware":{"color": "#444444", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.3, "marker": "v", "markersize": 4},
}


def run_sweep() -> dict[str, list]:
    """For each N in N_VALUES, run one trial per scheduler, record latency + throughput."""
    generator = WorkloadGenerator(seed=SEED)
    harness = BenchmarkHarness()

    # Results: {scheduler_name: [(N, decision_latency_ms, throughput)]}
    data: dict[str, list[tuple[int, float, float]]] = {name: [] for name in SCHEDULERS}

    for n in N_VALUES:
        trace = generator.generate_trace(count=n, arrival_pattern="uniform")
        print(f"\n  N={n}:")
        for sched_name, scheduler in SCHEDULERS.items():
            t0 = time.perf_counter()
            trial = harness.run_single_trial(scheduler=scheduler, trace=trace, time_step=TIME_STEP)
            elapsed_s = time.perf_counter() - t0
            # Decision latency = total elapsed / number of scheduling decisions
            n_decisions = max(trial.completed_jobs, 1)
            latency_ms = (elapsed_s * 1000) / n_decisions
            throughput = trial.scheduling_throughput
            data[sched_name].append((n, latency_ms, throughput))
            print(f"    {sched_name:14s}: latency={latency_ms:.2f}ms  throughput={throughput:.2f} jobs/s")

    return data


def save_results(data: dict[str, list], out_path: pathlib.Path) -> None:
    serializable = {
        name: [{"n": n, "decision_latency_ms": lat, "throughput_jobs_per_sec": tp}
               for n, lat, tp in points]
        for name, points in data.items()
    }
    payload = {
        "benchmark": "Control-Plane Scaling Sweep",
        "n_values": N_VALUES,
        "seed": SEED,
        "results": serializable,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


def plot_sweep(data: dict[str, list], out_path: pathlib.Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for sched_name, points in data.items():
        ns = [p[0] for p in points]
        latencies = [p[1] for p in points]
        throughputs = [p[2] for p in points]
        style = STYLES[sched_name]
        ax1.plot(ns, latencies, label=sched_name, **style)
        ax2.plot(ns, throughputs, label=sched_name, **style)

    ax1.set_xlabel("Job Queue Depth N", fontsize=10)
    ax1.set_ylabel("Scheduling Decision Latency (ms)", fontsize=10)
    ax1.set_title("Decision Latency vs. Queue Depth", fontsize=9)
    ax1.legend(fontsize=7.5)
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.set_xlim(0, 1050)

    ax2.set_xlabel("Job Queue Depth N", fontsize=10)
    ax2.set_ylabel("Enqueue Throughput (jobs/s)", fontsize=10)
    ax2.set_title("Throughput vs. Queue Depth", fontsize=9)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax2.set_xlim(0, 1050)

    fig.suptitle("DRIGS Control-Plane Scalability (SimulatedBackend, dual T4 topology)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    repo_root = pathlib.Path(__file__).parent.parent
    print("Running control-plane scaling sweep ...")
    data = run_sweep()
    save_results(data, repo_root / "kaggle_results" / "scaling_sweep.json")
    plot_sweep(data, repo_root / "docs" / "paper" / "figures" / "scaling_sweep.pdf")
