"""Re-run Experiment A with identical parameters as the original Kaggle run,
collecting per-job queue wait-time distributions for each scheduler.

The original experiment uses:
  seed=42, num_trials=5, trace_count=15, arrival_pattern="uniform", time_step=0.2
  PriorityScheduler(aging_factor=0.05)
  Internal seeds per trial: 42, 1042, 2042, 3042, 4042

The harness guarantees that all schedulers see identical arrival sequences within
each trial (same generator seed). This is the controlled variable in Table 1.

Outputs:
  kaggle_results/exp_a_per_job_waits.json  — raw per-job waits per (trial, policy)
  docs/paper/figures/scheduler_cdf.pdf     — CDF figure for embedding in paper

Run from repo root:
    python3 experiments/rerun_exp_a_with_seeds.py
"""

import json
import pathlib
import sys
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.harness import BenchmarkHarness, WorkloadGenerator
from drigs.scheduler.fifo import FIFOScheduler
from drigs.scheduler.priority import PriorityScheduler
from drigs.scheduler.fit import BestFitScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.scheduler.topology_aware import TopologyAwareScheduler

# Exact parameters matching the original run_experiment_a call
SEED = 42
NUM_TRIALS = 5
TRACE_COUNT = 15
TIME_STEP = 0.2
ARRIVAL_PATTERN = "uniform"

SCHEDULERS: dict[str, object] = {
    "FIFO": FIFOScheduler(),
    "Priority": PriorityScheduler(aging_factor=0.05),
    "BestFit": BestFitScheduler(),
    "MemoryAware": MemoryAwareScheduler(),
    "TopologyAware": TopologyAwareScheduler(),
}

STYLES: dict[str, dict] = {
    "FIFO":         {"color": "#000000", "linestyle": "-",             "linewidth": 1.6},
    "Priority":     {"color": "#444444", "linestyle": "--",            "linewidth": 1.6},
    "BestFit":      {"color": "#777777", "linestyle": "-.",            "linewidth": 1.4},
    "MemoryAware":  {"color": "#000000", "linestyle": ":",             "linewidth": 1.4},
    "TopologyAware":{"color": "#444444", "linestyle": (0, (3, 1, 1, 1)), "linewidth": 1.4},
}


def collect_per_job_waits() -> dict[str, list[float]]:
    """Run compare_schedulers with original parameters, return per-job wait times."""
    generator = WorkloadGenerator(seed=SEED)
    harness = BenchmarkHarness()

    results = harness.compare_schedulers(
        schedulers=SCHEDULERS,
        generator=generator,
        num_trials=NUM_TRIALS,
        trace_count=TRACE_COUNT,
        time_step=TIME_STEP,
        arrival_pattern=ARRIVAL_PATTERN,
    )

    # Pool per-job wait times across all trials for each scheduler
    per_scheduler: dict[str, list[float]] = {}
    for sched_name, bench_result in results.items():
        pooled: list[float] = []
        for trial in bench_result.trials:
            pooled.extend(trial.queue_wait_times)
        per_scheduler[sched_name] = pooled

        p95_idx = int(0.95 * len(sorted(pooled)))
        p95_val = sorted(pooled)[p95_idx] if pooled else 0.0
        print(f"  {sched_name:14s}: n={len(pooled):5d}  "
              f"mean={st.mean(pooled):.3f}s  P95={p95_val:.3f}s  "
              f"(table1 P95={bench_result.p95_wait_time_mean:.3f}s)")

    return per_scheduler


def save_raw_waits(waits: dict[str, list[float]], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "Experiment A — per-job wait times",
        "seed": SEED,
        "num_trials": NUM_TRIALS,
        "trace_count": TRACE_COUNT,
        "arrival_pattern": ARRIVAL_PATTERN,
        "internal_trial_seeds": [SEED + i * 1000 for i in range(NUM_TRIALS)],
        "waits_per_scheduler": waits,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_path}")


def plot_cdf(waits: dict[str, list[float]], out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    for sched_name, wait_list in waits.items():
        if not wait_list:
            continue
        sorted_w = sorted(wait_list)
        n = len(sorted_w)
        cdf = [(i + 1) / n for i in range(n)]
        ax.plot(sorted_w, cdf, label=sched_name, **STYLES[sched_name])

    ax.set_xlabel("Queue Wait Time (s)", fontsize=10)
    ax.set_ylabel("Cumulative Fraction of Jobs", fontsize=10)
    ax.set_title(
        "CDF of Per-Job Queue Wait Time by Scheduler\n"
        rf"({TRACE_COUNT} jobs $\times$ {NUM_TRIALS} trials, uniform arrivals, dual T4)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Annotate P95 lines for FIFO and Priority to illustrate HoL-blocking gap
    for sched_name in ("FIFO", "Priority"):
        wait_list = waits.get(sched_name, [])
        if not wait_list:
            continue
        sorted_w = sorted(wait_list)
        p95_idx = min(int(0.95 * len(sorted_w)), len(sorted_w) - 1)
        p95_val = sorted_w[p95_idx]
        ax.axvline(p95_val, color=STYLES[sched_name]["color"],
                   linestyle=":", linewidth=0.9, alpha=0.7)
        y_offset = 0.88 if sched_name == "FIFO" else 0.80
        ax.text(p95_val + 0.02, y_offset,
                f"{sched_name} P95={p95_val:.2f}s",
                fontsize=7, color=STYLES[sched_name]["color"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CDF figure: {out_path}")


if __name__ == "__main__":
    repo_root = pathlib.Path(__file__).parent.parent

    print(f"Running Experiment A  "
          f"(seed={SEED}, num_trials={NUM_TRIALS}, "
          f"trace_count={TRACE_COUNT}, pattern={ARRIVAL_PATTERN}) ...")
    waits = collect_per_job_waits()

    save_raw_waits(waits, repo_root / "kaggle_results" / "exp_a_per_job_waits.json")
    plot_cdf(waits, repo_root / "docs" / "paper" / "figures" / "scheduler_cdf.pdf")
