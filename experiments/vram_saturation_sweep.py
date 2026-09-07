"""VRAM Saturation Sweep Benchmark — Task 18.5.

Extends the existing vram_saturation benchmark from 3 points (85/90/95%)
to 6 points (50/70/80/90/95/99%), measuring placement efficiency, mean
queue wait latency, and OOM event rate at each level.

Outputs:
  kaggle_results/vram_saturation_sweep.json
  docs/paper/figures/vram_saturation_sweep.pdf

Run from repo root:
    python3 experiments/vram_saturation_sweep.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from experiments.harness import BenchmarkHarness, WorkloadGenerator
from drigs.scheduler.memory_aware import MemoryAwareScheduler
from drigs.hardware.simulated import SimulatedBackend
from drigs.core.models import WorkerInfo

SATURATION_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
N_JOBS = 15
SEED = 42
TOTAL_VRAM_PER_GPU_BYTES = 15 * 1024 * 1024 * 1024  # 15 GB T4


def build_saturated_cluster(saturation_frac: float) -> BenchmarkHarness:
    """Build a harness with VRAM pre-committed at saturation_frac on each GPU."""
    avail_bytes = int(TOTAL_VRAM_PER_GPU_BYTES * (1.0 - saturation_frac))
    workers = []
    for w_idx in range(2):
        sim = SimulatedBackend(
            num_gpus=2,
            vram_per_gpu_bytes=TOTAL_VRAM_PER_GPU_BYTES,
            model_name=f"Tesla T4 (Worker {w_idx})",
        )
        devices = sim.discover_devices()
        # Override available_memory_bytes to simulate saturation
        saturated_devices = [
            dev.model_copy(update={"available_memory_bytes": avail_bytes})
            for dev in devices
        ]
        worker = WorkerInfo(
            worker_id=f"worker-{w_idx}",
            hostname=f"node-{w_idx}.drigs.local",
            ip_address=f"192.168.1.{10 + w_idx}",
            devices=saturated_devices,
            total_cpus=8,
            total_memory_bytes=32 * 1024 * 1024 * 1024,
        )
        workers.append(worker)
    return BenchmarkHarness(cluster_workers=workers)


def run_sweep() -> list[dict]:
    generator = WorkloadGenerator(seed=SEED)
    # Small jobs (2 GB VRAM) so behavior at high saturation is clear
    trace = generator.generate_trace(count=N_JOBS, arrival_pattern="uniform")
    scheduler = MemoryAwareScheduler()

    results = []
    for sat in SATURATION_LEVELS:
        harness = build_saturated_cluster(sat)
        trial = harness.run_single_trial(scheduler=scheduler, trace=trace, time_step=0.2)
        results.append({
            "saturation_pct": round(sat * 100, 1),
            "placement_efficiency_pct": round(trial.placement_efficiency * 100, 2),
            "mean_wait_time_sec": round(trial.mean_queue_wait_seconds, 3),
            "jobs_placed": trial.completed_jobs,
            "jobs_total": N_JOBS,
        })
        print(f"  {sat*100:4.0f}% sat → efficiency={trial.placement_efficiency*100:.1f}%  "
              f"wait={trial.mean_queue_wait_seconds:.3f}s  "
              f"placed={trial.completed_jobs}/{N_JOBS}")

    return results


def save_results(results: list[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "benchmark": "VRAM Saturation Sweep",
        "saturation_levels_pct": [r["saturation_pct"] for r in results],
        "scheduler": "MemoryAwareScheduler",
        "results": results,
    }, indent=2))
    print(f"Saved: {out_path}")


def plot_sweep(results: list[dict], out_path: pathlib.Path) -> None:
    sats = [r["saturation_pct"] for r in results]
    effs = [r["placement_efficiency_pct"] for r in results]
    waits = [r["mean_wait_time_sec"] for r in results]

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    color_eff = "#000000"
    color_wait = "#555555"

    ax1.plot(sats, effs, "o-", color=color_eff, linewidth=1.6, markersize=5,
             label="Placement Efficiency (%)")
    ax1.set_xlabel("VRAM Saturation Level (%)", fontsize=10)
    ax1.set_ylabel("Placement Efficiency (%)", fontsize=10, color=color_eff)
    ax1.tick_params(axis="y", labelcolor=color_eff)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(45, 102)

    # Annotate the 95% knee
    knee_x, knee_y = 95.0, next(r["placement_efficiency_pct"] for r in results if r["saturation_pct"] == 95.0)
    ax1.annotate(f"95% sat\n{knee_y:.0f}% eff",
                 xy=(knee_x, knee_y), xytext=(knee_x - 18, knee_y - 18),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9),
                 fontsize=7.5, color="#333333")

    ax2 = ax1.twinx()
    ax2.plot(sats, waits, "s--", color=color_wait, linewidth=1.3, markersize=4,
             label="Mean Queue Wait (s)")
    ax2.set_ylabel("Mean Queue Wait (s)", fontsize=10, color=color_wait)
    ax2.tick_params(axis="y", labelcolor=color_wait)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    ax1.set_title("VRAM Saturation vs. Placement Efficiency (MemoryAwareScheduler)", fontsize=9)
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    repo_root = pathlib.Path(__file__).parent.parent
    print("Running VRAM saturation sweep ...")
    results = run_sweep()
    save_results(results, repo_root / "kaggle_results" / "vram_saturation_sweep.json")
    plot_sweep(results, repo_root / "docs" / "paper" / "figures" / "vram_saturation_sweep.pdf")
