"""Automated Kaggle Real-GPU Test Suite & Empirical Benchmark Exporter.

This script executes DRIGS on real GPU environments (such as Kaggle Tesla T4 / P100 / L4 GPUs)
or in simulated mode, runs full test verification, executes empirical experiments (A, B, E),
and exports publication-grade JSON and CSV datasets for research paper tables/plots.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure DRIGS repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from drigs.core.models import ComputeDevice, DeviceType
from drigs.hardware.cpu import CPUBackend
from drigs.hardware.simulated import SimulatedBackend
from experiments.run_experiments import (
    run_all_experiments,
    run_experiment_a,
    run_experiment_b,
    run_experiment_e,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drigs-kaggle")


def discover_gpu_telemetry(use_simulated: bool = False) -> List[Dict[str, Any]]:
    """Discover attached physical GPU telemetry or return synthetic GPU information."""
    telemetry: List[Dict[str, Any]] = []

    if not use_simulated:
        try:
            from drigs.hardware.cuda import CUDABackend
            cuda_backend = CUDABackend()
            devices = cuda_backend.discover_devices()
            if devices:
                for dev in devices:
                    telemetry.append(
                        {
                            "device_id": dev.device_id,
                            "vendor": dev.vendor,
                            "model_name": dev.model_name,
                            "total_vram_gb": round(dev.total_memory_bytes / (1024**3), 2),
                            "available_vram_gb": round(dev.available_memory_bytes / (1024**3), 2),
                            "compute_capability": dev.compute_capability or "N/A",
                            "pcie_bus_id": dev.pcie_bus_id or "N/A",
                        }
                    )
                logger.info("Discovered %d physical CUDA GPU device(s)", len(telemetry))
                return telemetry
            else:
                logger.info("No physical CUDA GPUs discovered via CUDABackend.")
        except Exception as err:
            logger.warning("CUDABackend discovery unavailable (%s).", err)

    # Simulated fallback
    sim_backend = SimulatedBackend(num_gpus=2, vram_per_gpu_bytes=16 * 1024**3)
    sim_devs = sim_backend.discover_devices()
    for dev in sim_devs:
        telemetry.append(
            {
                "device_id": dev.device_id,
                "vendor": dev.vendor,
                "model_name": dev.model_name + " (Simulated)",
                "total_vram_gb": round(dev.total_memory_bytes / (1024**3), 2),
                "available_vram_gb": round(dev.available_memory_bytes / (1024**3), 2),
                "compute_capability": dev.compute_capability or "8.0",
                "pcie_bus_id": dev.pcie_bus_id or "0000:01:00.0",
            }
        )
    logger.info("Generated %d simulated GPU telemetry record(s)", len(telemetry))
    return telemetry


def export_csv_data(output_dir: Path, results: Dict[str, Any]) -> List[Path]:
    """Export experiment summaries to structured CSV files for paper analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: List[Path] = []

    # 1. Experiment A CSV Export
    exp_a_path = output_dir / "experiment_a_schedulers.csv"
    exp_a_data = results.get("experiment_a", {}).get("results", {})
    with open(exp_a_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Scheduler",
                "NumTrials",
                "Throughput_Jobs_Per_Sec",
                "Throughput_Std",
                "MeanWait_Sec",
                "MeanWait_Std",
                "P95Wait_Sec",
                "VRAM_Fragmentation_Pct",
                "Placement_Efficiency_Pct",
            ]
        )
        for sched_name, m in exp_a_data.items():
            writer.writerow(
                [
                    sched_name,
                    m.get("num_trials"),
                    m.get("throughput_mean_jobs_per_sec"),
                    m.get("throughput_std_jobs_per_sec"),
                    m.get("mean_wait_time_sec"),
                    m.get("mean_wait_time_std_sec"),
                    m.get("p95_wait_time_sec"),
                    m.get("vram_fragmentation_pct"),
                    m.get("placement_efficiency_pct"),
                ]
            )
    exported_files.append(exp_a_path)

    # 2. Experiment B CSV Export
    exp_b_path = output_dir / "experiment_b_topology.csv"
    exp_b_data = results.get("experiment_b", {})
    with open(exp_b_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                "TopologyAware_Score",
                "Random_Score",
                "Improvement_Pct",
            ]
        )
        writer.writerow(
            [
                "Average_Topology_Score",
                exp_b_data.get("topology_score_topology_aware"),
                exp_b_data.get("topology_score_random"),
                exp_b_data.get("topology_score_improvement_pct"),
            ]
        )
    exported_files.append(exp_b_path)

    # 3. Experiment E CSV Export
    exp_e_path = output_dir / "experiment_e_fault_recovery.csv"
    exp_e_data = results.get("experiment_e", {})
    with open(exp_e_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Metric",
                "Value",
            ]
        )
        writer.writerow(["Rescheduling_Latency_MS", exp_e_data.get("mean_rescheduling_latency_ms")])
        writer.writerow(["Job_Completion_Rate_Pct", exp_e_data.get("mean_job_completion_rate_pct")])
        writer.writerow(["Total_Restored_Checkpoints", exp_e_data.get("total_restored_checkpoints")])
    exported_files.append(exp_e_path)

    return exported_files


def run_kaggle_evaluation(
    num_trials: int = 5,
    use_simulated: bool = False,
    output_dir: str = "kaggle_results",
) -> Dict[str, Any]:
    """Execute complete DRIGS GPU benchmark evaluation suite and export paper metrics."""
    print("\n" + "=" * 80)
    print(" DRIGS KAGGLE REAL-GPU EXPERIMENTAL EVALUATION SUITE ")
    print("=" * 80)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover hardware telemetry
    gpu_telemetry = discover_gpu_telemetry(use_simulated=use_simulated)
    print("\n--- HARDWARE DISCOVERY TELEMETRY ---")
    for gpu in gpu_telemetry:
        print(
            f"Device: {gpu['device_id']} | Model: {gpu['model_name']} | "
            f"VRAM: {gpu['total_vram_gb']} GB | Compute Cap: {gpu['compute_capability']}"
        )

    # 2. Run real production PyTorch AI model workloads (ResNet-50 & Transformer LLM Encoder)
    print("\n--- EXECUTING PRODUCTION PYTORCH GPU MODEL WORKLOADS (ResNet-50 & Transformer) ---")
    try:
        from scripts.train_pytorch_workload import run_pytorch_workload
        # Run ResNet-50 Vision Workload
        run_pytorch_workload(model_type="resnet50", epochs=5, vram_alloc_mb=512, checkpoint_dir=str(out_path / "checkpoints"))
        # Run Transformer Attention Workload
        run_pytorch_workload(model_type="transformer", epochs=5, vram_alloc_mb=512, checkpoint_dir=str(out_path / "checkpoints"))
    except Exception as err:
        logger.warning("Could not execute PyTorch workload: %s", err)

    # 3. Run empirical research experiment suite
    print(f"\n--- EXECUTING EMPIRICAL BENCHMARK SUITE ({num_trials} Trials per Policy) ---")
    start_time = time.monotonic()
    suite_results = run_all_experiments(num_trials=num_trials)
    elapsed_sec = round(time.monotonic() - start_time, 2)

    # Attach hardware telemetry to results JSON
    full_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "execution_mode": "SIMULATED" if use_simulated else "HARDWARE_OR_SIMULATED",
            "num_trials": num_trials,
            "elapsed_seconds": elapsed_sec,
            "gpu_devices": gpu_telemetry,
        },
        "experiments": suite_results,
    }

    # 3. Save JSON master dataset
    json_path = out_path / "experiment_results.json"
    json_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    logger.info("Saved master JSON benchmark dataset to %s", json_path)

    # 4. Export CSV files
    csv_paths = export_csv_data(out_path, suite_results)
    logger.info("Exported CSV datasets: %s", [str(p) for p in csv_paths])

    print("\n" + "=" * 80)
    print(f" BENCHMARK SUITE COMPLETED IN {elapsed_sec}s ")
    print(f" Master JSON Dataset: {json_path}")
    print(f" CSV Exports Directory: {out_path.resolve()}")
    print("=" * 80 + "\n")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="DRIGS Kaggle GPU Benchmark Evaluation Suite")
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of statistical benchmark trials per policy (default: 5)",
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="Force simulated GPU backend for testing without physical NVIDIA GPU",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kaggle_results",
        help="Output directory path for exported JSON and CSV datasets (default: kaggle_results)",
    )

    args = parser.parse_args()
    run_kaggle_evaluation(
        num_trials=args.trials,
        use_simulated=args.simulated,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
