"""Unit tests for Kaggle experiment runner and CSV export functionality."""

import json
from pathlib import Path
import pytest
from scripts.run_kaggle_experiments import (
    discover_gpu_telemetry,
    export_csv_data,
    run_kaggle_evaluation,
)


def test_discover_gpu_telemetry_simulated():
    """Test hardware telemetry discovery in simulated mode."""
    telemetry = discover_gpu_telemetry(use_simulated=True)
    assert isinstance(telemetry, list)
    assert len(telemetry) >= 1
    gpu = telemetry[0]
    assert "device_id" in gpu
    assert "model_name" in gpu
    assert "total_vram_gb" in gpu
    assert gpu["total_vram_gb"] > 0.0


def test_run_kaggle_evaluation(tmp_path: Path):
    """Test full Kaggle benchmark evaluation and dataset export using temporary output directory."""
    out_dir = str(tmp_path / "test_kaggle_results")

    results = run_kaggle_evaluation(
        num_trials=1,
        use_simulated=True,
        output_dir=out_dir,
    )

    assert isinstance(results, dict)
    assert "benchmark_metadata" in results
    assert "experiments" in results

    json_file = Path(out_dir) / "experiment_results.json"
    assert json_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "benchmark_metadata" in data

    # Verify CSV files created
    exp_a_csv = Path(out_dir) / "experiment_a_schedulers.csv"
    exp_b_csv = Path(out_dir) / "experiment_b_topology.csv"
    exp_e_csv = Path(out_dir) / "experiment_e_fault_recovery.csv"

    assert exp_a_csv.exists()
    assert exp_b_csv.exists()
    assert exp_e_csv.exists()

    # Read CSV lines
    a_lines = exp_a_csv.read_text(encoding="utf-8").splitlines()
    assert len(a_lines) >= 2  # Header + at least one row
    assert "Scheduler,NumTrials,Throughput_Jobs_Per_Sec" in a_lines[0]
