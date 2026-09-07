"""Unit and Integration Tests for End-to-End Hard Fault Recovery Benchmark."""

import pytest
from experiments.e2e_recovery_test import run_e2e_fault_recovery_benchmark


def test_run_e2e_fault_recovery_benchmark(tmp_path):
    """Verify SIGKILL fault injection and wall-clock recovery breakdown benchmark."""
    out_json = tmp_path / "e2e_results.json"
    res = run_e2e_fault_recovery_benchmark(output_path=str(out_json))

    assert res["injected_signal"] == "SIGKILL (kill -9)"
    assert res["rescheduled_successfully"] is True
    assert "total_e2e_recovery_latency_ms" in res["metrics_ms"]
    assert res["metrics_ms"]["total_e2e_recovery_latency_ms"] > 0.0
    assert out_json.exists()
