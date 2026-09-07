"""Unit and Integration Tests for High VRAM Saturation and OOM Recovery Benchmark."""

import pytest
from scripts.vram_saturation_test import build_saturated_cluster_state, run_vram_saturation_benchmark


def test_build_saturated_cluster_state():
    """Verify saturated cluster state construction."""
    cs = build_saturated_cluster_state(saturation_pct=95.0, total_vram_gb=16.0)
    assert len(cs.workers) == 2
    w0 = cs.workers["worker-0"]
    assert len(w0.devices) == 2
    assert w0.devices[0].utilization_pct == 95.0
    assert w0.devices[0].available_memory_bytes < w0.devices[0].total_memory_bytes


def test_run_vram_saturation_benchmark(tmp_path):
    """Verify high VRAM saturation placement and OOM re-queuing benchmark."""
    out_json = tmp_path / "vram_results.json"
    res = run_vram_saturation_benchmark(saturation_levels=[85.0, 95.0], output_path=str(out_json))

    assert "85.0%" in res["results_by_saturation"]
    assert "95.0%" in res["results_by_saturation"]
    assert res["oom_recovery"]["oom_requeued_successfully"] is True
    assert out_json.exists()
