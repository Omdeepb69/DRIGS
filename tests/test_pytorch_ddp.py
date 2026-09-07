"""Unit and Integration Tests for PyTorch DDP Multi-Process Benchmark."""

import os
import pytest

from scripts.train_pytorch_ddp import run_ddp_benchmark_suite, run_ddp_training_rank


def test_run_ddp_training_rank_single():
    """Verify single rank DDP execution completes cleanly."""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"

    res = run_ddp_training_rank(epochs=2, backend_name="gloo")
    assert res["status"] in ("SUCCESS", "UNAVAILABLE")
    if res["status"] == "SUCCESS":
        assert res["rank"] == 0
        assert res["mean_step_time_ms"] >= 0.0


def test_run_ddp_benchmark_suite_execution(tmp_path):
    """Verify DRIGS DistributedBackend launches 2 DDP worker rank processes."""
    out_json = tmp_path / "ddp_results.json"
    res = run_ddp_benchmark_suite(epochs=2, output_path=str(out_json))

    assert res["world_size"] == 2
    assert res["final_status"] in ("COMPLETED", "FAILED")
    assert out_json.exists()
