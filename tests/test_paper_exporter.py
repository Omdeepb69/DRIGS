"""Unit tests for DRIGS Publication LaTeX Exporter and Figure Generator."""

import json
from pathlib import Path
import tempfile
import pytest

from experiments.export_paper_data import (
    export_paper_artifacts,
    generate_latex_tables,
    generate_paper_figures,
    generate_paper_summary_markdown,
    load_benchmark_data,
)


@pytest.fixture
def mock_benchmark_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_file = tmp_path / "experiment_results.json"
        mock_data = {
            "experiments": {
                "experiment_a": {
                    "results": {
                        "TopologyAware": {
                            "throughput_mean_jobs_per_sec": 1.05,
                            "mean_wait_time_sec": 0.22,
                            "p95_wait_time_sec": 0.54,
                            "vram_fragmentation_pct": 12.5,
                            "placement_efficiency_pct": 100.0,
                        },
                        "FIFO": {
                            "throughput_mean_jobs_per_sec": 0.78,
                            "mean_wait_time_sec": 0.51,
                            "p95_wait_time_sec": 1.05,
                            "vram_fragmentation_pct": 22.1,
                            "placement_efficiency_pct": 100.0,
                        },
                    }
                },
                "experiment_b": {
                    "topology_score_topology_aware": 90.0,
                    "topology_score_random": 56.7,
                    "topology_score_improvement_pct": 58.7,
                },
                "experiment_e": {
                    "mean_rescheduling_latency_ms": 1.25,
                    "mean_job_completion_rate_pct": 100.0,
                    "total_restored_checkpoints": 3,
                },
            }
        }
        json_file.write_text(json.dumps(mock_data, indent=2), encoding="utf-8")
        yield tmp_path


def test_load_benchmark_data(mock_benchmark_dir):
    data = load_benchmark_data(mock_benchmark_dir)
    assert "experiment_a" in data
    assert "TopologyAware" in data["experiment_a"]["results"]
    assert data["experiment_b"]["topology_score_improvement_pct"] == 58.7
    assert data["experiment_e"]["mean_rescheduling_latency_ms"] == 1.25


def test_generate_latex_tables(mock_benchmark_dir):
    data = load_benchmark_data(mock_benchmark_dir)
    with tempfile.TemporaryDirectory() as out_dir:
        out_path = Path(out_dir) / "tables.tex"
        res_file = generate_latex_tables(data, out_path)
        assert res_file.exists()
        content = res_file.read_text(encoding="utf-8")

        assert "\\begin{table}" in content
        assert "\\caption" in content
        assert "TopologyAware" in content
        assert "58.7\\%" in content or "58.7" in content
        assert "Rescheduling Latency" in content


def test_generate_paper_figures(mock_benchmark_dir):
    data = load_benchmark_data(mock_benchmark_dir)
    with tempfile.TemporaryDirectory() as out_dir:
        figs = generate_paper_figures(data, Path(out_dir))
        # If matplotlib is installed, figures should be generated
        for fig_path in figs:
            assert fig_path.exists()


def test_generate_paper_summary_markdown(mock_benchmark_dir):
    data = load_benchmark_data(mock_benchmark_dir)
    with tempfile.TemporaryDirectory() as out_dir:
        doc_path = Path(out_dir) / "PAPER_EXPERIMENTS.md"
        res_file = generate_paper_summary_markdown(data, doc_path)
        assert res_file.exists()
        content = res_file.read_text(encoding="utf-8")
        assert "# DRIGS Research Paper Empirical Evaluation Summary" in content
        assert "TopologyAware" in content


def test_export_paper_artifacts(mock_benchmark_dir):
    with tempfile.TemporaryDirectory() as out_dir:
        out_path = Path(out_dir)
        artifacts = export_paper_artifacts(mock_benchmark_dir, out_path, out_path / "summary.md")
        assert artifacts["latex_tables"].exists()
        assert artifacts["paper_doc"].exists()
