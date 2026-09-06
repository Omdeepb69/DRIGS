"""Unit tests for DRIGS Typer CLI in drigs.cli.main."""

import tempfile
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from drigs.cli.main import app

runner = CliRunner()


def test_cli_submit_command():
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmp:
        tmp.write("""
name: cli-job
resources:
  cpus: 2
  gpus: 1
execution:
  entrypoint: ["python3", "main.py"]
""")
        tmp.flush()

        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {"job_id": "job-cli-101", "status": "QUEUED"}

        with patch("httpx.Client.post", return_value=mock_res):
            result = runner.invoke(app, ["submit", tmp.name, "--priority", "5"])
            assert result.exit_code == 0
            assert "Job Submitted Successfully!" in result.output
            assert "job-cli-101" in result.output


def test_cli_jobs_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [
        {
            "job_id": "j-1",
            "name": "job-1",
            "status": "RUNNING",
            "priority": 10,
            "allocation": {"worker_id": "node-1"},
            "submitted_at": "2026-09-07T00:00:00Z",
        }
    ]

    with patch("httpx.Client.get", return_value=mock_res):
        result = runner.invoke(app, ["jobs"])
        assert result.exit_code == 0
        assert "j-1" in result.output
        assert "job-1" in result.output
        assert "RUNNING" in result.output


def test_cli_workers_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [
        {
            "worker_id": "worker-1",
            "hostname": "node1",
            "ip_address": "10.0.0.1",
            "status": "HEALTHY",
            "total_cpus": 8,
            "devices": [{"device_type": "GPU"}],
        }
    ]

    with patch("httpx.Client.get", return_value=mock_res):
        result = runner.invoke(app, ["workers"])
        assert result.exit_code == 0
        assert "worker-1" in result.output
        assert "node1" in result.output


def test_cli_gpus_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = [
        {
            "device_id": "gpu-0",
            "model_name": "Tesla T4",
            "total_memory_bytes": 16 * 1024**3,
            "state": "HEALTHY",
        }
    ]

    with patch("httpx.Client.get", return_value=mock_res):
        result = runner.invoke(app, ["gpus"])
        assert result.exit_code == 0
        assert "gpu-0" in result.output
        assert "Tesla T4" in result.output


def test_cli_inspect_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "job_id": "j-inspect-1",
        "name": "job-inspect",
        "status": "COMPLETED",
    }

    with patch("httpx.Client.get", return_value=mock_res):
        result = runner.invoke(app, ["inspect", "j-inspect-1"])
        assert result.exit_code == 0
        assert "j-inspect-1" in result.output
        assert "COMPLETED" in result.output


def test_cli_cancel_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"job_id": "j-cancel-1", "status": "CANCELLED"}

    with patch("httpx.Client.post", return_value=mock_res):
        result = runner.invoke(app, ["cancel", "j-cancel-1"])
        assert result.exit_code == 0
        assert "cancelled successfully" in result.output


def test_cli_diagnose_command():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {
        "job_id": "j-diag-1",
        "status": "QUEUED",
        "spec": {"resources": {"cpus": 4, "gpus": 2, "gpu_memory_bytes": 8000}},
        "allocation": None,
    }

    with patch("httpx.Client.get", return_value=mock_res):
        result = runner.invoke(app, ["diagnose", "j-diag-1"])
        assert result.exit_code == 0
        assert "Diagnostic Report for Job:" in result.output
        assert "QUEUED" in result.output
