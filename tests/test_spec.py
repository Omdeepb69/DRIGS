"""Unit tests for drigs.core.spec parser and loader."""

import pytest
import tempfile
from pathlib import Path

from drigs.core.spec import (
    SpecParseError,
    parse_memory_bytes,
    parse_workload_spec,
    load_workload_spec,
    dump_workload_spec,
)
from drigs.core.models import WorkloadSpec, ExecutionBackendType


def test_parse_memory_bytes_valid():
    assert parse_memory_bytes(1024) == 1024
    assert parse_memory_bytes(0) == 0
    assert parse_memory_bytes("512B") == 512
    assert parse_memory_bytes("1KB") == 1024
    assert parse_memory_bytes("16MB") == 16 * 1024 * 1024
    assert parse_memory_bytes("16 GB") == 16 * 1024 * 1024 * 1024
    assert parse_memory_bytes("2TB") == 2 * 1024 * 1024 * 1024 * 1024


def test_parse_memory_bytes_invalid():
    with pytest.raises(SpecParseError, match="Invalid memory specification format"):
        parse_memory_bytes("invalid_memory")

    with pytest.raises(SpecParseError, match="Unknown memory unit"):
        parse_memory_bytes("100PB")


def test_parse_workload_spec_from_dict():
    raw_dict = {
        "name": "test-job",
        "resources": {
            "gpus": 2,
            "gpu_memory": "12GB",
            "cpus": 8,
            "memory": "32GB",
        },
        "execution": {
            "backend": "NATIVE",
            "entrypoint": "python train.py",
            "args": ["--batch-size", "32"],
        },
        "distribution": {
            "mode": "distributed",
            "world_size": 2,
        },
        "checkpoint": {
            "enabled": True,
            "interval": 600,
        },
    }

    spec = parse_workload_spec(raw_dict)
    assert isinstance(spec, WorkloadSpec)
    assert spec.name == "test-job"
    assert spec.resources.gpus == 2
    assert spec.resources.gpu_memory_bytes == 12 * 1024**3
    assert spec.resources.cpus == 8
    assert spec.resources.memory_bytes == 32 * 1024**3
    assert spec.execution.backend == ExecutionBackendType.NATIVE
    assert spec.execution.entrypoint == "python train.py"
    assert spec.execution.args == ["--batch-size", "32"]
    assert spec.distribution.world_size == 2
    assert spec.checkpoint.enabled is True
    assert spec.checkpoint.interval_seconds == 600


def test_parse_workload_spec_from_yaml_string():
    yaml_str = """
name: resnet-training
resources:
  gpus: 4
  gpu_memory: 16GB
  cpus: 16
  memory: 64GB
execution:
  backend: NATIVE
  entrypoint:
    - python
    - main.py
runtime:
  container:
    image: pytorch/pytorch:latest
checkpoint:
  enabled: true
  interval: 300
"""

    spec = parse_workload_spec(yaml_str)
    assert spec.name == "resnet-training"
    assert spec.resources.gpus == 4
    assert spec.resources.gpu_memory_bytes == 16 * 1024**3
    assert spec.execution.entrypoint == ["python", "main.py"]
    assert spec.runtime.container_image == "pytorch/pytorch:latest"
    assert spec.checkpoint.enabled is True
    assert spec.checkpoint.interval_seconds == 300


def test_parse_workload_spec_invalid_inputs():
    with pytest.raises(SpecParseError, match="Cannot parse empty"):
        parse_workload_spec("   ")

    with pytest.raises(SpecParseError, match="missing required 'name'"):
        parse_workload_spec({"execution": {"entrypoint": "ls"}})

    with pytest.raises(SpecParseError, match="Validation error"):
        parse_workload_spec({
            "name": "invalid-spec",
            "execution": {"entrypoint": "ls"},
            "resources": {"gpus": -5},
        })


def test_load_and_dump_workload_spec_file():
    yaml_str = """
name: load-file-test
resources:
  gpus: 1
  gpu_memory_bytes: 8589934592
execution:
  backend: NATIVE
  entrypoint: echo 'hello world'
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "test_spec.yaml"
        spec_path.write_text(yaml_str, encoding="utf-8")

        spec = load_workload_spec(spec_path)
        assert spec.name == "load-file-test"
        assert spec.resources.gpus == 1

        dumped_yaml = dump_workload_spec(spec)
        assert "load-file-test" in dumped_yaml

        spec_reparsed = parse_workload_spec(dumped_yaml)
        assert spec_reparsed.name == spec.name


def test_load_workload_spec_file_not_found():
    with pytest.raises(SpecParseError, match="not found"):
        load_workload_spec("/path/to/nonexistent/spec.yaml")
