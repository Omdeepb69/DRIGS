"""Workload specification parser and YAML loader for DRIGS."""

import json
from pathlib import Path
import re
from typing import Any, Dict, Union
import yaml

from drigs.core.models import WorkloadSpec


class SpecParseError(Exception):
    """Exception raised when parsing a workload specification fails."""
    pass


def parse_memory_bytes(value: Union[str, int, float]) -> int:
    """Parse human-readable memory string (e.g. '16GB', '512MB', '1024') to byte integer."""
    if isinstance(value, (int, float)):
        return int(value)

    val_str = str(value).strip().upper()
    if not val_str:
        return 0

    match = re.match(r"^([0-9.]+)\s*([A-Z]*)$", val_str)
    if not match:
        raise SpecParseError(f"Invalid memory specification format: {value!r}")

    num_part, unit_part = match.groups()
    try:
        num = float(num_part)
    except ValueError:
        raise SpecParseError(f"Invalid numeric value in memory specification: {value!r}")

    units = {
        "": 1,
        "B": 1,
        "BYTES": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }

    if unit_part not in units:
        raise SpecParseError(f"Unknown memory unit {unit_part!r} in specification {value!r}")

    return int(num * units[unit_part])


def _normalize_spec_dict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw YAML/JSON dictionary format before Pydantic validation."""
    if not isinstance(raw_data, dict):
        raise SpecParseError("Workload specification root must be a dictionary")

    data = dict(raw_data)

    if "name" not in data or not data["name"]:
        raise SpecParseError("Workload specification missing required 'name' field")

    # Normalize resources section if present
    res = data.get("resources", {})
    if isinstance(res, dict):
        res_copy = dict(res)
        if "gpu_memory" in res_copy and "gpu_memory_bytes" not in res_copy:
            res_copy["gpu_memory_bytes"] = parse_memory_bytes(res_copy.pop("gpu_memory"))
        elif "gpu_memory_bytes" in res_copy:
            res_copy["gpu_memory_bytes"] = parse_memory_bytes(res_copy["gpu_memory_bytes"])

        if "memory" in res_copy and "memory_bytes" not in res_copy:
            res_copy["memory_bytes"] = parse_memory_bytes(res_copy.pop("memory"))
        elif "memory_bytes" in res_copy:
            res_copy["memory_bytes"] = parse_memory_bytes(res_copy["memory_bytes"])

        data["resources"] = res_copy

    # Normalize runtime section if present
    rt = data.get("runtime", {})
    if isinstance(rt, dict):
        rt_copy = dict(rt)
        if "container" in rt_copy:
            val = rt_copy.pop("container")
            if isinstance(val, str):
                rt_copy["container_image"] = val
            elif isinstance(val, dict) and "image" in val:
                rt_copy["container_image"] = val["image"]
        data["runtime"] = rt_copy

    # Normalize checkpoint section if present
    cp = data.get("checkpoint", {})
    if isinstance(cp, dict):
        cp_copy = dict(cp)
        if "interval" in cp_copy and "interval_seconds" not in cp_copy:
            cp_copy["interval_seconds"] = cp_copy.pop("interval")
        data["checkpoint"] = cp_copy

    return data


def parse_workload_spec(content: Union[str, Dict[str, Any]]) -> WorkloadSpec:
    """Parse YAML string, JSON string, or dictionary into a WorkloadSpec instance."""
    if isinstance(content, dict):
        raw_dict = content
    elif isinstance(content, str):
        content_stripped = content.strip()
        if not content_stripped:
            raise SpecParseError("Cannot parse empty workload specification string")
        try:
            raw_dict = yaml.safe_load(content_stripped)
        except Exception as err:
            raise SpecParseError(f"YAML parsing error: {err}") from err
    else:
        raise SpecParseError(f"Unsupported content type for spec parsing: {type(content)}")

    if not isinstance(raw_dict, dict):
        raise SpecParseError("Parsed spec root is not a dictionary")

    normalized = _normalize_spec_dict(raw_dict)
    try:
        return WorkloadSpec.model_validate(normalized)
    except Exception as err:
        raise SpecParseError(f"Validation error for WorkloadSpec: {err}") from err


def load_workload_spec(filepath: Union[str, Path]) -> WorkloadSpec:
    """Load workload specification from a YAML or JSON file."""
    path = Path(filepath)
    if not path.is_file():
        raise SpecParseError(f"Workload spec file not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as err:
        raise SpecParseError(f"Failed to read file {path}: {err}") from err

    return parse_workload_spec(content)


def dump_workload_spec(spec: WorkloadSpec) -> str:
    """Serialize WorkloadSpec to YAML string format."""
    data = spec.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False)
