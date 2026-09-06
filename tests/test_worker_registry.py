"""Unit tests for WorkerRegistry in drigs.workers.registry."""

from datetime import datetime, timedelta, timezone
import time
import pytest

from drigs.core.interfaces import WorkerRegistryProtocol
from drigs.core.models import (
    ComputeDevice,
    DeviceState,
    DeviceType,
    WorkerInfo,
)
from drigs.workers.registry import WorkerRegistry


def _make_dummy_worker(worker_id: str = "w-1", hostname: str = "node-1") -> WorkerInfo:
    device = ComputeDevice(
        device_id="gpu-0",
        device_type=DeviceType.GPU,
        model_name="RTX 4090",
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=24 * 1024**3,
    )
    return WorkerInfo(
        worker_id=worker_id,
        hostname=hostname,
        ip_address="127.0.0.1",
        devices=[device],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )


def test_worker_registry_protocol_conformance():
    registry = WorkerRegistry()
    assert isinstance(registry, WorkerRegistryProtocol)


def test_worker_registry_register_and_get():
    registry = WorkerRegistry()
    w1 = _make_dummy_worker("w-1", "host-1")
    w2 = _make_dummy_worker("w-2", "host-2")

    assert registry.register(w1) is True
    assert registry.register(w2) is True

    fetched_1 = registry.get_worker("w-1")
    assert fetched_1 is not None
    assert fetched_1.worker_id == "w-1"
    assert fetched_1.hostname == "host-1"

    active = registry.get_active_workers()
    assert len(active) == 2


def test_worker_registry_heartbeat_updates():
    registry = WorkerRegistry()
    w1 = _make_dummy_worker("w-1", "host-1")
    registry.register(w1)

    assert registry.heartbeat("w-unregistered") is False

    time.sleep(0.05)
    assert registry.heartbeat("w-1", status=DeviceState.DEGRADED) is True

    updated = registry.get_worker("w-1")
    assert updated is not None
    assert updated.status == DeviceState.DEGRADED

    active = registry.get_active_workers()
    assert len(active) == 1
    assert active[0].status == DeviceState.DEGRADED


def test_worker_registry_timeout_detection():
    registry = WorkerRegistry(timeout_seconds=0.1)
    w1 = _make_dummy_worker("w-timeout-1")
    registry.register(w1)

    assert len(registry.get_active_workers()) == 1

    # Wait past timeout
    time.sleep(0.15)

    timed_out = registry.check_timeouts()
    assert "w-timeout-1" in timed_out

    # Active workers should now exclude timed-out node
    assert len(registry.get_active_workers()) == 0

    # All workers should still contain it, but with OFFLINE status
    all_workers = registry.get_all_workers()
    assert len(all_workers) == 1
    assert all_workers[0].status == DeviceState.OFFLINE


def test_worker_registry_deregister():
    registry = WorkerRegistry()
    w1 = _make_dummy_worker("w-dereg")
    registry.register(w1)

    assert registry.get_worker("w-dereg") is not None
    assert registry.deregister("w-dereg") is True
    assert registry.get_worker("w-dereg") is None
    assert registry.deregister("w-dereg") is False
