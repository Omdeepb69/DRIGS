"""Unit tests for TopologyAwareScheduler in drigs.scheduler.topology_aware."""

import pytest

from drigs.core.interfaces import Scheduler
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
    WorkerInfo,
)
from drigs.scheduler.topology_aware import TopologyAwareScheduler


def _make_gpu(dev_id: str, nvlink_group: str = "", numa_node: int = 0, pcie_bus_id: str = "") -> ComputeDevice:
    tags = {}
    if nvlink_group:
        tags["nvlink_group"] = nvlink_group
    return ComputeDevice(
        device_id=dev_id,
        device_type=DeviceType.GPU,
        model_name="RTX 4090",
        total_memory_bytes=24 * 1024**3,
        available_memory_bytes=24 * 1024**3,
        numa_node=numa_node,
        pcie_bus_id=pcie_bus_id or f"0000:0{dev_id[-1]}:00.0",
        topology_tags=tags,
    )


def test_topology_aware_scheduler_protocol():
    scheduler = TopologyAwareScheduler()
    assert isinstance(scheduler, Scheduler)


def test_topology_aware_scheduler_nvlink_preference():
    g0 = _make_gpu("gpu-0", nvlink_group="nv-0")
    g1 = _make_gpu("gpu-1", nvlink_group="nv-0")
    g2 = _make_gpu("gpu-2", numa_node=1)
    g3 = _make_gpu("gpu-3", numa_node=1)

    worker = WorkerInfo(
        worker_id="w-node",
        hostname="node-1",
        ip_address="127.0.0.1",
        devices=[g0, g1, g2, g3],
        total_cpus=16,
        total_memory_bytes=64 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    cluster = ClusterState(workers={"w-node": worker})

    spec = WorkloadSpec(
        name="topo-job",
        resources=ResourceRequirements(gpus=2, cpus=2),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )
    job = Job(name="job-1", spec=spec)

    scheduler = TopologyAwareScheduler()
    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    alloc = allocations[0]
    assert alloc.worker_id == "w-node"
    # Should pick NVLink pair (gpu-0, gpu-1) over NUMA pair (gpu-2, gpu-3)
    assert sorted(alloc.assigned_device_ids) == ["gpu-0", "gpu-1"]


def test_topology_aware_scheduler_worker_selection():
    # Worker 1: 2 GPUs connected via PCIe (score 50.0)
    w1_g0 = _make_gpu("w1-g0", numa_node=0)
    w1_g1 = _make_gpu("w1-g1", numa_node=0)
    worker1 = WorkerInfo(
        worker_id="worker-pcie",
        hostname="pcie-node",
        ip_address="127.0.0.1",
        devices=[w1_g0, w1_g1],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )

    # Worker 2: 2 GPUs connected via NVLink (score 90.0)
    w2_g0 = _make_gpu("w2-g0", nvlink_group="nv-group-1")
    w2_g1 = _make_gpu("w2-g1", nvlink_group="nv-group-1")
    worker2 = WorkerInfo(
        worker_id="worker-nvlink",
        hostname="nvlink-node",
        ip_address="127.0.0.2",
        devices=[w2_g0, w2_g1],
        total_cpus=8,
        total_memory_bytes=32 * 1024**3,
        status=DeviceState.HEALTHY,
    )

    cluster = ClusterState(workers={"worker-pcie": worker1, "worker-nvlink": worker2})

    spec = WorkloadSpec(
        name="multi-gpu-topo",
        resources=ResourceRequirements(gpus=2, cpus=2),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )
    job = Job(name="job-topo", spec=spec)

    scheduler = TopologyAwareScheduler()
    allocations = scheduler.schedule([job], cluster)

    assert len(allocations) == 1
    alloc = allocations[0]
    assert alloc.worker_id == "worker-nvlink"
    assert sorted(alloc.assigned_device_ids) == ["w2-g0", "w2-g1"]


def test_topology_aware_scheduler_insufficient_gpus():
    w_g0 = _make_gpu("g0")
    worker = WorkerInfo(
        worker_id="worker-1",
        hostname="node-1",
        ip_address="127.0.0.1",
        devices=[w_g0],
        total_cpus=4,
        total_memory_bytes=16 * 1024**3,
        status=DeviceState.HEALTHY,
    )
    cluster = ClusterState(workers={"worker-1": worker})

    # Needs 2 GPUs, but worker only has 1
    spec = WorkloadSpec(
        name="job-too-big",
        resources=ResourceRequirements(gpus=2, cpus=1),
        execution=WorkloadExecutionConfig(entrypoint=["echo", "test"]),
    )
    job = Job(name="job-big", spec=spec)

    scheduler = TopologyAwareScheduler()
    allocations = scheduler.schedule([job], cluster)
    assert len(allocations) == 0
