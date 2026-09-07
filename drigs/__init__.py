"""DRIGS: Distributed Resource & Intelligent GPU Scheduling."""

__version__ = "0.1.0"

from drigs.core.controller import LocalController
from drigs.core.interfaces import (
    ExecutionBackend,
    HardwareBackend,
    Scheduler,
    WorkerRegistryProtocol,
)
from drigs.core.models import (
    ClusterState,
    ComputeDevice,
    DeviceState,
    DeviceType,
    Job,
    JobStatus,
    ResourceAllocation,
    ResourceRequirements,
    WorkloadSpec,
)
from drigs.core.queue import AdmissionController, JobQueue
from drigs.core.resource_manager import ResourceManager
from drigs.execution.distributed import DistributedBackend
from drigs.execution.docker import DockerBackend
from drigs.execution.native import NativeProcessBackend
from drigs.scheduler import (
    BestFitScheduler,
    BinPackScheduler,
    FIFOScheduler,
    FirstFitScheduler,
    GangScheduler,
    MemoryAwareScheduler,
    PriorityScheduler,
    TopologyAwareScheduler,
)
from drigs.workers.agent import WorkerAgent

__all__ = [
    "__version__",
    "LocalController",
    "JobQueue",
    "AdmissionController",
    "ResourceManager",
    "Job",
    "JobStatus",
    "WorkloadSpec",
    "ResourceRequirements",
    "ResourceAllocation",
    "ClusterState",
    "ComputeDevice",
    "DeviceState",
    "DeviceType",
    "HardwareBackend",
    "Scheduler",
    "ExecutionBackend",
    "WorkerRegistryProtocol",
    "FIFOScheduler",
    "FirstFitScheduler",
    "BestFitScheduler",
    "MemoryAwareScheduler",
    "PriorityScheduler",
    "BinPackScheduler",
    "GangScheduler",
    "TopologyAwareScheduler",
    "NativeProcessBackend",
    "DockerBackend",
    "DistributedBackend",
    "WorkerAgent",
]
