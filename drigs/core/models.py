"""Core domain models for DRIGS using Pydantic v2."""

from datetime import datetime, timezone
from enum import Enum
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Lifecycle status of a workload job."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RECOVERING = "RECOVERING"


class DeviceType(str, Enum):
    """Hardware compute device types."""

    CPU = "CPU"
    GPU = "GPU"
    ACCELERATOR = "ACCELERATOR"


class DeviceState(str, Enum):
    """Health and operational state of a device or worker node."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    OFFLINE = "OFFLINE"


class ResourceState(str, Enum):
    """State of an allocated or tracked resource."""

    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    ALLOCATED = "ALLOCATED"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


class SchedulingPolicy(str, Enum):
    """Supported scheduling strategies."""

    FIFO = "FIFO"
    PRIORITY = "PRIORITY"
    FIRST_FIT = "FIRST_FIT"
    BEST_FIT = "BEST_FIT"
    WORST_FIT = "WORST_FIT"
    BIN_PACK = "BIN_PACK"
    MEMORY_AWARE = "MEMORY_AWARE"
    TOPOLOGY_AWARE = "TOPOLOGY_AWARE"


class ExecutionBackendType(str, Enum):
    """Workload execution engine types."""

    NATIVE = "NATIVE"
    DOCKER = "DOCKER"
    PYTORCH = "PYTORCH"
    DISTRIBUTED = "DISTRIBUTED"


class ComputeDevice(BaseModel):
    """Hardware compute device representation (GPU, CPU, etc.)."""

    model_config = ConfigDict(frozen=True)

    device_id: str
    device_type: DeviceType
    vendor: str = "NVIDIA"
    model_name: str
    total_memory_bytes: int = Field(ge=0)
    available_memory_bytes: int = Field(ge=0)
    utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None
    compute_capability: Optional[str] = None
    pcie_bus_id: Optional[str] = None
    numa_node: Optional[int] = None
    topology_tags: Dict[str, str] = Field(default_factory=dict)


class DeviceStatus(BaseModel):
    """Real-time status snippet for a compute device."""

    device_id: str
    state: DeviceState = DeviceState.HEALTHY
    utilization_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_used_bytes: int = Field(default=0, ge=0)
    memory_free_bytes: int = Field(default=0, ge=0)
    temperature_celsius: Optional[float] = None
    updated_at: datetime = Field(default_factory=_now_utc)


class ResourceRequirements(BaseModel):
    """Resource requirements requested by a workload."""

    model_config = ConfigDict(frozen=True)

    gpus: int = Field(default=0, ge=0)
    gpu_memory_bytes: int = Field(default=0, ge=0)
    cpus: int = Field(default=1, ge=1)
    memory_bytes: int = Field(default=0, ge=0)
    topology_aware: bool = False
    min_compute_capability: Optional[str] = None


class ResourceAllocation(BaseModel):
    """Active allocation of physical devices to a specific job."""

    model_config = ConfigDict(frozen=True)

    allocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    worker_id: str
    assigned_device_ids: List[str] = Field(default_factory=list)
    assigned_cpu_cores: List[int] = Field(default_factory=list)
    memory_bytes: int = Field(default=0, ge=0)
    allocated_at: datetime = Field(default_factory=_now_utc)


class WorkloadExecutionConfig(BaseModel):
    """Execution backend & entrypoint configuration."""

    model_config = ConfigDict(frozen=True)

    backend: ExecutionBackendType = ExecutionBackendType.NATIVE
    entrypoint: Union[str, List[str]]
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    working_dir: Optional[str] = None


class WorkloadDistributionConfig(BaseModel):
    """Distributed execution settings (PyTorch DDP / NCCL)."""

    model_config = ConfigDict(frozen=True)

    mode: str = "single"
    world_size: int = Field(default=1, ge=1)
    master_addr: Optional[str] = None
    master_port: Optional[int] = None


class WorkloadRuntimeConfig(BaseModel):
    """Runtime environment & container configuration."""

    model_config = ConfigDict(frozen=True)

    container_image: Optional[str] = None
    volumes: Dict[str, str] = Field(default_factory=dict)


class WorkloadCheckpointConfig(BaseModel):
    """Workload fault tolerance and checkpoint configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    interval_seconds: int = Field(default=300, ge=1)
    checkpoint_dir: Optional[str] = None


class WorkloadSpec(BaseModel):
    """Complete declarative specification of an AI workload."""

    model_config = ConfigDict(frozen=True)

    name: str
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)
    execution: WorkloadExecutionConfig
    distribution: WorkloadDistributionConfig = Field(default_factory=WorkloadDistributionConfig)
    runtime: WorkloadRuntimeConfig = Field(default_factory=WorkloadRuntimeConfig)
    checkpoint: WorkloadCheckpointConfig = Field(default_factory=WorkloadCheckpointConfig)


class Job(BaseModel):
    """Managed job instance within the DRIGS control plane."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    status: JobStatus = JobStatus.PENDING
    priority: int = Field(default=0, ge=0)
    spec: WorkloadSpec
    submitted_at: datetime = Field(default_factory=_now_utc)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    allocation: Optional[ResourceAllocation] = None
    error_message: Optional[str] = None

    @property
    def id(self) -> str:
        return self.job_id


class WorkerInfo(BaseModel):
    """Information and capacity reported by a worker node."""

    worker_id: str
    hostname: str
    ip_address: str
    devices: List[ComputeDevice] = Field(default_factory=list)
    total_cpus: int = Field(default=1, ge=1)
    total_memory_bytes: int = Field(default=0, ge=0)
    status: DeviceState = DeviceState.HEALTHY
    last_heartbeat: datetime = Field(default_factory=_now_utc)


class ClusterState(BaseModel):
    """Cluster-wide aggregated state snapshot."""

    workers: Dict[str, WorkerInfo] = Field(default_factory=dict)
    active_jobs: Dict[str, Job] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=_now_utc)
