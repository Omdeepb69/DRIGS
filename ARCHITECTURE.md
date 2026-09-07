# Architecture

Stable architectural decisions for **DRIGS (Distributed Resource & Intelligent GPU Scheduling)**.
This file records module boundaries, core interfaces, structural invariants, and design decisions.

---

## System Architecture Overview

```text
                               ┌───────────────────────┐
                               │       DRIGS CLI       │
                               │                       │
                               │ submit / status / ... │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │      DRIGS API        │
                               │       Server          │
                               └───────────┬───────────┘
                                           │
                                           ▼
                ┌─────────────────────────────────────────────────────┐
                │                   CONTROL PLANE                     │
                │                                                     │
                │  Job Queue & Admission Controller                   │
                │  Pluggable Scheduler (FIFO/Memory/Topology/Gang)    │
                │  Resource Manager                                   │
                │  Worker Registry & Heartbeat Monitor                │
                │  Recovery Manager & Failure Detector                │
                └──────────────────────────┬──────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
       ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
       │ DRIGS       │              │ DRIGS       │              │ DRIGS       │
       │ Worker 0    │              │ Worker 1    │              │ Worker 2    │
       │             │              │             │              │             │
       │ Executor    │              │ Executor    │              │ Executor    │
       │ GPU Monitor │              │ GPU Monitor │              │ GPU Monitor │
       │ Heartbeat   │              │ Heartbeat   │              │ Heartbeat   │
       └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
              │                            │                            │
              └────────────────────────────┼────────────────────────────┘
                                           ▼
                               Distributed Execution
                               PyTorch / NCCL / etc.
```

---

## Module Map

| Module Directory | Responsibility & Ownership | Key Dependencies |
| :--- | :--- | :--- |
| `drigs/core/` | Domain logic, generic models (`Job`, `ComputeDevice`, `ResourceAllocation`, `ClusterState`), and abstract protocols (`HardwareBackend`, `Scheduler`, `ExecutionBackend`). | Pure Python 3.10+, Pydantic. **Zero vendor dependencies.** |
| `drigs/hardware/` | Hardware discovery & status collectors (`CUDABackend`, `CPUBackend`, `SimulatedBackend`). Interacts with NVML, PyTorch CUDA APIs, and system topology files. | `pynvml`, `torch.cuda` (lazy/optional), `psutil`. |
| `drigs/scheduler/` | Pluggable scheduling policies (`FIFOScheduler`, `FirstFitScheduler`, `BestFitScheduler`, `MemoryAwareScheduler`, `GangScheduler`, `TopologyAwareScheduler`, `BinPackScheduler`). | Depends only on `drigs/core/`. |
| `drigs/workers/` | Worker node agent, local monitor, registration client, and heartbeat sender (`WorkerAgent`, `WorkerRegistry`, `HeartbeatManager`). | `drigs/core/`, `drigs/hardware/`, `asyncio`, `httpx`/`grpc`. |
| `drigs/execution/` | Workload execution drivers (`NativeProcessBackend`, `DockerBackend`, `DistributedBackend`). Handles process spawning, `CUDA_VISIBLE_DEVICES` isolation, and environment variables. | `subprocess`, `asyncio`, `docker` (optional). |
| `drigs/distributed/` | Distributed environment rendezvous, master address assignment, process group orchestration (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`). | `drigs/core/`, `drigs/execution/`. |
| `drigs/recovery/` | Fault detection, health timeouts, checkpoint discovery, and automatic job rescheduling (`FailureDetector`, `CheckpointManager`, `Rescheduler`). | `drigs/core/`. |
| `drigs/monitoring/` | Prometheus exporter and runtime metrics collector (`MetricsCollector`, `PrometheusExporter`). | `prometheus_client` (optional). |
| `drigs/diagnostics/` | Workload bottleneck analysis and diagnostic reporting (`DiagnosticAnalyzer`). | `drigs/core/`, `drigs/monitoring/`. |
| `drigs/api/` | REST API control plane server (`FastAPI`, endpoints for jobs, workers, gpus, metrics). | `fastapi`, `uvicorn`. |
| `drigs/cli/` | Command-line interface for human and script interaction (`typer` or `argparse`). | `typer` / `rich`. |
| `experiments/` | Reproducible benchmark suite and scheduling policy evaluation framework. | `drigs/*`, `matplotlib` / `pandas` (for reports). |

---

## Interfaces / Protocols

### 1. `HardwareBackend` Protocol (`drigs/core/interfaces.py`)
- `discover_devices() -> list[ComputeDevice]`
- `get_device_status(device_id: str) -> ComputeDeviceStatus`
- `get_topology() -> TopologyGraph`

*Consumed by:* Worker agent, Resource manager.
*Implemented by:* `drigs/hardware/cuda.py`, `drigs/hardware/cpu.py`, `drigs/hardware/simulated.py`.

---

### 2. `Scheduler` Protocol (`drigs/core/interfaces.py`)
- `schedule(pending_jobs: list[Job], cluster_state: ClusterState) -> list[ResourceAllocation]`

*Consumed by:* Controller scheduling loop.
*Implemented by:* `drigs/scheduler/fifo.py`, `memory_aware.py`, `topology_aware.py`, `gang.py`, etc.

---

### 3. `ExecutionBackend` Protocol (`drigs/core/interfaces.py`)
- `launch(job: Job, allocation: ResourceAllocation) -> ExecutionHandle`
- `stop(handle: ExecutionHandle) -> bool`
- `get_status(handle: ExecutionHandle) -> JobStatus`
- `stream_logs(handle: ExecutionHandle) -> AsyncIterator[str]`

*Consumed by:* Worker agent / Executor.
*Implemented by:* `drigs/execution/native.py`, `docker.py`, `distributed.py`.

---

### 4. `WorkerRegistryProtocol` Protocol (`drigs/core/interfaces.py`)
- `register(worker: WorkerInfo) -> bool`
- `heartbeat(worker_id: str, status: WorkerStatus) -> bool`
- `get_active_workers() -> list[WorkerInfo]`

*Consumed by:* Resource Manager, Scheduler, Recovery Manager.
*Implemented by:* `drigs/workers/registry.py`.

---

## Boundaries & Invariants

1. **Core Independence**: `drigs/core/` must never import hardware-specific or vendor-specific libraries (`torch`, `pynvml`, `docker`, `fastapi`, `prometheus_client`). All external concepts enter via abstractions.
2. **Policy vs Resource Distinction**: Schedulers decide *which job runs on which resource allocation*. Resource Manager tracks *what hardware exists and its current state*. Neither directly launches processes.
3. **Execution Separation**: Schedulers and Resource Managers do not invoke `subprocess` or `docker` commands directly; all execution flows through `ExecutionBackend` implementations.
4. **Native-First Operating Mode**: Native process execution is the primary default, allowing DRIGS to run cleanly in restricted environments (like Kaggle or standard Linux VMs) without requiring Docker or root privileges.
5. **Config as Data**: All component configurations live in Pydantic dataclasses/models, loaded from explicit YAML files. No global mutable state or hardcoded magic values.

---

## Decisions Log

### 2026-09-07 — Architecture Foundation & Layered Abstractions
**Reasoning**: System requires pluggability across scheduling policies (FIFO, Memory-Aware, Topology-Aware, Gang) and execution environments (Native, Docker, Multi-GPU Distributed). Abstract protocols in `drigs/core/` cleanly isolate core domain logic from execution engines and hardware APIs.
**Alternatives considered**: Direct coupling to CUDA/PyTorch in scheduler — rejected because it breaks CPU-only testing, hardware abstraction, and pluggable research benchmarking.

### 2026-09-07 — Ephemeral Remote Worker Outbound Registration Architecture
**Reasoning**: Remote GPU nodes operating behind NAT or in ephemeral notebook environments (Kaggle, Google Colab) cannot accept inbound connections. DRIGS worker agents use an outbound phone-home HTTP client architecture to register with the control plane (`POST /v1/workers/register`), stream device telemetry, maintain periodic heartbeats, and pull/execute assigned workload payloads.
**Alternatives considered**: Inbound SSH/gRPC to workers — rejected because ephemeral nodes lack public IPs/inbound port forwarding.

### 2026-09-07 — Kaggle Real-GPU Evaluation Suite & Publication Data Pipeline Architecture
**Reasoning**: To enable reproducible research paper submission from ephemeral cloud environments (Kaggle T4/P100 GPUs), DRIGS provides self-contained notebook evaluation runners (`scripts/run_kaggle_experiments.py`) that clone repository state, verify real `CUDABackend` device discovery, execute empirical benchmarks, and generate structured JSON/CSV metrics alongside publication LaTeX tables and figures (`experiments/export_paper_data.py`).
**Alternatives considered**: Manual notebook script execution — rejected because automated export guarantees exact statistical reproducibility.

### 2026-09-07 — Production Hardening & Reliability Architecture (Phase 13)
**Reasoning**: To resolve control plane single points of failure (SPOF), security risks, and orphan process leaks identified during critique:
1. **Control Plane State Persistence**: `StorageBackend` protocol with standard library `SQLiteStore` default engine storing job queue events, worker states, and resource allocations in ACID transactions.
2. **API Security Model**: Bearer token / API Key header authentication middleware (`drigs/api/auth.py`) securing REST endpoints and worker outbound registration.
3. **Process Lifecycle Guard**: Recursive process tree teardown via `psutil` + NVML orphan process PID sweeper preventing zombie CUDA memory allocations.
4. **Dynamic NVML Topology Discovery**: Native NVML common ancestor queries mapped directly into `TopologyGraph`.
**Alternatives considered**: External heavy database (PostgreSQL/etcd) — rejected because stdlib `sqlite3` preserves zero external daemon requirements for single-controller deployments.

### 2026-09-07 — PyPI Package & Open-Source Library Distribution Architecture (Phase 20)
**Reasoning**: To transition DRIGS from a research repository to a universally accessible open-source Python library and CLI tool, DRIGS is configured for distribution via PyPI (`pip install drigs`):
1. **Clean Public API Surface**: Top-level `drigs` package (`drigs/__init__.py`) exports primary domain models and control plane abstractions (`LocalController`, `FIFOScheduler`, `GangScheduler`, `TopologyAwareScheduler`, `JobSpec`, `ResourceRequirements`).
2. **Unified Console CLI Entrypoint**: Standard `project.scripts` configuration in `pyproject.toml` exposing `drigs = "drigs.cli.main:app"` for direct command-line execution (`drigs submit`, `drigs status`, `drigs worker`).
3. **Standard Distribution Build Pipeline**: Packaging via `setuptools` build backend producing standard source distributions (`.tar.gz`) and binary wheels (`.whl`).
**Alternatives considered**: Requiring manual `git clone` or container deployment — rejected because standard PyPI packages enable instant adoption in research scripts, Jupyter notebooks, and production AI pipelines.


