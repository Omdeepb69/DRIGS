# DRIGS: Distributed Resource & Intelligent GPU Scheduling

<p align="center">
  <a href="https://pypi.org/project/drigs/"><img src="https://img.shields.io/pypi/v/drigs.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/drigs/"><img src="https://img.shields.io/pypi/pyversions/drigs.svg" alt="Python Versions"></a>
  <a href="https://github.com/Omdeepb69/drigs/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/Omdeepb69/drigs"><img src="https://img.shields.io/github/stars/Omdeepb69/drigs.svg?style=social" alt="GitHub Stars"></a>
</p>

**DRIGS** is a lightweight, high-performance, GPU-aware distributed compute runtime and orchestration engine designed for scheduling, executing, monitoring, and recovering AI workloads across multi-GPU workstations, cloud notebook environments (e.g. Kaggle, Google Colab, RunPod), and heterogeneous multi-node clusters.

---

## Key Features

- ⚡ **Pluggable Scheduler Core**: Built-in scheduling algorithms including `GangScheduler` (atomic all-or-nothing allocation), `MemoryAwareScheduler`, `TopologyAwareScheduler` (NVLink/PCIe matrix optimization), `DRFScheduler` (Dominant Resource Fairness), `BinPackScheduler`, and `FIFOScheduler`.
- 🛡️ **NVML Orphan Sweeper & VRAM Guard**: Automatic NVML process tracking to detect and purge zombie CUDA allocations (`pynvml`), with real-time VRAM telemetry and automatic CUDA OOM interception.
- 🔌 **Native & Container Execution Drivers**: Native process isolation (`NativeProcessBackend`), Docker container orchestration (`DockerBackend`), and PyTorch DDP / `torchrun` multi-GPU rendezvous (`DistributedBackend`).
- 📡 **Ephemeral Worker Discovery**: Outbound phone-home HTTP/gRPC bootstrap agent for dynamic GPU nodes operating behind NAT firewalls or cloud Jupyter environments.
- 💾 **ACID Control Plane State**: Integrated WAL-mode SQLite state engine (`SQLiteStore`) ensuring zero-loss queue resilience and seamless controller restart recovery.
- 🚀 **REST API & CLI**: Full FastAPI REST server (`drigs server`) and intuitive command-line interface (`drigs status`, `drigs submit`).

---

## Architecture Overview

```text
                               ┌───────────────────────┐
                               │       DRIGS CLI       │
                               │  submit / status / .. │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │     FastAPI Server    │
                               └───────────┬───────────┘
                                           │
                                           ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                   CONTROL PLANE                     │
                 │                                                     │
                 │  Job Queue & Admission Controller                   │
                 │  Pluggable Scheduler (Gang / Topology / Memory)     │
                 │  Resource Manager & Cluster State                   │
                 │  Worker Registry & Heartbeat Monitor                │
                 │  Failure Detector & Rescheduler                     │
                 └──────────────────────────┬──────────────────────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
               ▼                            ▼                            ▼
        ┌─────────────┐              ┌─────────────┐              ┌─────────────┐
        │ DRIGS       │              │ DRIGS       │              │ DRIGS       │
        │ Worker 0    │              │ Worker 1    │              │ Worker 2    │
        │ (Executor)  │              │ (Executor)  │              │ (Executor)  │
        └──────┬──────┘              └──────┬──────┘              └──────┬──────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                                 Distributed GPUs / CUDA
```

---

## Installation

Install DRIGS from PyPI:

```bash
pip install drigs
```

For development or building from source:

```bash
git clone https://github.com/Omdeepb69/drigs.git
cd drigs
pip install -e .
```

---

## Quickstart

### 1. Python API Usage

```python
import drigs
from drigs.core.controller import LocalController
from drigs.core.models import ResourceRequirements, WorkloadExecutionConfig, WorkloadSpec
from drigs.scheduler.gang import GangScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler

# 1. Initialize DRIGS Local Controller with GangScheduler & MemoryAwareScheduler
gang_scheduler = GangScheduler(inner_scheduler=MemoryAwareScheduler())
controller = LocalController(scheduler=gang_scheduler)

# 2. Define a Multi-GPU Workload Spec
spec = WorkloadSpec(
    name="llama3-training-gang",
    resources=ResourceRequirements(
        gpus=2,
        gpu_memory_bytes=24 * 1024**3,
        cpus=4,
        memory_bytes=16 * 1024**3
    ),
    execution=WorkloadExecutionConfig(
        entrypoint="python3",
        args=["train_ddp.py", "--epochs", "10"]
    )
)

# 3. Submit & Schedule Job
job = controller.submit_job(spec, priority=10)
print(f"Submitted Job '{job.name}' (ID: {job.id}) to DRIGS Control Plane.")
```

### 2. Kaggle Dual T4 GPU LLM Hosting

Host large LLMs (e.g. `GLM-4`, `GLM-5.3-Flash`, `LLaMA-3`) across Dual Tesla T4 GPUs ($30\text{ GB}$ total VRAM) with 4-bit BitsAndBytes quantization, NVML zombie process sweeping, and full multi-turn chat history:

```python
from drigs.hardware.cuda import CUDABackend
from drigs.scheduler.gang import GangScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler

# DRIGS NVML process sweeper purges dead CUDA allocations before model loading
cuda_backend = CUDABackend()
cuda_backend.cleanup_orphan_processes()

# Initialize GangScheduler to reserve both T4 GPUs atomically
gang_scheduler = GangScheduler(inner_scheduler=MemoryAwareScheduler())
```

> See the full Kaggle host script at [`scripts/kaggle_glm_flash_host.py`](file:///mnt/data/projects/drigs/scripts/kaggle_glm_flash_host.py).

### 3. Command Line Interface (CLI)

```bash
# Discover GPU hardware and cluster status
drigs status

# Start DRIGS REST API Control Plane Server
drigs server --host 0.0.0.0 --port 8000

# Submit a workload specification file
drigs submit job_spec.yaml
```

---

## Scheduling Policies

| Scheduler Policy | Class | Description |
| :--- | :--- | :--- |
| **Gang Scheduler** | `GangScheduler` | Atomic all-or-nothing multi-GPU reservation. Ensures all $N$ GPUs are available simultaneously for DDP/FSDP workloads. |
| **Memory Aware** | `MemoryAwareScheduler` | Dynamic VRAM-aware scheduling that places workloads based on real-time free VRAM telemetry. |
| **Topology Aware** | `TopologyAwareScheduler` | NVLink and PCIe matrix interconnect optimization to minimize inter-GPU communication latency. |
| **Dominant Resource Fair** | `DRFScheduler` | Multi-resource fair sharing policy balancing CPU, GPU, and RAM allocations across competing users. |
| **Bin Packing** | `BinPackScheduler` | Consolidates workloads onto the fewest possible GPU nodes to maximize cluster density and idle shutdown. |
| **FIFO** | `FIFOScheduler` | First-In, First-Out queue scheduling for basic single-tenant execution. |

---

## Research Paper & Citation

If you use DRIGS in your research, please cite our manuscript:

```bibtex
@inproceedings{borkar2026drigs,
  title={DRIGS: Distributed Resource \& Intelligent GPU Scheduling for Heterogeneous AI Workloads},
  author={Borkar, Omdeep},
  booktitle={IEEE International Conference on Distributed Computing Systems (ICDCS)},
  year={2026}
}
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
