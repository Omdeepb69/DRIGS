# DRIGS: Distributed Resource & Intelligent GPU Scheduling

[![PyPI Version](https://img.shields.io/pypi/v/drigs.svg)](https.pypi.org/project/drigs/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**DRIGS** is a lightweight, high-performance, GPU-aware distributed compute runtime designed for scheduling, orchestrating, monitoring, and recovering AI workloads across single-GPU workstations, multi-GPU servers, and dynamic cloud notebook instances (e.g. Kaggle, Google Colab).

---

## Key Features

- ⚡ **Pluggable Policy Core**: Built-in implementations for `FIFOScheduler`, `MemoryAwareScheduler`, `DominantResourceFairness` (`DRFScheduler`), `GangScheduler`, `BestFitScheduler`, `BinPackScheduler`, and `TopologyAwareScheduler`.
- 🔌 **Native & Container Backends**: Supports rootless native processes (`NativeProcessBackend`), Docker containers (`DockerBackend`), and PyTorch DDP / `torchrun` gang synchronization (`DistributedBackend`).
- 📡 **Ephemeral Worker Outbound Discovery**: Outbound phone-home HTTP bootstrap client for remote GPU nodes operating behind NAT firewalls.
- 🛡️ **Process Tree & VRAM Guard**: Automated recursive process tree teardown via `psutil` and real-time NVML orphan PID memory sweeper.
- 💾 **ACID Control Plane State**: Integrated WAL-mode `SQLiteStore` for controller restart resilience and zero-loss queue state recovery.

---

## Quickstart

### 1. Installation

Install via `pip`:

```bash
pip install drigs
```

### 2. Python API Usage

```python
import drigs

# Initialize Local Controller & Resource Manager
controller = drigs.LocalController(
    scheduler=drigs.TopologyAwareScheduler(),
    backend=drigs.NativeProcessBackend()
)

# Define a GPU Workload Spec
spec = drigs.WorkloadSpec(
    name="resnet50-training",
    command="python3 train.py --batch-size 64",
    resources=drigs.ResourceRequirements(gpus=2, cpus=4, memory_bytes=8 * 1024**3)
)

# Submit & Schedule Job
job_id = controller.submit_job(spec)
print(f"Submitted Job ID: {job_id}")
```

### 3. Command Line Interface (CLI)

```bash
# Start DRIGS REST API Control Plane Server
drigs server --host 127.0.0.1 --port 8000

# Check cluster status and GPU telemetry
drigs status

# Submit a job YAML specification
drigs submit job.yaml
```

---

## Research Paper & Citation

If you use DRIGS in your academic research, please cite our manuscript:

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
