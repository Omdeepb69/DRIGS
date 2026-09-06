# DRIGS Ephemeral Remote Worker Deployment & Dynamic Cluster Guide

This guide explains how to deploy **DRIGS (Distributed Resource & Intelligent GPU Scheduling)** across a local control plane and ephemeral remote GPU nodes (Kaggle Notebooks, Google Colab instances, or cloud VMs).

---

## 1. Architecture Overview

DRIGS uses a **hardware-agnostic control plane** with an **outbound phone-home worker architecture**. 

```text
                       +-----------------------------+
                       |      DRIGS Controller       |
                       |  (Laptop / Central Server)  |
                       |     REST API (:8000)        |
                       +--------------+--------------+
                                      ^
                                      | Outbound HTTP
                                      | Phone-Home Registration & Heartbeats
                +---------------------+---------------------+
                |                                           |
    +-----------+-----------+                   +-----------+-----------+
    |    Kaggle GPU Worker  |                   |   Local CPU / GPU Worker  |
    |  (1x Tesla T4 / 16GB) |                   |  (Host System Hardware)   |
    +-----------------------+                   +-----------------------+
```

### Key Advantages for Ephemeral Environments
- **Zero Inbound Port Requirement:** Remote worker nodes behind NATs or inside Jupyter/Kaggle environments communicate outbound via HTTP (`POST /v1/workers/register` and `POST /v1/workers/{id}/heartbeat`).
- **Automatic GPU Discovery:** `CUDABackend` automatically discovers attached NVIDIA GPUs, VRAM capacity, and compute capability without manual GPU configuration.
- **Fault-Tolerant Session Death Recovery:** When an ephemeral Kaggle notebook shuts down or times out, DRIGS' `FailureDetector` automatically marks the node `OFFLINE`, and `Rescheduler` restores the job from saved checkpoint manifests onto replacement nodes.

---

## 2. Setting Up the Control Plane (Laptop / Server)

### Step 1: Start DRIGS Control Plane REST API
On your local machine or central server, launch the DRIGS control plane:

```bash
# Start FastAPI Control Plane REST API Server
python3 -m uvicorn drigs.api.server:app --host 0.0.0.0 --port 8000
```

### Step 2: Expose Public API Endpoint (Optional for Remote Networks)
If your remote GPU worker (e.g. Kaggle/Colab) is on an external network, expose the local port `8000` via a public tunnel (such as Cloudflare Tunnel, ngrok, or localtunnel):

```bash
# Example using localtunnel or cloudflared
cloudflared tunnel --url http://localhost:8000
```
*Note your public URL: e.g. `https://drigs-controller.trycloudflare.com`.*

---

## 3. Connecting a Kaggle GPU Worker

In your Kaggle GPU Notebook, run the following bootstrap snippet:

```python
# Install DRIGS package
!pip install git+https://github.com/Omdeepb69/DRIGS.git

# Start DRIGS Remote Worker Agent (phones home to Controller)
!drigs worker agent \
    --controller-url https://drigs-controller.trycloudflare.com \
    --worker-id kaggle-t4-01 \
    --heartbeat-interval 5.0
```

Upon execution, the worker agent will:
1. Initialize `CUDABackend` and discover attached GPUs (e.g., Tesla T4 with 16GB VRAM).
2. Register itself with the central controller (`POST /v1/workers/register`).
3. Send periodic heartbeats every 5 seconds (`POST /v1/workers/kaggle-t4-01/heartbeat`).

---

## 4. Monitoring the Cluster & Submitting Workloads

From your local machine or CLI:

### Inspect Cluster Status
```bash
drigs workers
```

Output:
```text
DRIGS Workers
┌──────────────────┬──────────────────────┬─────────┬──────┬─────────────┐
│ Worker ID        │ Hostname             │ Status  │ GPUs │ Total VRAM  │
├──────────────────┼──────────────────────┼─────────┼──────┼─────────────┤
│ worker-local     │ localhost            │ HEALTHY │    0 │ 0 GB        │
│ kaggle-t4-01     │ kaggle-session-node  │ HEALTHY │    1 │ 16.0 GB     │
└──────────────────┴──────────────────────┴─────────┴──────┴─────────────┘
```

### Submit GPU Workload
Create a workload spec `llama_eval.yaml`:

```yaml
name: llama-eval-job
resources:
  gpus: 1
  gpu_memory_bytes: 12884901888  # 12 GB VRAM
  cpus: 2
execution:
  entrypoint: python
  args: ["-c", "import torch; print('Running CUDA on Tesla T4:', torch.cuda.get_device_name(0))"]
checkpoint:
  enabled: true
  interval_seconds: 60
```

Submit via CLI:
```bash
drigs submit llama_eval.yaml
```

DRIGS automatically selects `kaggle-t4-01` and dispatches the execution payload.

---

## 5. Ephemeral Failure Recovery Demo

To test automated fault tolerance when Kaggle sessions terminate unexpectedly:

Run the included interactive simulation script:

```bash
python3 scripts/demo_kaggle_worker.py
```

### Verification Flow
1. Control Plane starts.
2. Ephemeral Kaggle GPU Worker registers dynamically.
3. GPU Workload is scheduled & assigned to `worker-kaggle-t4-01`.
4. Checkpoint step 200 is persisted.
5. Kaggle session death is simulated (heartbeat timeout).
6. DRIGS detects node failure, deallocates dead node resources, restores checkpoint step 200, and re-queues job for execution.
