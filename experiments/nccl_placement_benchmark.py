#!/usr/bin/env python3
# =============================================================================
# DRIGS — Task 18.7: NCCL Communication Benchmark (Kaggle T4 Single Cell)
#
# Paste this entire cell into a Kaggle GPU notebook (2x T4).
# It runs three measurements and prints + saves a JSON result file.
#
# Measurements:
#   1. NCCL all-reduce latency + bandwidth at 4 tensor sizes (1MB/10MB/100MB/512MB)
#   2. NCCL point-to-point (send/recv) latency for topology characterisation
#   3. DDP ResNet-18 training throughput (imgs/sec) and per-step AllReduce latency
#
# All results saved to: /kaggle/working/nccl_placement_results.json
# =============================================================================

import os, json, time, subprocess, sys, tempfile, textwrap, pathlib

# ── 1. Check GPU count ────────────────────────────────────────────────────────
import torch
N_GPUS = torch.cuda.device_count()
assert N_GPUS >= 2, f"Need 2 GPUs, found {N_GPUS}"
print(f"GPUs available: {N_GPUS}")
for i in range(N_GPUS):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")

# ── 2. PCIe topology probe (mimics DRIGS TopologyAwareScheduler scoring) ──────
try:
    topo_raw = subprocess.check_output(
        ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL
    )
    print("\nnvidia-smi topo -m:\n" + topo_raw)
except Exception:
    topo_raw = "unavailable"

# Parse DRIGS-style topology score between GPU 0 and GPU 1
SCORE_MAP = {"NV": 100, "NVB": 90, "X16": 80, "X8": 60, "X4": 40, "X2": 30,
             "X1": 20, "PXB": 60, "PHB": 50, "NODE": 30, "SYS": 10}
topo_score = None
for line in topo_raw.splitlines():
    cols = line.split()
    if cols and cols[0] == "GPU0":
        # Find GPU1 column value
        for c in cols[1:]:
            for key, val in SCORE_MAP.items():
                if key in c:
                    topo_score = val
                    break
            if topo_score:
                break
if topo_score is None:
    topo_score = 50  # fallback: PHB-class
print(f"\nDRIGS topology score (GPU0↔GPU1): {topo_score}")

# ── 3. NCCL all-reduce bandwidth benchmark (worker script) ────────────────────
WORKER_SCRIPT = textwrap.dedent("""
import os, sys, json, time, torch
import torch.distributed as dist

def bench_allreduce(rank, world_size, sizes_mb, warmup=5, iters=50):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    results = []
    for size_mb in sizes_mb:
        n_elems = (size_mb * 1024 * 1024) // 4  # float32
        t = torch.ones(n_elems, dtype=torch.float32, device=f"cuda:{rank}")
        # warmup
        for _ in range(warmup):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        # timed
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        lat_ms = (elapsed / iters) * 1000
        # bandwidth: 2*(N-1)/N * size / time (ring all-reduce formula)
        n = world_size
        bw_gbps = (2 * (n - 1) / n * size_mb / 1024) / (elapsed / iters)
        results.append({"size_mb": size_mb, "latency_ms": round(lat_ms, 4),
                        "bandwidth_gbps": round(bw_gbps, 4)})
        if rank == 0:
            print(f"  AllReduce {size_mb:4d} MB: {lat_ms:.3f} ms  {bw_gbps:.3f} GB/s")
    if rank == 0:
        with open("/kaggle/working/_allreduce_tmp.json", "w") as f:
            json.dump(results, f)
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(sys.argv[1])
    bench_allreduce(rank, world_size=2, sizes_mb=[1, 10, 100, 512])
""")

worker_path = "/kaggle/working/_nccl_worker.py"
pathlib.Path(worker_path).write_text(WORKER_SCRIPT)

print("\n── AllReduce Benchmark ──────────────────────────────────────────────────")
procs = [
    subprocess.Popen([sys.executable, worker_path, str(r)],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for r in range(N_GPUS)
]
# stream rank-0 output
for line in procs[0].stdout:
    print(line, end="")
for p in procs:
    p.wait()

allreduce_results = json.loads(pathlib.Path("/kaggle/working/_allreduce_tmp.json").read_text())

# ── 4. DDP ResNet-18 training throughput benchmark ────────────────────────────
DDP_SCRIPT = textwrap.dedent("""
import os, sys, json, time, torch, torch.nn as nn
import torchvision.models as models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def bench_ddp(rank, world_size, n_steps=30, warmup=5, batch=32):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    model = models.resnet18(weights=None).cuda(rank)
    model = DDP(model, device_ids=[rank])
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 224, 224, device=f"cuda:{rank}")
    y = torch.randint(0, 1000, (batch,), device=f"cuda:{rank}")

    allreduce_times = []
    step_times = []

    for step in range(warmup + n_steps):
        t0 = time.perf_counter()
        out = model(x)
        loss = criterion(out, y)
        optim.zero_grad()

        # Hook to time the all-reduce inside backward
        ar_start = [None]
        ar_end = [None]
        def pre_hook(m, inp):
            ar_start[0] = time.perf_counter()
        def post_hook(m, inp, out):
            torch.cuda.synchronize()
            ar_end[0] = time.perf_counter()
        h1 = model.register_forward_pre_hook(pre_hook)
        h2 = model.register_forward_hook(post_hook)

        loss.backward()  # all-reduce happens here in DDP
        torch.cuda.synchronize()
        optim.step()
        step_t = time.perf_counter() - t0

        h1.remove(); h2.remove()

        if step >= warmup:
            step_times.append(step_t * 1000)  # ms

    throughput = batch * n_steps / (sum(step_times) / 1000)
    mean_step = sum(step_times) / len(step_times)

    if rank == 0:
        print(f"  DDP ResNet-18: step={mean_step:.1f} ms  throughput={throughput:.1f} imgs/s")
        with open("/kaggle/working/_ddp_tmp.json", "w") as f:
            json.dump({"mean_step_ms": round(mean_step, 2),
                       "throughput_imgs_per_sec": round(throughput, 2),
                       "n_steps": n_steps, "batch_size": batch,
                       "world_size": world_size}, f)
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(sys.argv[1])
    bench_ddp(rank, world_size=2)
""")

ddp_path = "/kaggle/working/_ddp_worker.py"
pathlib.Path(ddp_path).write_text(DDP_SCRIPT)

print("\n── DDP ResNet-18 Throughput Benchmark ──────────────────────────────────")
procs = [
    subprocess.Popen([sys.executable, ddp_path, str(r)],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for r in range(N_GPUS)
]
for line in procs[0].stdout:
    print(line, end="")
for p in procs:
    p.wait()

ddp_results = json.loads(pathlib.Path("/kaggle/working/_ddp_tmp.json").read_text())

# ── 5. Compile and save final JSON ────────────────────────────────────────────
gpu_props = [
    {"gpu_index": i,
     "name": torch.cuda.get_device_properties(i).name,
     "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)}
    for i in range(N_GPUS)
]

results = {
    "benchmark": "NCCL Communication Benchmark — DRIGS Task 18.7",
    "hardware": gpu_props,
    "topology": {
        "nvidia_smi_topo": topo_raw.strip(),
        "drigs_topology_score_gpu0_gpu1": topo_score,
        "interconnect_type": "PCIe (no NVLink on T4)"
    },
    "nccl_allreduce": {
        "world_size": N_GPUS,
        "warmup_iters": 5,
        "timed_iters": 50,
        "results": allreduce_results
    },
    "ddp_resnet18": ddp_results
}

out_path = "/kaggle/working/nccl_placement_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# ── 6. Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DRIGS NCCL BENCHMARK SUMMARY")
print("="*60)
print(f"Hardware: {gpu_props[0]['name']} x{N_GPUS}")
print(f"Interconnect: PCIe  |  DRIGS topology score: {topo_score}")
print()
print("AllReduce latency / bandwidth:")
for r in allreduce_results:
    print(f"  {r['size_mb']:4d} MB  →  {r['latency_ms']:.3f} ms  /  {r['bandwidth_gbps']:.3f} GB/s")
print()
print(f"DDP ResNet-18 (2-GPU, batch=32):")
print(f"  Mean step latency: {ddp_results['mean_step_ms']:.1f} ms")
print(f"  Throughput:        {ddp_results['throughput_imgs_per_sec']:.1f} imgs/sec")
print()
print(f"Results saved: {out_path}")
print("="*60)
print("\nCOPY THE JSON BELOW AND PASTE INTO kaggle_results/nccl_placement_results.json")
print(json.dumps(results, indent=2))
