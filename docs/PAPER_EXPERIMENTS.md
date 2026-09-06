# DRIGS Research Paper Empirical Evaluation Summary

This document summarizes empirical benchmark results for paper submission.

## 1. Scheduling Policy Evaluation (Experiment A)
| Scheduler | Throughput (jobs/sec) | Mean Wait (s) | P95 Wait (s) | VRAM Frag. (%) | Placement Eff. (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FIFO** | 0.623 | 0.97 | 9.21 | 56.7% | 100.0% |
| **Priority** | 0.602 | 0.73 | 5.06 | 47.9% | 100.0% |
| **BestFit** | 0.562 | 0.44 | 3.99 | 55.0% | 100.0% |
| **MemoryAware** | 0.602 | 0.72 | 3.59 | 49.4% | 100.0% |
| **TopologyAware** | 0.626 | 0.71 | 2.38 | 50.8% | 100.0% |

## 2. Interconnect Topology Awareness (Experiment B)
- **TopologyAware Score**: `77.78` / 100.0
- **Random Placement Score**: `48.67` / 100.0
- **Relative Interconnect Bandwidth Improvement**: `+59.81%`

## 3. Fault Recovery & Rescheduling Latency (Experiment E)
- **Mean Rescheduling Latency**: `0.54 ms`
- **Job Completion Rate under Failure**: `100.0%`
- **Total Restored Checkpoints**: `2`