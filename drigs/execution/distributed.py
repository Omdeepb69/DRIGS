"""Distributed Execution Backend for PyTorch Distributed (DDP / NCCL) workloads."""

import asyncio
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import tempfile
from threading import RLock
from typing import AsyncIterator, Dict, List, Optional
import uuid

from drigs.core.interfaces import ExecutionHandle
from drigs.core.models import (
    Job,
    JobStatus,
    ResourceAllocation,
)
from drigs.execution.isolation import clean_gpu_id, format_cuda_visible_devices

logger = logging.getLogger(__name__)


def find_free_port(default_port: int = 29500) -> int:
    """Find an available TCP port for PyTorch Distributed MASTER_PORT rendezvous."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]
    except Exception:
        return default_port


class DistributedProcessRecord:
    """Record tracking a multi-process PyTorch Distributed job execution."""

    def __init__(
        self,
        handle: ExecutionHandle,
        processes: List[subprocess.Popen],
        log_filepaths: List[Path],
    ):
        self.handle = handle
        self.processes = processes
        self.log_filepaths = log_filepaths
        self.completed_status: Optional[JobStatus] = None


class DistributedBackend:
    """Orchestrates multi-process PyTorch Distributed execution with rank environment isolation."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(tempfile.gettempdir()) / "drigs_distributed_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._records: Dict[str, DistributedProcessRecord] = {}

    def launch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Launch PyTorch Distributed workload processes with rank environment isolation."""
        with self._lock:
            exec_config = job.spec.execution
            dist_config = job.spec.distribution
            entrypoint = exec_config.entrypoint

            if isinstance(entrypoint, str):
                cmd_parts = shlex.split(entrypoint)
            else:
                cmd_parts = list(entrypoint)

            extra_args = getattr(exec_config, "args", None) or getattr(exec_config, "command_args", None) or []
            if extra_args:
                cmd_parts.extend(extra_args)

            if not cmd_parts:
                raise ValueError(f"Job {job.id} has an empty execution entrypoint")

            gpus = allocation.assigned_device_ids
            world_size = dist_config.world_size if dist_config.world_size > 1 else max(1, len(gpus))
            master_addr = dist_config.master_addr or "127.0.0.1"
            master_port = dist_config.master_port or find_free_port()

            processes: List[subprocess.Popen] = []
            log_files: List[Path] = []

            for rank in range(world_size):
                env = dict(os.environ)
                user_env = getattr(exec_config, "env", None) or getattr(exec_config, "environment", None) or {}
                env.update(user_env)

                env["DRIGS_JOB_ID"] = job.id
                env["DRIGS_ALLOCATION_ID"] = allocation.allocation_id
                env["DRIGS_WORKER_ID"] = allocation.worker_id

                # Distributed Rendezvous Envs
                env["WORLD_SIZE"] = str(world_size)
                env["RANK"] = str(rank)
                env["LOCAL_RANK"] = str(rank)
                env["MASTER_ADDR"] = str(master_addr)
                env["MASTER_PORT"] = str(master_port)

                # GPU isolation per rank if GPUs are allocated
                if gpus and rank < len(gpus):
                    env["CUDA_VISIBLE_DEVICES"] = clean_gpu_id(gpus[rank])
                elif gpus:
                    env["CUDA_VISIBLE_DEVICES"] = format_cuda_visible_devices(gpus)

                log_path = self._log_dir / f"{job.id}_rank_{rank}.log"
                log_files.append(log_path)
                f_out = open(log_path, "w", encoding="utf-8")

                working_dir = exec_config.working_dir or os.getcwd()

                try:
                    p = subprocess.Popen(
                        cmd_parts,
                        env=env,
                        cwd=working_dir,
                        stdout=f_out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append(p)
                except Exception as e:
                    logger.error("Failed to launch rank %d process for job %s: %s", rank, job.id, e)
                    f_out.close()
                    # Terminate any already spawned rank processes
                    for spawned in processes:
                        try:
                            os.killpg(os.getpgid(spawned.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    raise RuntimeError(f"Failed to launch distributed rank {rank}: {e}") from e
                finally:
                    f_out.close()

            handle = ExecutionHandle(
                handle_id=str(uuid.uuid4()),
                job_id=job.id,
                backend_type="DISTRIBUTED",
                pid=processes[0].pid if processes else None,
                status=JobStatus.RUNNING,
                metadata={
                    "world_size": world_size,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "log_files": [str(p) for p in log_files],
                },
            )

            record = DistributedProcessRecord(
                handle=handle,
                processes=processes,
                log_filepaths=log_files,
            )
            self._records[handle.handle_id] = record
            return handle

    def get_status(self, handle: ExecutionHandle) -> JobStatus:
        """Poll operational status of all rank processes in a distributed job."""
        with self._lock:
            record = self._records.get(handle.handle_id)
            if not record:
                return handle.status

            if record.completed_status:
                return record.completed_status

            statuses = []
            failed = False

            for p in record.processes:
                ret = p.poll()
                if ret is None:
                    statuses.append(None)
                elif ret == 0:
                    statuses.append(0)
                else:
                    statuses.append(ret)
                    failed = True

            if failed:
                record.completed_status = JobStatus.FAILED
                handle.status = JobStatus.FAILED
                return JobStatus.FAILED

            if all(s == 0 for s in statuses):
                record.completed_status = JobStatus.COMPLETED
                handle.status = JobStatus.COMPLETED
                return JobStatus.COMPLETED

            handle.status = JobStatus.RUNNING
            return JobStatus.RUNNING

    def stop(self, handle: ExecutionHandle) -> bool:
        """Stop all rank process groups for a distributed job handle."""
        with self._lock:
            record = self._records.get(handle.handle_id)
            if not record:
                return False

            for p in record.processes:
                if p.poll() is None:
                    try:
                        pgid = os.getpgid(p.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        p.terminate()

            for p in record.processes:
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        p.kill()

            record.completed_status = JobStatus.CANCELLED
            handle.status = JobStatus.CANCELLED
            return True

    def read_logs(self, handle: ExecutionHandle, rank: Optional[int] = None) -> str:
        """Read log outputs across rank processes."""
        with self._lock:
            record = self._records.get(handle.handle_id)
            log_paths = record.log_filepaths if record else [Path(p) for p in handle.metadata.get("log_files", [])]

        if not log_paths:
            return ""

        if rank is not None and 0 <= rank < len(log_paths):
            p = log_paths[rank]
            return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

        combined = []
        for i, p in enumerate(log_paths):
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="replace")
                lines = [f"[Rank {i}] {line}" for line in text.splitlines()]
                combined.extend(lines)
        return "\n".join(combined)

    async def stream_logs(self, handle: ExecutionHandle) -> AsyncIterator[str]:
        """Stream log lines asynchronously across rank processes."""
        record = self._records.get(handle.handle_id)
        if not record:
            return

        last_positions = [0] * len(record.log_filepaths)

        while True:
            active = False
            for r_idx, path in enumerate(record.log_filepaths):
                if path.exists():
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_positions[r_idx])
                            new_lines = f.readlines()
                            last_positions[r_idx] = f.tell()
                            for line in new_lines:
                                yield f"[Rank {r_idx}] {line.rstrip()}"
                    except Exception:
                        pass

            status = self.get_status(handle)
            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            await asyncio.sleep(0.2)
