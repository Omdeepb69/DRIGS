"""Native Process Execution Backend for DRIGS."""

import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import AsyncIterator, Dict, List, Optional, Set

try:
    import psutil
except ImportError:
    psutil = None

from drigs.core.interfaces import ExecutionHandle
from drigs.core.models import (
    Job,
    JobStatus,
    ResourceAllocation,
)

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Exception raised for errors during workload execution."""
    pass


class ProcessRecord:
    """Internal record tracking a managed native process."""

    def __init__(
        self,
        handle: ExecutionHandle,
        popen: subprocess.Popen,
        log_filepath: Path,
    ):
        self.handle = handle
        self.popen = popen
        self.log_filepath = log_filepath
        self.completed_status: Optional[JobStatus] = None


class NativeProcessBackend:
    """Native OS subprocess execution backend with GPU isolation, process tree cleanup, and orphan sweeper."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(tempfile.gettempdir()) / "drigs_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._processes: Dict[str, ProcessRecord] = {}
        self._sweeper_task: Optional[asyncio.Task] = None
        self._sweeper_running = False

    def launch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Launch job as a native process bound to allocated devices."""
        with self._lock:
            exec_config = job.spec.execution
            import shlex
            entrypoint = exec_config.entrypoint
            if isinstance(entrypoint, str):
                cmd_parts = shlex.split(entrypoint)
            else:
                cmd_parts = list(entrypoint)

            extra_args = getattr(exec_config, "args", None) or getattr(exec_config, "command_args", None) or []
            if extra_args:
                cmd_parts.extend(extra_args)

            if not cmd_parts:
                raise ExecutionError(f"Job {job.id} has an empty execution entrypoint")

            env = dict(os.environ)
            user_env = getattr(exec_config, "env", None) or getattr(exec_config, "environment", None) or {}
            env.update(user_env)

            env["DRIGS_JOB_ID"] = job.id
            env["DRIGS_ALLOCATION_ID"] = allocation.allocation_id
            env["DRIGS_WORKER_ID"] = allocation.worker_id

            gpu_ids = []
            allocated_devices = getattr(allocation, "allocated_devices", None)
            if allocated_devices:
                for dev in allocated_devices:
                    if getattr(dev, "device_type", None) and str(dev.device_type).upper() == "GPU":
                        if getattr(dev, "pcie_bus_id", None):
                            gpu_ids.append(dev.pcie_bus_id)
                        else:
                            gpu_ids.append(dev.device_id.replace("gpu-", "").replace("GPU-", ""))
            elif allocation.assigned_device_ids:
                for dev_id in allocation.assigned_device_ids:
                    if "gpu" in dev_id.lower() or dev_id.isdigit():
                        gpu_ids.append(dev_id.replace("gpu-", "").replace("GPU-", ""))

            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            elif "CUDA_VISIBLE_DEVICES" not in user_env:
                env["CUDA_VISIBLE_DEVICES"] = ""

            working_dir = exec_config.working_dir or os.getcwd()
            if not os.path.isdir(working_dir):
                Path(working_dir).mkdir(parents=True, exist_ok=True)

            handle_id = f"exec_{job.id}_{uuid.uuid4().hex[:8]}"
            log_filepath = self._log_dir / f"{handle_id}.log"

            log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

            try:
                popen = subprocess.Popen(
                    cmd_parts,
                    cwd=working_dir,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                log_file.close()
                logger.error("Failed to launch process for job %s: %s", job.id, e)
                raise ExecutionError(f"Failed to launch process: {e}") from e

            handle = ExecutionHandle(
                handle_id=handle_id,
                job_id=job.id,
                backend_type="native",
                pid=popen.pid,
                status=JobStatus.RUNNING,
                metadata={
                    "cmd": cmd_parts,
                    "log_file": str(log_filepath),
                    "allocation_id": allocation.allocation_id,
                    "working_dir": working_dir,
                    "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
                },
            )

            record = ProcessRecord(handle=handle, popen=popen, log_filepath=log_filepath)
            self._processes[handle_id] = record
            logger.info("Launched job %s with PID %d (Handle: %s)", job.id, popen.pid, handle_id)
            return handle

    def get_status(self, handle: ExecutionHandle) -> JobStatus:
        """Query current execution status of handle."""
        with self._lock:
            record = self._processes.get(handle.handle_id)
            if not record:
                return handle.status

            if record.completed_status is not None:
                return record.completed_status

            poll_code = record.popen.poll()
            if poll_code is None:
                return JobStatus.RUNNING
            elif poll_code == 0:
                record.completed_status = JobStatus.COMPLETED
            elif poll_code < 0 and -poll_code in (signal.SIGTERM, signal.SIGKILL):
                record.completed_status = JobStatus.CANCELLED
            else:
                record.completed_status = JobStatus.FAILED

            record.handle.status = record.completed_status
            return record.completed_status

    def terminate_process_tree(self, pid: int, timeout: float = 2.0) -> List[int]:
        """Recursively terminate process tree starting at target PID (SIGTERM -> SIGKILL)."""
        terminated_pids: List[int] = []
        if psutil is not None:
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                procs = children + [parent]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                procs = []

            for p in procs:
                try:
                    p.terminate()
                    terminated_pids.append(p.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if procs:
                gone, alive = psutil.wait_procs(procs, timeout=timeout)
                for p in alive:
                    try:
                        p.kill()
                        if p.pid not in terminated_pids:
                            terminated_pids.append(p.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        else:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
                terminated_pids.append(pid)
                time.sleep(min(timeout, 0.5))
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                    terminated_pids.append(pid)
                except Exception:
                    pass

        return terminated_pids

    def stop(self, handle: ExecutionHandle, timeout: float = 2.0) -> bool:
        """Stop/terminate process tree associated with handle."""
        with self._lock:
            record = self._processes.get(handle.handle_id)
            if not record:
                return False

            if record.popen.poll() is not None:
                self.get_status(handle)
                return True

            pid = record.popen.pid
            self.terminate_process_tree(pid, timeout=timeout)

            record.completed_status = JobStatus.CANCELLED
            record.handle.status = JobStatus.CANCELLED
            logger.info("Stopped process tree for handle %s (PID %d)", handle.handle_id, pid)
            return True

    def get_active_process_pids(self) -> Set[int]:
        """Return set of PIDs and child process PIDs actively managed by this backend."""
        with self._lock:
            active_pids: Set[int] = set()
            for record in self._processes.values():
                if record.completed_status is None and record.popen.poll() is None:
                    root_pid = record.popen.pid
                    active_pids.add(root_pid)
                    if psutil is not None:
                        try:
                            parent = psutil.Process(root_pid)
                            for child in parent.children(recursive=True):
                                active_pids.add(child.pid)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            return active_pids

    def sweep_orphan_processes(self, cuda_backend=None) -> List[int]:
        """Scan system for orphaned DRIGS/CUDA child processes and terminate them."""
        active_pids = self.get_active_process_pids()
        orphan_pids: Set[int] = set()

        if psutil is not None:
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'environ']):
                try:
                    info = proc.info
                    pid = info['pid']
                    ppid = info['ppid']
                    env = info.get('environ') or {}

                    if "DRIGS_JOB_ID" in env or "DRIGS_WORKER_ID" in env:
                        if pid not in active_pids:
                            # Process is tagged with DRIGS but not in active backend state
                            # Or parent PID is 1 (reparented init) or dead parent
                            if ppid == 1 or not psutil.pid_exists(ppid) or ppid not in active_pids:
                                orphan_pids.add(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        # Check NVML compute running processes if available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                dev_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(dev_handle)
                for gpu_proc in gpu_procs:
                    gpu_pid = gpu_proc.pid
                    if gpu_pid not in active_pids and gpu_pid not in orphan_pids:
                        if psutil is not None and psutil.pid_exists(gpu_pid):
                            try:
                                proc = psutil.Process(gpu_pid)
                                env = proc.environ()
                                if "DRIGS_JOB_ID" in env or proc.ppid() == 1:
                                    orphan_pids.add(gpu_pid)
                            except Exception:
                                pass
        except Exception:
            pass

        swept_pids: List[int] = []
        for orphan_pid in orphan_pids:
            logger.warning("Sweeping orphan DRIGS/CUDA process PID %d", orphan_pid)
            term_pids = self.terminate_process_tree(orphan_pid)
            swept_pids.extend(term_pids)

        return list(set(swept_pids))

    def start_orphan_sweeper(self, interval: float = 10.0, cuda_backend=None) -> None:
        """Start background task that periodically sweeps orphan processes."""
        if self._sweeper_running:
            return
        self._sweeper_running = True

        async def _sweeper_loop():
            while self._sweeper_running:
                try:
                    await asyncio.sleep(interval)
                    if not self._sweeper_running:
                        break
                    self.sweep_orphan_processes(cuda_backend=cuda_backend)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in orphan sweeper loop: %s", e)

        self._sweeper_task = asyncio.create_task(_sweeper_loop())

    def stop_orphan_sweeper(self) -> None:
        """Stop background orphan sweeper task."""
        self._sweeper_running = False
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            self._sweeper_task = None

    def read_logs(self, handle: ExecutionHandle, lines: Optional[int] = None) -> str:
        """Read accumulated log text from handle's output file."""
        with self._lock:
            record = self._processes.get(handle.handle_id)
            log_path = Path(handle.metadata.get("log_file", "")) if not record else record.log_filepath

        if not log_path or not log_path.exists():
            return ""

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content_lines = f.readlines()
                if lines and len(content_lines) > lines:
                    content_lines = content_lines[-lines:]
                return "".join(content_lines)
        except Exception as e:
            logger.error("Error reading log file %s: %s", log_path, e)
            return f"Error reading log file: {e}"

    async def stream_logs(self, handle: ExecutionHandle) -> AsyncIterator[str]:
        """Yield log stream lines from stdout/stderr of the handle asynchronously."""
        with self._lock:
            record = self._processes.get(handle.handle_id)
            log_path = Path(handle.metadata.get("log_file", "")) if not record else record.log_filepath

        while not log_path.exists():
            await asyncio.sleep(0.1)

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    status = self.get_status(handle)
                    if status not in (JobStatus.RUNNING, JobStatus.PENDING):
                        line = f.readline()
                        if line:
                            yield line
                        break
                    await asyncio.sleep(0.1)
