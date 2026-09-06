"""Native Process Execution Backend for DRIGS."""

import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from threading import RLock
from typing import AsyncIterator, Dict, List, Optional

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
    """Native OS subprocess execution backend with GPU isolation and log streaming."""

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(tempfile.gettempdir()) / "drigs_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._processes: Dict[str, ProcessRecord] = {}

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

    def stop(self, handle: ExecutionHandle) -> bool:
        """Stop/terminate process associated with handle."""
        with self._lock:
            record = self._processes.get(handle.handle_id)
            if not record:
                return False

            if record.popen.poll() is not None:
                self.get_status(handle)
                return True

            pid = record.popen.pid
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning("Error terminating process group for PID %d: %s", pid, e)
                try:
                    record.popen.terminate()
                except ProcessLookupError:
                    pass

            try:
                record.popen.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    record.popen.kill()
                record.popen.wait(timeout=2.0)

            record.completed_status = JobStatus.CANCELLED
            record.handle.status = JobStatus.CANCELLED
            logger.info("Stopped process handle %s (PID %d)", handle.handle_id, pid)
            return True

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
