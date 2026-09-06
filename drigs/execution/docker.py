"""Docker Container Execution Backend for DRIGS."""

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from threading import RLock
from typing import AsyncIterator, Dict, List, Optional

from drigs.core.interfaces import ExecutionHandle
from drigs.core.models import Job, JobStatus, ResourceAllocation
from drigs.execution.native import ExecutionError, NativeProcessBackend, ProcessRecord

logger = logging.getLogger(__name__)


class DockerBackend:
    """Docker container execution backend with GPU passthrough (--gpus), resource limits, and volume mounts."""

    def __init__(
        self,
        docker_cmd: str = "docker",
        fallback_to_native: bool = True,
        log_dir: Optional[str] = None,
    ):
        self.docker_cmd = docker_cmd
        self.fallback_to_native = fallback_to_native
        self.native_backend = NativeProcessBackend(log_dir=log_dir) if fallback_to_native else None

        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(tempfile.gettempdir()) / "drigs_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._processes: Dict[str, ProcessRecord] = {}

    def is_docker_available(self) -> bool:
        """Check if docker command exists and daemon is responsive."""
        if not shutil.which(self.docker_cmd):
            return False
        try:
            res = subprocess.run(
                [self.docker_cmd, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            return res.returncode == 0
        except Exception:
            return False

    def build_docker_cmd(self, job: Job, allocation: ResourceAllocation, container_name: str) -> List[str]:
        """Construct `docker run` command with GPU passthrough, volume mounts, env vars, and resource limits."""
        exec_config = job.spec.execution
        runtime_config = getattr(job.spec, "runtime", None)

        image = (
            getattr(runtime_config, "container_image", None)
            or getattr(exec_config, "container_image", None)
            or getattr(exec_config, "image", None)
            or "python:3.10-slim"
        )

        cmd = [
            self.docker_cmd,
            "run",
            "--rm",
            "--name",
            container_name,
        ]

        # 1. GPU Passthrough
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
            cmd.extend(["--gpus", f'device={",".join(gpu_ids)}'])

        # 2. Resource limits (CPUs & Memory)
        req = job.spec.resources
        if req.cpus > 0:
            cmd.extend(["--cpus", str(req.cpus)])
        if req.memory_bytes > 0:
            cmd.extend(["--memory", f"{req.memory_bytes}b"])

        # 3. Environment variables
        user_env = getattr(exec_config, "env", None) or getattr(exec_config, "environment", None) or {}
        env_vars = dict(user_env)
        env_vars["DRIGS_JOB_ID"] = job.id
        env_vars["DRIGS_ALLOCATION_ID"] = allocation.allocation_id
        env_vars["DRIGS_WORKER_ID"] = allocation.worker_id

        for k, v in env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        # 4. Volume bindings
        if runtime_config and getattr(runtime_config, "volumes", None):
            rt_vols = runtime_config.volumes
            if isinstance(rt_vols, dict):
                for host_path, container_path in rt_vols.items():
                    cmd.extend(["-v", f"{host_path}:{container_path}:rw"])
            elif isinstance(rt_vols, list):
                for vol in rt_vols:
                    cmd.extend(["-v", vol])

        exec_vols = getattr(exec_config, "volumes", None) or getattr(exec_config, "volume_mounts", None) or []
        for vol in exec_vols:
            if isinstance(vol, str):
                cmd.extend(["-v", vol])
            elif isinstance(vol, dict):
                host_path = vol.get("host_path") or vol.get("source")
                container_path = vol.get("container_path") or vol.get("target")
                mode = vol.get("mode", "rw")
                if host_path and container_path:
                    cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])

        # Working dir if set
        working_dir = exec_config.working_dir
        if working_dir:
            cmd.extend(["-w", working_dir])

        # Image
        cmd.append(image)

        # Entrypoint / command
        entrypoint = exec_config.entrypoint
        if entrypoint:
            if isinstance(entrypoint, str):
                cmd.extend(shlex.split(entrypoint))
            else:
                cmd.extend(list(entrypoint))

        extra_args = getattr(exec_config, "args", None) or getattr(exec_config, "command_args", None) or []
        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def launch(self, job: Job, allocation: ResourceAllocation) -> ExecutionHandle:
        """Launch job container or fallback to native process execution if docker is unavailable."""
        if not self.is_docker_available():
            if self.fallback_to_native and self.native_backend:
                logger.warning(
                    "Docker daemon unavailable for job %s. Falling back to NativeProcessBackend.",
                    job.id,
                )
                return self.native_backend.launch(job, allocation)
            raise ExecutionError(f"Docker daemon is unavailable to launch job {job.id}")

        with self._lock:
            container_name = f"drigs_container_{job.id}_{uuid.uuid4().hex[:6]}"
            docker_cmd_parts = self.build_docker_cmd(job, allocation, container_name)

            handle_id = f"docker_{job.id}_{uuid.uuid4().hex[:8]}"
            log_filepath = self._log_dir / f"{handle_id}.log"
            log_file = open(log_filepath, "w", encoding="utf-8", buffering=1)

            try:
                popen = subprocess.Popen(
                    docker_cmd_parts,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception as e:
                log_file.close()
                logger.error("Failed to launch docker container for job %s: %s", job.id, e)
                raise ExecutionError(f"Failed to launch docker container: {e}") from e

            handle = ExecutionHandle(
                handle_id=handle_id,
                job_id=job.id,
                backend_type="docker",
                pid=popen.pid,
                status=JobStatus.RUNNING,
                metadata={
                    "container_name": container_name,
                    "cmd": docker_cmd_parts,
                    "log_file": str(log_filepath),
                    "allocation_id": allocation.allocation_id,
                },
            )

            record = ProcessRecord(handle=handle, popen=popen, log_filepath=log_filepath)
            self._processes[handle_id] = record
            logger.info("Launched container %s (Handle: %s)", container_name, handle_id)
            return handle

    def get_status(self, handle: ExecutionHandle) -> JobStatus:
        """Query execution status of handle."""
        if handle.backend_type == "native" and self.native_backend:
            return self.native_backend.get_status(handle)

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
            else:
                record.completed_status = JobStatus.FAILED

            record.handle.status = record.completed_status
            return record.completed_status

    def stop(self, handle: ExecutionHandle) -> bool:
        """Stop container associated with handle."""
        if handle.backend_type == "native" and self.native_backend:
            return self.native_backend.stop(handle)

        with self._lock:
            record = self._processes.get(handle.handle_id)
            if not record:
                return False

            container_name = handle.metadata.get("container_name")
            if container_name:
                try:
                    subprocess.run(
                        [self.docker_cmd, "stop", "-t", "2", container_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                except Exception as e:
                    logger.warning("Error stopping container %s: %s", container_name, e)

            if record.popen.poll() is None:
                try:
                    record.popen.terminate()
                    record.popen.wait(timeout=2.0)
                except Exception:
                    record.popen.kill()

            record.completed_status = JobStatus.CANCELLED
            record.handle.status = JobStatus.CANCELLED
            return True

    async def stream_logs(self, handle: ExecutionHandle) -> AsyncIterator[str]:
        """Stream log output for handle."""
        if handle.backend_type == "native" and self.native_backend:
            async for line in self.native_backend.stream_logs(handle):
                yield line
            return

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
