"""Unit tests for recursive process tree cleanup and orphan GPU process sweeper in NativeProcessBackend."""

import asyncio
import os
import subprocess
import sys
import time
import pytest
import psutil

from drigs.core.models import Job, JobStatus, ResourceAllocation, WorkloadExecutionConfig, WorkloadSpec
from drigs.execution.native import NativeProcessBackend


def test_recursive_process_tree_termination(tmp_path):
    backend = NativeProcessBackend(log_dir=str(tmp_path))
    
    # Spawn a parent python process that spawns a child subprocess
    code = (
        "import subprocess, time; "
        "p = subprocess.Popen(['sleep', '100']); "
        "time.sleep(100)"
    )
    parent_popen = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(0.5)

    parent_pid = parent_popen.pid
    parent_proc = psutil.Process(parent_pid)
    children = parent_proc.children(recursive=True)
    assert len(children) >= 1
    child_pid = children[0].pid

    # Terminate process tree recursively
    terminated_pids = backend.terminate_process_tree(parent_pid, timeout=2.0)
    assert parent_pid in terminated_pids or not parent_proc.is_running()
    
    time.sleep(0.2)
    assert not psutil.pid_exists(parent_pid)
    assert not psutil.pid_exists(child_pid)


def test_orphan_process_sweeping(tmp_path):
    backend = NativeProcessBackend(log_dir=str(tmp_path))

    # 1. Launch a normal managed job
    spec = WorkloadSpec(
        name="managed-job",
        resources={"cpus": 1, "gpus": 0},
        execution=WorkloadExecutionConfig(entrypoint=["sleep", "60"]),
    )
    job = Job(id="managed-job-100", name="managed-job", spec=spec)
    alloc = ResourceAllocation(allocation_id="alloc-100", job_id=job.id, worker_id="worker-1")
    handle = backend.launch(job, alloc)

    # 2. Launch an unmanaged orphan process tagged with DRIGS_JOB_ID
    orphan_env = dict(os.environ)
    orphan_env["DRIGS_JOB_ID"] = "orphan-job-555"
    orphan_popen = subprocess.Popen(["sleep", "60"], env=orphan_env)
    orphan_pid = orphan_popen.pid
    time.sleep(0.2)

    # Managed job should be active, orphan job should be detected
    active_pids = backend.get_active_process_pids()
    assert handle.pid in active_pids
    assert orphan_pid not in active_pids

    # Sweep orphan processes
    swept_pids = backend.sweep_orphan_processes()
    assert orphan_pid in swept_pids

    time.sleep(0.2)
    assert not psutil.pid_exists(orphan_pid)
    # Managed job should remain running
    assert psutil.pid_exists(handle.pid)

    # Clean up managed job
    backend.stop(handle)


def test_orphan_sweeper_background_loop(tmp_path):
    backend = NativeProcessBackend(log_dir=str(tmp_path))

    async def _test():
        backend.start_orphan_sweeper(interval=0.1)
        assert backend._sweeper_running is True

        # Spawn an orphan process
        orphan_env = dict(os.environ)
        orphan_env["DRIGS_JOB_ID"] = "orphan-bg-777"
        orphan_popen = subprocess.Popen(["sleep", "60"], env=orphan_env)
        orphan_pid = orphan_popen.pid

        # Wait for sweeper loop iteration
        await asyncio.sleep(0.3)
        assert not psutil.pid_exists(orphan_pid)

        backend.stop_orphan_sweeper()
        assert backend._sweeper_running is False

    asyncio.run(_test())
