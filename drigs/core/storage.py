"""SQLite Control Plane State Persistence Engine for DRIGS.

This module provides ACID SQLite storage for Job state, Worker registrations, and Resource allocations,
enabling control plane state restoration across process restarts and crashes.
"""

import json
import logging
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from drigs.core.models import Job, JobStatus, ResourceAllocation, WorkerInfo

logger = logging.getLogger(__name__)


class SQLiteStore:
    """ACID SQLite storage backend for DRIGS control plane state persistence."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        self._lock = RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database tables for jobs, workers, and allocations."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS jobs (
                            job_id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            priority INTEGER NOT NULL,
                            status TEXT NOT NULL,
                            submitted_at TEXT NOT NULL,
                            started_at TEXT,
                            completed_at TEXT,
                            error_message TEXT,
                            spec_json TEXT NOT NULL,
                            allocation_json TEXT
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS workers (
                            worker_id TEXT PRIMARY KEY,
                            hostname TEXT NOT NULL,
                            ip_address TEXT NOT NULL,
                            total_cpus INTEGER NOT NULL,
                            total_memory_bytes INTEGER NOT NULL,
                            status TEXT NOT NULL,
                            last_heartbeat TEXT NOT NULL,
                            devices_json TEXT NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS allocations (
                            job_id TEXT PRIMARY KEY,
                            worker_id TEXT NOT NULL,
                            assigned_device_ids_json TEXT NOT NULL,
                            assigned_cpu_cores_json TEXT NOT NULL,
                            memory_bytes INTEGER NOT NULL,
                            allocated_at TEXT NOT NULL
                        )
                        """
                    )
            finally:
                conn.close()

    def save_job(self, job: Job) -> None:
        """Upsert job model into SQLite database."""
        with self._lock:
            conn = self._get_connection()
            try:
                data = job.model_dump(mode="json")
                spec_json = json.dumps(data.get("spec", {}))
                allocation_json = json.dumps(data.get("allocation")) if data.get("allocation") else None

                with conn:
                    conn.execute(
                        """
                        INSERT INTO jobs (
                            job_id, name, priority, status, submitted_at, started_at, completed_at, error_message, spec_json, allocation_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(job_id) DO UPDATE SET
                            status = excluded.status,
                            started_at = excluded.started_at,
                            completed_at = excluded.completed_at,
                            error_message = excluded.error_message,
                            allocation_json = excluded.allocation_json
                        """,
                        (
                            job.id,
                            job.name,
                            job.priority,
                            job.status.value,
                            job.submitted_at.isoformat(),
                            job.started_at.isoformat() if job.started_at else None,
                            job.completed_at.isoformat() if job.completed_at else None,
                            job.error_message,
                            spec_json,
                            allocation_json,
                        ),
                    )
            finally:
                conn.close()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job model by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                job_dict = dict(row)
                job_dict["job_id"] = job_dict["job_id"]
                job_dict["spec"] = json.loads(job_dict["spec_json"])
                if job_dict["allocation_json"]:
                    job_dict["allocation"] = json.loads(job_dict["allocation_json"])
                else:
                    job_dict["allocation"] = None

                job_dict.pop("spec_json", None)
                job_dict.pop("allocation_json", None)

                return Job.model_validate(job_dict)
            finally:
                conn.close()

    def load_all_jobs(self) -> List[Job]:
        """Retrieve all stored jobs ordered by priority and submission time."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs ORDER BY priority DESC, submitted_at ASC")
                rows = cursor.fetchall()
                jobs: List[Job] = []
                for row in rows:
                    job_dict = dict(row)
                    job_dict["job_id"] = job_dict["job_id"]
                    job_dict["spec"] = json.loads(job_dict["spec_json"])
                    if job_dict["allocation_json"]:
                        job_dict["allocation"] = json.loads(job_dict["allocation_json"])
                    else:
                        job_dict["allocation"] = None

                    job_dict.pop("spec_json", None)
                    job_dict.pop("allocation_json", None)
                    jobs.append(Job.model_validate(job_dict))
                return jobs
            finally:
                conn.close()

    def delete_job(self, job_id: str) -> bool:
        """Delete job from SQLite database."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                    return cursor.rowcount > 0
            finally:
                conn.close()

    def save_worker(self, worker: WorkerInfo) -> None:
        """Upsert worker node metadata into SQLite database."""
        with self._lock:
            conn = self._get_connection()
            try:
                data = worker.model_dump(mode="json")
                devices_json = json.dumps(data.get("devices", []))

                with conn:
                    conn.execute(
                        """
                        INSERT INTO workers (
                            worker_id, hostname, ip_address, total_cpus, total_memory_bytes, status, last_heartbeat, devices_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(worker_id) DO UPDATE SET
                            hostname = excluded.hostname,
                            ip_address = excluded.ip_address,
                            total_cpus = excluded.total_cpus,
                            total_memory_bytes = excluded.total_memory_bytes,
                            status = excluded.status,
                            last_heartbeat = excluded.last_heartbeat,
                            devices_json = excluded.devices_json
                        """,
                        (
                            worker.worker_id,
                            worker.hostname,
                            worker.ip_address,
                            worker.total_cpus,
                            worker.total_memory_bytes,
                            worker.status.value,
                            worker.last_heartbeat.isoformat(),
                            devices_json,
                        ),
                    )
            finally:
                conn.close()

    def load_all_workers(self) -> List[WorkerInfo]:
        """Retrieve all stored worker nodes."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM workers")
                rows = cursor.fetchall()
                workers: List[WorkerInfo] = []
                for row in rows:
                    w_dict = dict(row)
                    w_dict["devices"] = json.loads(w_dict["devices_json"])
                    w_dict.pop("devices_json", None)
                    workers.append(WorkerInfo.model_validate(w_dict))
                return workers
            finally:
                conn.close()

    def delete_worker(self, worker_id: str) -> bool:
        """Delete worker node registration from SQLite database."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
                    return cursor.rowcount > 0
            finally:
                conn.close()

    def save_allocation(self, allocation: ResourceAllocation) -> None:
        """Upsert resource allocation into SQLite database."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO allocations (
                            job_id, worker_id, assigned_device_ids_json, assigned_cpu_cores_json, memory_bytes, allocated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(job_id) DO UPDATE SET
                            worker_id = excluded.worker_id,
                            assigned_device_ids_json = excluded.assigned_device_ids_json,
                            assigned_cpu_cores_json = excluded.assigned_cpu_cores_json,
                            memory_bytes = excluded.memory_bytes,
                            allocated_at = excluded.allocated_at
                        """,
                        (
                            allocation.job_id,
                            allocation.worker_id,
                            json.dumps(allocation.assigned_device_ids),
                            json.dumps(allocation.assigned_cpu_cores),
                            allocation.memory_bytes,
                            allocation.allocated_at.isoformat(),
                        ),
                    )
            finally:
                conn.close()

    def load_all_allocations(self) -> List[ResourceAllocation]:
        """Retrieve all active resource allocations."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM allocations")
                rows = cursor.fetchall()
                allocs: List[ResourceAllocation] = []
                for row in rows:
                    a_dict = dict(row)
                    a_dict["assigned_device_ids"] = json.loads(a_dict["assigned_device_ids_json"])
                    a_dict["assigned_cpu_cores"] = json.loads(a_dict["assigned_cpu_cores_json"])
                    a_dict.pop("assigned_device_ids_json", None)
                    a_dict.pop("assigned_cpu_cores_json", None)
                    allocs.append(ResourceAllocation.model_validate(a_dict))
                return allocs
            finally:
                conn.close()

    def delete_allocation(self, job_id: str) -> bool:
        """Delete resource allocation by job ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute("DELETE FROM allocations WHERE job_id = ?", (job_id,))
                    return cursor.rowcount > 0
            finally:
                conn.close()

    def clear(self) -> None:
        """Purge all data from storage tables."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM jobs")
                    conn.execute("DELETE FROM workers")
                    conn.execute("DELETE FROM allocations")
            finally:
                conn.close()
