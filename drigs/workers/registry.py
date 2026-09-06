"""Central Worker Registry for cluster node tracking and heartbeat monitoring."""

from datetime import datetime, timezone
import logging
from threading import RLock
from typing import Dict, List, Optional

from drigs.core.interfaces import WorkerRegistryProtocol
from drigs.core.models import DeviceState, WorkerInfo

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class WorkerRegistry:
    """Thread-safe central registry tracking active worker nodes and detecting heartbeat timeouts."""

    def __init__(self, timeout_seconds: float = 15.0):
        self.timeout_seconds = timeout_seconds
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = RLock()

    def register(self, worker: WorkerInfo) -> bool:
        """Register a new or re-connected worker node."""
        with self._lock:
            # Re-create worker with updated heartbeat timestamp
            updated_worker = WorkerInfo(
                worker_id=worker.worker_id,
                hostname=worker.hostname,
                ip_address=worker.ip_address,
                devices=worker.devices,
                total_cpus=worker.total_cpus,
                total_memory_bytes=worker.total_memory_bytes,
                status=worker.status,
                last_heartbeat=worker.last_heartbeat or _now_utc(),
            )
            self._workers[worker.worker_id] = updated_worker
            logger.info("Registered worker %s (%s)", worker.worker_id, worker.hostname)
            return True

    def heartbeat(self, worker_id: str, status: DeviceState = DeviceState.HEALTHY) -> bool:
        """Record periodic heartbeat ping from a worker node."""
        with self._lock:
            existing = self._workers.get(worker_id)
            if not existing:
                logger.warning("Received heartbeat for unregistered worker %s", worker_id)
                return False

            updated_worker = WorkerInfo(
                worker_id=existing.worker_id,
                hostname=existing.hostname,
                ip_address=existing.ip_address,
                devices=existing.devices,
                total_cpus=existing.total_cpus,
                total_memory_bytes=existing.total_memory_bytes,
                status=status,
                last_heartbeat=_now_utc(),
            )
            self._workers[worker_id] = updated_worker
            return True

    def deregister(self, worker_id: str) -> bool:
        """Deregister a worker node from the cluster."""
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("Deregistered worker %s", worker_id)
                return True
            return False

    def check_timeouts(self) -> List[str]:
        """Scan all workers and transition status for nodes exceeding timeout thresholds."""
        with self._lock:
            now = _now_utc()
            timed_out_ids: List[str] = []

            for worker_id, worker in list(self._workers.items()):
                elapsed = (now - worker.last_heartbeat).total_seconds()
                if elapsed > self.timeout_seconds:
                    if worker.status != DeviceState.OFFLINE:
                        logger.warning(
                            "Worker %s timed out (%.1fs since last heartbeat). Marking OFFLINE.",
                            worker_id,
                            elapsed,
                        )
                        updated_worker = WorkerInfo(
                            worker_id=worker.worker_id,
                            hostname=worker.hostname,
                            ip_address=worker.ip_address,
                            devices=worker.devices,
                            total_cpus=worker.total_cpus,
                            total_memory_bytes=worker.total_memory_bytes,
                            status=DeviceState.OFFLINE,
                            last_heartbeat=worker.last_heartbeat,
                        )
                        self._workers[worker_id] = updated_worker
                        timed_out_ids.append(worker_id)

            return timed_out_ids

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Fetch WorkerInfo for worker_id after evaluating timeouts."""
        with self._lock:
            self.check_timeouts()
            return self._workers.get(worker_id)

    def get_active_workers(self) -> List[WorkerInfo]:
        """Return list of all currently active and healthy worker nodes."""
        with self._lock:
            self.check_timeouts()
            active = []
            for worker in self._workers.values():
                if worker.status in (DeviceState.HEALTHY, DeviceState.DEGRADED):
                    active.append(worker)
            return active

    def get_all_workers(self) -> List[WorkerInfo]:
        """Return list of all registered workers regardless of health state."""
        with self._lock:
            self.check_timeouts()
            return list(self._workers.values())

