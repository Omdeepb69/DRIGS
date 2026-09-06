"""Remote Worker Agent Bootstrap Client for DRIGS Control Plane Registration."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import List, Optional
import httpx

from drigs.core.interfaces import HardwareBackend, WorkerRegistryProtocol
from drigs.core.models import DeviceState, WorkerInfo
from drigs.workers.agent import WorkerAgent

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RemoteHTTPWorkerRegistryClient(WorkerRegistryProtocol):
    """HTTP REST client implementing WorkerRegistryProtocol for outbound worker registration & heartbeat dispatch."""

    def __init__(self, controller_url: str, timeout: float = 10.0):
        self.controller_url = controller_url.rstrip("/")
        self.timeout = timeout

    def _get_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.controller_url, timeout=self.timeout)

    def register(self, worker: WorkerInfo) -> bool:
        """Register worker with central DRIGS REST control plane API."""
        try:
            payload = worker.model_dump(mode="json")
            with self._get_client() as client:
                res = client.post("/v1/workers/register", json=payload)
                if res.status_code in (200, 201):
                    logger.info("Successfully registered remote worker %s with controller at %s", worker.worker_id, self.controller_url)
                    return True
                logger.warning("Remote worker registration failed (%d): %s", res.status_code, res.text)
                return False
        except Exception as err:
            logger.error("Connection error registering worker %s at %s: %s", worker.worker_id, self.controller_url, err)
            return False

    def heartbeat(self, worker_id: str, status: DeviceState = DeviceState.HEALTHY) -> bool:
        """Send periodic heartbeat ping to central DRIGS REST control plane API."""
        try:
            with self._get_client() as client:
                res = client.post(f"/v1/workers/{worker_id}/heartbeat", params={"status": status.value})
                if res.status_code == 200:
                    return True
                logger.warning("Remote worker heartbeat failed (%d): %s", res.status_code, res.text)
                return False
        except Exception as err:
            logger.error("Connection error sending heartbeat for worker %s: %s", worker_id, err)
            return False

    def deregister(self, worker_id: str) -> bool:
        """Deregister worker node from remote control plane."""
        try:
            with self._get_client() as client:
                res = client.post(f"/v1/workers/{worker_id}/deregister")
                return res.status_code == 200
        except Exception as err:
            logger.error("Error deregistering worker %s: %s", worker_id, err)
            return False

    def get_active_workers(self) -> List[WorkerInfo]:
        """Fetch list of active registered workers from remote control plane."""
        try:
            with self._get_client() as client:
                res = client.get("/v1/workers")
                if res.status_code == 200:
                    data = res.json()
                    return [WorkerInfo.model_validate(w) for w in data]
                return []
        except Exception as err:
            logger.error("Error fetching active workers from remote control plane: %s", err)
            return []


def create_remote_worker_agent(
    controller_url: str,
    worker_id: Optional[str] = None,
    heartbeat_interval: float = 5.0,
    use_gpu: bool = True,
) -> WorkerAgent:
    """Construct and configure a WorkerAgent connected to a remote control plane controller URL."""
    registry_client = RemoteHTTPWorkerRegistryClient(controller_url=controller_url)

    hardware_backend: HardwareBackend
    if use_gpu:
        try:
            from drigs.hardware.cuda import CUDABackend
            cuda_backend = CUDABackend()
            devices = cuda_backend.discover_devices()
            if devices:
                hardware_backend = cuda_backend
                logger.info("Discovered %d GPU devices via CUDABackend", len(devices))
            else:
                from drigs.hardware.cpu import CPUBackend
                hardware_backend = CPUBackend()
                logger.info("No GPU devices discovered; falling back to CPUBackend")
        except Exception as err:
            logger.warning("CUDABackend unavailable (%s); falling back to CPUBackend", err)
            from drigs.hardware.cpu import CPUBackend
            hardware_backend = CPUBackend()
    else:
        from drigs.hardware.cpu import CPUBackend
        hardware_backend = CPUBackend()

    agent = WorkerAgent(
        worker_id=worker_id,
        hardware_backend=hardware_backend,
        registry=registry_client,
        heartbeat_interval=heartbeat_interval,
    )
    return agent


def run_remote_worker_agent(
    controller_url: str,
    worker_id: Optional[str] = None,
    heartbeat_interval: float = 5.0,
    use_gpu: bool = True,
    duration_seconds: Optional[float] = None,
) -> WorkerAgent:
    """Start and run a remote worker agent loop connected to a DRIGS control plane URL."""
    agent = create_remote_worker_agent(
        controller_url=controller_url,
        worker_id=worker_id,
        heartbeat_interval=heartbeat_interval,
        use_gpu=use_gpu,
    )

    async def _main():
        await agent.start()
        logger.info("Worker agent %s active and heartbeating to %s", agent.worker_id, controller_url)
        if duration_seconds is not None and duration_seconds > 0:
            await asyncio.sleep(duration_seconds)
            await agent.stop()
        else:
            try:
                while True:
                    await asyncio.sleep(3600)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Shutting down worker agent %s...", agent.worker_id)
                await agent.stop()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Worker agent stopped by user.")

    return agent
