"""Checkpoint Manager for job state persistence and recovery step tracking."""

import json
import logging
from pathlib import Path

import tempfile
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages workload checkpoint step manifests, storage paths, and recovery metadata."""

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path(tempfile.gettempdir()) / "drigs_checkpoints"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _get_job_dir(self, job_id: str) -> Path:
        job_dir = self.storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_checkpoint(
        self,
        job_id: str,
        step: int,
        checkpoint_data: Dict[str, Any],
        filepath: Optional[str] = None,
    ) -> str:
        """Save a checkpoint manifest for a job step."""
        with self._lock:
            job_dir = self._get_job_dir(job_id)
            manifest_path = job_dir / f"checkpoint_step_{step}.json"

            manifest = {
                "job_id": job_id,
                "step": step,
                "filepath": filepath or str(manifest_path),
                "data": checkpoint_data,
            }

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info("Saved checkpoint for job %s step %d at %s", job_id, step, manifest_path)
            return str(manifest_path)

    def list_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        """List all saved checkpoint manifests for a job, sorted by step ascending."""
        with self._lock:
            job_dir = self._get_job_dir(job_id)
            manifests = []
            for file_path in job_dir.glob("checkpoint_step_*.json"):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    manifests.append(data)
                except Exception as err:
                    logger.warning("Could not parse checkpoint file %s: %s", file_path, err)

            manifests.sort(key=lambda m: m.get("step", 0))
            return manifests

    def get_latest_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest checkpoint manifest for a job."""
        manifests = self.list_checkpoints(job_id)
        return manifests[-1] if manifests else None

    def cleanup_checkpoints(self, job_id: str, keep_last_n: int = 3) -> int:
        """Delete older checkpoint files beyond keep_last_n."""
        with self._lock:
            manifests = self.list_checkpoints(job_id)
            if len(manifests) <= keep_last_n:
                return 0

            to_delete = manifests[:-keep_last_n]
            deleted_count = 0
            for m in to_delete:
                p = Path(m.get("filepath", ""))
                if p.exists():
                    try:
                        p.unlink()
                        deleted_count += 1
                    except Exception as err:
                        logger.warning("Error deleting checkpoint file %s: %s", p, err)

            return deleted_count
