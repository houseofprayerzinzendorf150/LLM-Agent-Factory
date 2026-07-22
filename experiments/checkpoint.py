"""
Checkpoint / resume system for experiments.

Saves results incrementally to JSON files so experiments can be
resumed from the exact point where they stopped.
"""

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.agent_runner import AgentAnswer

CHECKPOINT_DIR = Path("experiments/checkpoints")


class CheckpointManager:
    """Thread-safe checkpoint manager for experiment results."""

    def __init__(self, experiment_id: str, checkpoint_dir: Path | None = None):
        self.experiment_id = experiment_id
        self.dir = checkpoint_dir or CHECKPOINT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._results: dict[str, dict] = {}
        self._metadata: dict[str, Any] = {}
        self._load()

    @property
    def filepath(self) -> Path:
        return self.dir / f"{self.experiment_id}.json"

    def _load(self) -> None:
        """Load existing checkpoint from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    data = json.load(f)
                self._results = data.get("results", {})
                self._metadata = data.get("metadata", {})
            except (json.JSONDecodeError, OSError):
                self._results = {}
                self._metadata = {}

    def _save(self) -> None:
        """Save checkpoint to disk (must be called under lock)."""
        data = {
            "experiment_id": self.experiment_id,
            "metadata": self._metadata,
            "results": self._results,
            "total_completed": len(self._results),
        }
        # Write to temp file first, then rename for atomicity
        tmp_path = self.filepath.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # On Windows, need to remove target first
        if self.filepath.exists():
            os.remove(self.filepath)
        os.rename(tmp_path, self.filepath)

    def is_completed(self, sample_id: str) -> bool:
        """Check if a sample has already been processed."""
        with self._lock:
            return sample_id in self._results

    def get_completed_ids(self) -> set[str]:
        """Get set of all completed sample IDs."""
        with self._lock:
            return set(self._results.keys())

    def save_result(self, answer: AgentAnswer) -> None:
        """Save a single result (thread-safe, writes to disk immediately)."""
        with self._lock:
            self._results[answer.sample_id] = asdict(answer)
            self._save()

    def save_results_batch(self, answers: list[AgentAnswer]) -> None:
        """Save a batch of results (thread-safe, single disk write)."""
        with self._lock:
            for answer in answers:
                self._results[answer.sample_id] = asdict(answer)
            self._save()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the experiment."""
        with self._lock:
            self._metadata[key] = value
            self._save()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        with self._lock:
            return self._metadata.get(key, default)

    def get_all_results(self) -> list[dict]:
        """Get all saved results."""
        with self._lock:
            return list(self._results.values())

    @property
    def completed_count(self) -> int:
        """Number of completed samples."""
        with self._lock:
            return len(self._results)

    def clear(self) -> None:
        """Clear all results (for re-running from scratch)."""
        with self._lock:
            self._results = {}
            self._metadata = {}
            if self.filepath.exists():
                os.remove(self.filepath)


def get_checkpoint_id(config_id: str, dataset_name: str) -> str:
    """Generate a checkpoint ID from config and dataset."""
    return f"{config_id}__{dataset_name}"


def list_checkpoints(checkpoint_dir: Path | None = None) -> list[dict]:
    """List all existing checkpoints with their status."""
    d = checkpoint_dir or CHECKPOINT_DIR
    if not d.exists():
        return []

    checkpoints = []
    for f in d.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            checkpoints.append(
                {
                    "experiment_id": data.get("experiment_id", f.stem),
                    "total_completed": data.get("total_completed", 0),
                    "metadata": data.get("metadata", {}),
                    "file": str(f),
                }
            )
        except Exception:
            continue

    return checkpoints
