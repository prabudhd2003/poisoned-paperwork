"""Small, CPU-safe checkpoint helpers shared by GPU stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


BASELINE_REQUIRED_FIELDS = (
    "run_id", "stage", "experiment_key", "record_id", "dataset_name",
    "experiment_split", "model_key", "model_id", "model_revision", "task",
    "prompt_id", "prompt_sha256", "clean_target", "attack_target",
    "attack_eligible", "raw_response", "prediction_normalized", "parse_status",
    "clean_correct", "runtime_seconds", "peak_gpu_memory_mb", "worker_id",
    "seed", "git_commit", "config_sha256", "input_page_count", "status", "error",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    """Flush a temporary file before atomically replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def experiment_key(
    record_id: str, model_id: str, task: str, prompt_id: str, split: str
) -> str:
    """Stable filesystem-safe key containing every experimental factor."""
    factors = {
        "model_id": model_id,
        "prompt_id": prompt_id,
        "record_id": record_id,
        "split": split,
        "task": task,
    }
    return sha256_text(json.dumps(factors, sort_keys=True, separators=(",", ":")))


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def record_is_final(
    path: Path, *, expected: dict[str, Any], config_sha256: str
) -> bool:
    """Accept only a complete or explicit failed record with exact provenance."""
    record = load_json(path)
    if record is None or any(field not in record for field in BASELINE_REQUIRED_FIELDS):
        return False
    if record.get("status") not in {"complete", "failed"}:
        return False
    if record.get("config_sha256") != config_sha256:
        return False
    exact_fields = [
        "experiment_key", "record_id", "dataset_name", "experiment_split",
        "model_key", "model_id", "model_revision", "task", "prompt_id",
        "prompt_sha256",
    ]
    if "git_commit" in expected:
        exact_fields.append("git_commit")
    return all(record.get(field) == expected.get(field) for field in exact_fields)
