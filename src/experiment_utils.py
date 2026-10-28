"""Shared contracts for external-receipt and post-attack experiment stages."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from checkpoints import atomic_json, atomic_text, sha256_file
from sharding import assigned_record_ids


COLLECTED_DATASET = "collected_receipts"
COLLECTED_SPLIT = "external_test"


def repo_root_from_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in path.parents:
        if parent.joinpath(".git").exists() and parent.joinpath("src").is_dir():
            return parent
    raise ValueError(f"config must live inside the poisoned-paperwork repository: {path}")


def git_commit(root: Path, *, require_clean: bool = True) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if require_clean:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
        if status:
            raise RuntimeError("experiment requires a clean committed worktree")
    return commit


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_collected_manifest(root: Path, config: dict[str, Any]):
    """Load and validate the prospectively collected external-test manifest."""
    import pandas as pd

    relative = Path(config.get(
        "manifest", "data/processed/collected_receipts/manifest.csv"
    ))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("collected manifest must be a repository-relative path")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"collected manifest not found: {path}")
    expected_hash = config.get("manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError(
            "freeze the final manifest and replace manifest_sha256 before running"
        )
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"collected manifest hash mismatch: {observed_hash} != {expected_hash}"
        )
    frame = pd.read_csv(path, dtype={"clean_target": str, "attack_target": str})
    required = {
        "record_id", "dataset_name", "experiment_split", "task",
        "processed_path", "processed_sha256", "clean_target", "attack_target",
        "status", "usable", "attack_eligible", "num_pages",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"collected manifest is missing columns: {missing}")
    selected = frame.loc[
        frame.dataset_name.eq(COLLECTED_DATASET)
        & frame.experiment_split.eq(COLLECTED_SPLIT)
        & frame.status.eq("verified")
        & frame.usable.eq(True)
        & frame.attack_eligible.eq(True)
    ].copy()
    expected = int(config.get("expected_documents", 100))
    if len(selected) != expected:
        raise RuntimeError(
            f"collected external set must contain {expected} verified usable receipts; "
            f"found {len(selected)}"
        )
    if selected.record_id.duplicated().any():
        raise RuntimeError("collected manifest contains duplicate record IDs")
    if set(selected.task) != {"receipt_total"} or set(selected.num_pages.astype(int)) != {1}:
        raise RuntimeError("collected set must contain one-page receipt_total records only")
    for row in selected.itertuples(index=False):
        image_path = root / str(row.processed_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"processed collected image missing: {image_path}")
        if sha256_file(image_path) != str(row.processed_sha256):
            raise RuntimeError(f"processed image hash mismatch: {row.record_id}")
    return selected.sort_values("record_id").set_index("record_id", drop=False), path


def load_collected_image(root: Path, row) -> Any:
    from PIL import Image

    path = root / str(row["processed_path"])
    with Image.open(path) as source:
        image = source.convert("RGB").copy()
    return image


def initialize_run(
    *, root: Path, config_path: Path, config: dict[str, Any], stage: str,
    worker_id: int, num_workers: int, record_ids: Iterable[str],
) -> tuple[Path, Path, str, str, list[str]]:
    """Create immutable run metadata and return output/shard provenance."""
    from checkpoints import sha256_bytes

    config_bytes = config_path.read_bytes()
    config_hash = sha256_bytes(config_bytes)
    commit = git_commit(root, require_clean=bool(config.get("require_clean_git", True)))
    output = root / "outputs" / stage / str(config["run_id"])
    shard = output / "shards" / f"worker_{worker_id:02d}"
    copied = output / "config.json"
    if copied.exists() and sha256_file(copied) != config_hash:
        raise RuntimeError("run directory contains a different config")
    if not copied.exists():
        atomic_text(copied, config_bytes.decode("utf-8"))
    all_ids = sorted(record_ids)
    ids = assigned_record_ids(all_ids, worker_id, num_workers)
    assignment = output / "assignments" / f"worker_{worker_id:02d}.txt"
    expected_text = "".join(f"{record_id}\n" for record_id in ids)
    if assignment.exists() and assignment.read_text(encoding="utf-8") != expected_text:
        raise RuntimeError("worker assignment differs from this frozen run")
    atomic_text(assignment, expected_text)
    atomic_json(output / "run_contract.json", {
        "run_id": config["run_id"], "stage": stage,
        "config_sha256": config_hash, "git_commit": commit,
        "num_workers": num_workers, "records": len(all_ids),
    })
    return output, shard, config_hash, commit, ids


def write_worker_complete(
    shard: Path, *, config: dict[str, Any], config_hash: str, commit: str,
    worker_id: int, expected_records: int,
) -> None:
    records = list((shard / "records").glob("*.json"))
    if len(records) != expected_records:
        raise RuntimeError(
            f"worker incomplete: found {len(records)} records; expected {expected_records}"
        )
    atomic_json(shard / "WORKER_COMPLETE.json", {
        "run_id": config["run_id"], "worker_id": worker_id,
        "expected_records": expected_records, "completed_records": len(records),
        "config_sha256": config_hash, "git_commit": commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


def merge_json_records(
    output: Path, *, config: dict[str, Any], config_hash: str,
    num_workers: int, expected_records: int, sort_columns: list[str],
) -> bool:
    """Strict CPU-safe merge used by new terminal stages."""
    import pandas as pd

    markers = [
        output / "shards" / f"worker_{index:02d}" / "WORKER_COMPLETE.json"
        for index in range(num_workers)
    ]
    if not all(path.is_file() for path in markers):
        return False
    lock_path = output / ".merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        commits, rows, keys = set(), [], set()
        for worker_id, marker_path in enumerate(markers):
            marker = read_json(marker_path)
            if marker.get("worker_id") != worker_id:
                raise RuntimeError(f"wrong worker marker: {marker_path}")
            if marker.get("config_sha256") != config_hash:
                raise RuntimeError(f"config drift in worker {worker_id}")
            commits.add(marker.get("git_commit"))
            for path in sorted(marker_path.parent.joinpath("records").glob("*.json")):
                row = read_json(path)
                key = row.get("experiment_key")
                if not key or key in keys:
                    raise RuntimeError(f"missing or duplicate experiment key: {path}")
                keys.add(key)
                rows.append(row)
        if len(commits) != 1 or None in commits:
            raise RuntimeError("workers used different Git commits")
        if len(rows) != expected_records:
            raise RuntimeError(f"merge found {len(rows)} records; expected {expected_records}")
        frame = pd.DataFrame(rows).sort_values(sort_columns)
        merged = output / "merged"
        merged.mkdir(parents=True, exist_ok=True)
        csv_tmp = merged / "records.csv.tmp"
        parquet_tmp = merged / "records.parquet.tmp"
        frame.to_csv(csv_tmp, index=False)
        frame.to_parquet(parquet_tmp, index=False)
        os.replace(csv_tmp, merged / "records.csv")
        os.replace(parquet_tmp, merged / "records.parquet")
        atomic_json(merged / "summary.json", {
            "run_id": config["run_id"], "stage": config["stage"],
            "records": len(frame), "unique_documents": int(frame.record_id.nunique()),
            "failures": int(frame.status.ne("complete").sum()) if "status" in frame else 0,
            "git_commit": next(iter(commits)),
            "merged_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

