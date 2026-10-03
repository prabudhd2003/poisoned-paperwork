"""Checkpointed clean Donut receipt baseline for SROIE and CORD."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.donut import DEFAULT_MODEL_ID, DEFAULT_REVISION, DonutDocVQA
from normalization import normalize_receipt_total
from sharding import assigned_record_ids


DEFAULT_QUESTION = (
    "What is the final total amount shown on this receipt? Return only the amount, "
    "with no currency symbol, label, or explanation."
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    required = {"run_id", "seed", "split", "datasets", "model_id", "model_revision"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"baseline config is missing: {', '.join(missing)}")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if config["split"] not in {"validation", "test"}:
        raise ValueError("baseline split must be validation or test")
    if config["split"] == "test" and not config.get("frozen_for_test", False):
        raise ValueError("test baseline requires frozen_for_test=true")
    if config["model_id"] != DEFAULT_MODEL_ID:
        raise ValueError("this first baseline implementation supports the pinned Donut model only")
    if config["model_revision"] != DEFAULT_REVISION:
        raise ValueError("the Donut revision must match the project execution contract")
    datasets = set(config["datasets"])
    if not datasets or not datasets <= {"sroie", "cord_v2"}:
        raise ValueError("datasets must contain sroie and/or cord_v2")


def _select_manifest(config: dict[str, Any], repo_root: Path):
    import pandas as pd

    manifest_path = repo_root / "data" / "processed" / "master_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    required = {
        "record_id",
        "dataset_name",
        "document_id",
        "source_split",
        "source_index",
        "experiment_split",
        "task",
        "clean_target",
        "attack_target",
        "usable",
        "attack_eligible",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"processed manifest is missing: {', '.join(missing)}")
    selected = manifest[
        manifest["dataset_name"].isin(config["datasets"])
        & manifest["experiment_split"].eq(config["split"])
        & manifest["task"].eq("receipt_total")
        & manifest["usable"].eq(True)
    ].copy()
    smoke_per_dataset = config.get("smoke_per_dataset")
    if smoke_per_dataset is not None:
        if int(smoke_per_dataset) < 1:
            raise ValueError("smoke_per_dataset must be positive")
        selected = (
            selected.sort_values("record_id")
            .groupby("dataset_name", group_keys=False)
            .head(int(smoke_per_dataset))
        )
    if selected.empty:
        raise ValueError("no usable receipt records were selected")
    if selected["record_id"].duplicated().any():
        raise ValueError("processed manifest contains duplicate selected record IDs")
    return selected.sort_values("record_id").set_index("record_id", drop=False)


def _load_rows(repo_root: Path, split: str, assigned: list[str]) -> dict[str, dict[str, Any]]:
    from datasets import load_from_disk

    wanted = set(assigned)
    rows: dict[str, dict[str, Any]] = {}
    for dataset_name in ("sroie", "cord_v2"):
        if not any(value.startswith(f"{dataset_name}/") for value in wanted):
            continue
        dataset = load_from_disk(str(repo_root / "data" / "processed" / dataset_name))[split]
        for row in dataset:
            record_id = f"{dataset_name}/{row['source_split']}/{row['document_id']}"
            if record_id in wanted:
                rows[record_id] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"processed datasets are missing {len(missing)} assigned records")
    return rows


def _merge_if_complete(
    output_root: Path,
    *,
    expected_record_ids: set[str],
    num_workers: int,
) -> bool:
    markers = [
        output_root / "shards" / f"worker_{worker_id:02d}" / "WORKER_COMPLETE.json"
        for worker_id in range(num_workers)
    ]
    if not all(path.is_file() for path in markers):
        return False

    lock_path = output_root / ".merge.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assignments: list[str] = []
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for worker_id in range(num_workers):
            assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
            assignments.extend(
                line for line in assignment_path.read_text().splitlines() if line
            )
            record_dir = output_root / "shards" / f"worker_{worker_id:02d}" / "records"
            for path in sorted(record_dir.glob("*.json")):
                record = json.loads(path.read_text())
                record_id = record["record_id"]
                if record_id in seen:
                    raise RuntimeError(f"duplicate baseline record during merge: {record_id}")
                seen.add(record_id)
                records.append(record)
        if len(assignments) != len(set(assignments)):
            raise RuntimeError("baseline worker assignments overlap")
        if set(assignments) != expected_record_ids:
            raise RuntimeError("baseline assignments do not cover the expected records")
        if seen != expected_record_ids:
            raise RuntimeError("baseline final records do not cover the expected records")

        import pandas as pd

        frame = pd.DataFrame(records).sort_values(["dataset_name", "record_id"])
        merged = output_root / "merged"
        merged.mkdir(parents=True, exist_ok=True)
        parquet_tmp = merged / "predictions.parquet.tmp"
        csv_tmp = merged / "predictions.csv.tmp"
        frame.to_parquet(parquet_tmp, index=False)
        frame.to_csv(csv_tmp, index=False)
        os.replace(parquet_tmp, merged / "predictions.parquet")
        os.replace(csv_tmp, merged / "predictions.csv")

        clean_correct = sorted(
            frame.loc[frame["clean_correct"].eq(True), "record_id"].tolist()
        )
        clean_incorrect = sorted(
            frame.loc[frame["clean_correct"].ne(True), "record_id"].tolist()
        )
        _atomic_text(
            merged / "clean_correct_ids.txt",
            "".join(f"{record_id}\n" for record_id in clean_correct),
        )
        _atomic_text(
            merged / "clean_incorrect_ids.txt",
            "".join(f"{record_id}\n" for record_id in clean_incorrect),
        )

        failures: list[dict[str, Any]] = []
        for worker_id in range(num_workers):
            path = output_root / "shards" / f"worker_{worker_id:02d}" / "failures.jsonl"
            if path.is_file():
                failures.extend(
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                )
        failure_frame = pd.DataFrame(failures)
        failures_tmp = merged / "failures.csv.tmp"
        failure_frame.to_csv(failures_tmp, index=False)
        os.replace(failures_tmp, merged / "failures.csv")

        grouped = (
            frame.groupby(["model_id", "dataset_name", "experiment_split"], dropna=False)
            .agg(
                documents=("record_id", "size"),
                clean_correct=("clean_correct", "sum"),
                parse_failures=("parse_status", lambda values: int((values == "failed").sum())),
                median_runtime_seconds=("runtime_seconds", "median"),
                peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
            )
            .reset_index()
        )
        grouped["exact_match_accuracy"] = grouped["clean_correct"] / grouped["documents"]
        _atomic_json(
            merged / "summary.json",
            {
                "records": len(records),
                "clean_correct": len(clean_correct),
                "clean_incorrect": len(clean_incorrect),
                "failures_logged": len(failures),
                "groups": json.loads(grouped.to_json(orient="records")),
                "merged_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True


def run(config, config_path, user_id, worker_id, num_workers):
    """Run one deterministic clean baseline shard (Donut by default, Qwen if configured)."""
    if config.get("model_family") == "qwen":
        from stages.baseline_qwen import run_qwen_baseline

        return run_qwen_baseline(config, config_path, user_id, worker_id, num_workers)

    _validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The Donut baseline requires a CUDA GPU; submit it through CARC Slurm.")

    repo_root = Path(config_path).resolve().parents[2]
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if config.get("require_clean_git", True) and git_status.strip():
        raise RuntimeError("refusing to run baseline from a dirty worktree")
    config_bytes = Path(config_path).read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest = _select_manifest(config, repo_root)
    assignments = assigned_record_ids(manifest.index, worker_id, num_workers)
    output_root = repo_root / "outputs" / "baseline" / config["run_id"]
    shard_root = output_root / "shards" / f"worker_{worker_id:02d}"
    assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
    expected_assignment = "".join(f"{record_id}\n" for record_id in assignments)
    if assignment_path.exists() and assignment_path.read_text() != expected_assignment:
        raise RuntimeError("existing baseline assignment differs from this run")
    _atomic_text(assignment_path, expected_assignment)

    copied_config = output_root / "config.json"
    if copied_config.exists() and _sha256_file(copied_config) != config_sha256:
        raise RuntimeError("baseline output already contains a different configuration")
    if not copied_config.exists():
        _atomic_text(copied_config, config_bytes.decode())

    rows = _load_rows(repo_root, config["split"], assignments)
    device = torch.device("cuda")
    model = None
    if assignments:
        model = DonutDocVQA(
            device=device,
            model_id=config["model_id"],
            revision=config["model_revision"],
            max_length=int(config.get("max_length", 64)),
        )

    import datasets
    import PIL
    import transformers

    environment_path = output_root / "environment.json"
    if not environment_path.exists():
        _atomic_json(
            environment_path,
            {
                "python": os.sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
                "pillow": PIL.__version__,
                "gpu": torch.cuda.get_device_name(device),
            },
        )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    question = config.get("question", DEFAULT_QUESTION)
    prompt_id = config.get("prompt_id", "receipt_total_v1")
    verify_count = int(config.get("determinism_check_per_worker", 1))
    failures: list[dict[str, str]] = []

    for index, record_id in enumerate(assignments):
        record_path = shard_root / "records" / f"{record_id.replace('/', '_')}.json"
        if record_path.is_file():
            continue
        row = rows[record_id]
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        try:
            torch.cuda.reset_peak_memory_stats(device)
            raw_response = model.predict_pil(row["image"].convert("RGB"), question)
            prediction = normalize_receipt_total(raw_response, row["dataset_name"])
            deterministic = None
            if index < verify_count:
                repeated = model.predict_pil(row["image"].convert("RGB"), question)
                repeated_prediction = normalize_receipt_total(repeated, row["dataset_name"])
                deterministic = repeated == raw_response and repeated_prediction == prediction
                if not deterministic:
                    raise RuntimeError("two consecutive clean Donut predictions differed")
            runtime = time.monotonic() - started
            record = {
                "run_id": config["run_id"],
                "stage": "baseline",
                "record_id": record_id,
                "dataset_name": row["dataset_name"],
                "document_id": row["document_id"],
                "experiment_split": config["split"],
                "source_split": row["source_split"],
                "source_index": row["source_index"],
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "task": "receipt_total",
                "prompt_id": prompt_id,
                "prompt_text": question,
                "clean_target": row["clean_target"],
                "attack_target": row["attack_target"],
                "attack_eligible": bool(row["attack_eligible"]),
                "raw_response": raw_response,
                "prediction_normalized": prediction,
                "parse_status": "parsed" if prediction is not None else "failed",
                "clean_correct": prediction == row["clean_target"],
                "determinism_checked": deterministic is not None,
                "deterministic": deterministic,
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                "processor_settings": {
                    "height": model.settings.height,
                    "width": model.settings.width,
                    "image_mean": model.settings.image_mean,
                    "image_std": model.settings.image_std,
                    "align_long_axis": model.settings.align_long_axis,
                },
                "worker_id": worker_id,
                "user_id": user_id,
                "seed": seed,
                "git_commit": git_commit,
                "config_sha256": config_sha256,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "error": None,
            }
            _atomic_json(record_path, record)
        except Exception as exc:
            failure = {"record_id": record_id, "error": repr(exc)}
            failures.append(failure)
            failure_path = shard_root / "failures.jsonl"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("a") as handle:
                handle.write(json.dumps(failure, sort_keys=True) + "\n")

        completed = len(list((shard_root / "records").glob("*.json")))
        _atomic_json(
            shard_root / "progress.json",
            {
                "completed_records": completed,
                "expected_records": len(assignments),
                "failures_this_process": len(failures),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        )

    actual = len(list((shard_root / "records").glob("*.json")))
    if actual != len(assignments):
        raise RuntimeError(
            f"baseline worker incomplete: found {actual} records, expected {len(assignments)}"
        )
    _atomic_json(
        shard_root / "WORKER_COMPLETE.json",
        {
            "run_id": config["run_id"],
            "worker_id": worker_id,
            "expected_records": len(assignments),
            "completed_records": actual,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _merge_if_complete(
        output_root,
        expected_record_ids=set(manifest.index),
        num_workers=num_workers,
    )
