"""Checkpointed clean Qwen2.5-VL baseline for SROIE, CORD, and resumes.

Called from ``stages.baseline.run`` when ``config["model_family"] == "qwen"``.
One JSON record is written atomically per (document, model, task, prompt); valid
completed records are skipped on restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.qwen import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
)
from normalization import normalize_receipt_total, normalize_resume_degree
from sharding import assigned_record_ids


MODEL_TAG = "qwen2_5_vl_3b"

PROMPTS: dict[str, str] = {
    "receipt_total_v1": (
        "What is the final total amount shown on this receipt? Return only the "
        "amount, with no currency symbol, label, or explanation."
    ),
    "resume_degree_v1": (
        "What is the highest degree level stated in this resume? Return exactly "
        "one label from: secondary, certificate_or_diploma, associate, bachelor, "
        "postgraduate, master, doctorate."
    ),
}

REQUIRED_RECORD_FIELDS = (
    "run_id", "record_id", "dataset_name", "experiment_split", "model_id",
    "model_revision", "task", "prompt_id", "clean_target", "attack_target",
    "attack_eligible", "raw_response", "prediction_normalized", "parse_status",
    "clean_correct", "runtime_seconds", "peak_gpu_memory_mb", "worker_id", "seed",
    "git_commit", "config_sha256", "input_page_count", "status", "error",
)
VALID_STATUSES = {"complete", "failed"}


# --------------------------------------------------------------------------- #
# Small pure helpers (unit-tested on CPU)
# --------------------------------------------------------------------------- #
def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prompt_sha256(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def experiment_key(record_id: str, task: str, prompt_id: str, split: str) -> str:
    """Filesystem-safe unique key: record + model + task + prompt + split."""
    raw = f"{record_id}__{MODEL_TAG}__{task}__{prompt_id}__{split}"
    return raw.replace("/", "_")


def validate_config(config: dict[str, Any]) -> None:
    required = {"run_id", "seed", "split", "model_id", "model_revision", "jobs"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"qwen baseline config is missing: {', '.join(missing)}")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if config["split"] not in {"validation", "test"}:
        raise ValueError("baseline split must be validation or test")
    if config["split"] == "test" and not config.get("frozen_for_test", False):
        raise ValueError("test baseline requires frozen_for_test=true")
    if config["model_id"] != DEFAULT_MODEL_ID:
        raise ValueError("model_id must be the pinned Qwen2.5-VL-3B-Instruct")
    if config["model_revision"] != DEFAULT_REVISION:
        raise ValueError("Qwen revision must match the project execution contract")
    if not config["jobs"]:
        raise ValueError("config must define at least one job")
    seen: set[tuple[str, str]] = set()
    for job in config["jobs"]:
        for field in ("task", "datasets", "prompt_id"):
            if field not in job:
                raise ValueError(f"job is missing '{field}': {job}")
        if job["task"] not in {"receipt_total", "resume_degree"}:
            raise ValueError(f"unsupported task: {job['task']}")
        if job["prompt_id"] not in PROMPTS and "prompt_text" not in job:
            raise ValueError(
                f"prompt_id {job['prompt_id']!r} is new; provide 'prompt_text' in the job"
            )
        if job["prompt_id"] in PROMPTS and "prompt_text" in job:
            if job["prompt_text"] != PROMPTS[job["prompt_id"]]:
                raise ValueError(
                    f"prompt_id {job['prompt_id']!r} already exists with different "
                    "text; create a new prompt ID instead of editing it"
                )
        marker = (job["task"], job["prompt_id"])
        if marker in seen:
            raise ValueError(f"duplicate job for task/prompt: {marker}")
        seen.add(marker)


def job_prompt_text(job: dict[str, Any]) -> str:
    return job.get("prompt_text", PROMPTS.get(job["prompt_id"], ""))


def normalize_prediction(task: str, raw_response: str | None, dataset_name: str) -> str | None:
    if task == "receipt_total":
        return normalize_receipt_total(raw_response, dataset_name)
    if task == "resume_degree":
        return normalize_resume_degree(raw_response)
    raise ValueError(f"unsupported task: {task}")


def is_clean_correct(prediction: str | None, clean_target: Any) -> bool:
    """True only when parsing succeeded and the value exactly matches the target."""
    return prediction is not None and prediction == str(clean_target)


def _scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars (and NaN) into JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def build_record(
    *,
    config: dict[str, Any],
    config_sha256: str,
    job: dict[str, Any],
    manifest_row: dict[str, Any],
    raw_response: str | None,
    runtime_seconds: float,
    peak_gpu_memory_mb: float | None,
    page_count: int,
    worker_id: int,
    git_commit: str,
    settings: dict[str, Any],
    started_at: str,
    ended_at: str,
    error: str | None = None,
    deterministic: bool | None = None,
    input_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_name = manifest_row["dataset_name"]
    failed = error is not None
    prediction = None if failed else normalize_prediction(job["task"], raw_response, dataset_name)
    clean_target = _scalar(manifest_row["clean_target"])
    prompt_text = job_prompt_text(job)
    return {
        "run_id": config["run_id"],
        "stage": "baseline",
        "record_id": manifest_row["record_id"],
        "dataset_name": dataset_name,
        "experiment_split": config["split"],
        "source_split": _scalar(manifest_row.get("source_split")),
        "source_index": _scalar(manifest_row.get("source_index")),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "task": job["task"],
        "prompt_id": job["prompt_id"],
        "prompt_sha256": prompt_sha256(prompt_text),
        "clean_target": clean_target,
        "attack_target": _scalar(manifest_row.get("attack_target")),
        "attack_eligible": bool(_scalar(manifest_row.get("attack_eligible"))),
        "raw_response": raw_response,
        "prediction_normalized": prediction,
        "parse_status": "ok" if prediction is not None else "failed",
        "clean_correct": is_clean_correct(prediction, clean_target),
        "runtime_seconds": round(float(runtime_seconds), 4),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "worker_id": worker_id,
        "seed": int(config["seed"]),
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "input_page_count": page_count,
        "processor_settings": settings,
        "input_info": input_info,
        "deterministic_check": deterministic,
        "started_at": started_at,
        "ended_at": ended_at,
        "status": "failed" if failed else "complete",
        "error": error,
    }


def record_is_valid(path: Path, config_sha256: str) -> bool:
    """A record is skippable only if it parses, matches the config, and is complete."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if any(field not in record for field in REQUIRED_RECORD_FIELDS):
        return False
    return record["status"] == "complete" and record["config_sha256"] == config_sha256


def record_is_final(path: Path, config_sha256: str) -> bool:
    """Complete OR explicit failed record for the current config (for the marker)."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        all(field in record for field in REQUIRED_RECORD_FIELDS)
        and record["status"] in VALID_STATUSES
        and record["config_sha256"] == config_sha256
    )


# --------------------------------------------------------------------------- #
# Data selection / loading
# --------------------------------------------------------------------------- #
def select_job_manifest(manifest, job: dict[str, Any], config: dict[str, Any]):
    required = {
        "record_id", "dataset_name", "source_split", "source_index",
        "experiment_split", "task", "clean_target", "attack_target",
        "usable", "attack_eligible",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"processed manifest is missing: {', '.join(missing)}")
    selected = manifest[
        manifest["dataset_name"].isin(job["datasets"])
        & manifest["experiment_split"].eq(config["split"])
        & manifest["task"].eq(job["task"])
        & manifest["usable"].eq(True)
    ].copy()
    smoke = config.get("smoke_per_dataset")
    if smoke is not None:
        if int(smoke) < 1:
            raise ValueError("smoke_per_dataset must be positive")
        selected = (
            selected.sort_values("record_id")
            .groupby("dataset_name", group_keys=False)
            .head(int(smoke))
        )
    if selected.empty:
        raise ValueError(f"no usable records for job {job['task']}/{job['prompt_id']}")
    if selected["record_id"].duplicated().any():
        raise ValueError("manifest contains duplicate selected record IDs")
    return selected.sort_values("record_id").set_index("record_id", drop=False)


def load_rows(repo_root: Path, split: str, assigned: list[str]) -> dict[str, dict[str, Any]]:
    """Load only the assigned documents (images) from the processed datasets."""
    from datasets import load_from_disk

    wanted = set(assigned)
    rows: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({value.split("/", 1)[0] for value in wanted}):
        dataset = load_from_disk(str(repo_root / "data" / "processed" / dataset_name))[split]
        for row in dataset:
            record_id = f"{dataset_name}/{row['source_split']}/{row['document_id']}"
            if record_id in wanted:
                rows[record_id] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"processed datasets are missing {len(missing)} assigned records")
    return rows


def row_images(row: dict[str, Any]) -> list[Any]:
    """Receipts have one `image`; resumes have ordered `images` (all pages)."""
    images = row.get("images")
    if images:
        return list(images)
    image = row.get("image")
    if image is None:
        raise ValueError("row has neither 'image' nor 'images'")
    return [image]


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
def run_qwen_baseline(
    config: dict[str, Any],
    config_path: str,
    user_id: str,
    worker_id: int,
    num_workers: int,
) -> None:
    import pandas as pd
    import torch

    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen baseline requires a CUDA GPU; submit through CARC Slurm.")

    repo_root = Path(config_path).resolve().parents[2]
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    if config.get("require_clean_git", True) and git_status.strip():
        raise RuntimeError("refusing to run baseline from a dirty worktree")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()

    config_bytes = Path(config_path).read_bytes()
    config_sha256 = _sha256_bytes(config_bytes)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest_path = repo_root / "data" / "processed" / "master_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)

    # 1. Deterministic per-job assignments, saved BEFORE any model loads.
    output_root = repo_root / "outputs" / "baseline" / config["run_id"]
    shard_root = output_root / "shards" / f"worker_{worker_id:02d}"
    job_manifests, job_assigned = [], []
    for job in config["jobs"]:
        job_manifest = select_job_manifest(manifest, job, config)
        job_manifests.append(job_manifest)
        job_assigned.append(assigned_record_ids(job_manifest.index, worker_id, num_workers))

    union = sorted({rid for assigned in job_assigned for rid in assigned})
    assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
    expected_assignment = "".join(f"{rid}\n" for rid in union)
    if assignment_path.exists() and assignment_path.read_text() != expected_assignment:
        raise RuntimeError("existing baseline assignment differs from this run")
    _atomic_text(assignment_path, expected_assignment)

    copied_config = output_root / "config.json"
    if copied_config.exists() and _sha256_bytes(copied_config.read_bytes()) != config_sha256:
        raise RuntimeError("baseline output already contains a different configuration")
    if not copied_config.exists():
        _atomic_text(copied_config, config_bytes.decode())

    # 2. Expected experiment keys for this worker (used for the completion marker).
    expected: dict[str, Path] = {}
    for job, assigned in zip(config["jobs"], job_assigned):
        for rid in assigned:
            key = experiment_key(rid, job["task"], job["prompt_id"], config["split"])
            expected[key] = shard_root / "records" / f"{key}.json"

    pending_work = [
        (job, rid)
        for job, assigned in zip(config["jobs"], job_assigned)
        for rid in assigned
        if not record_is_valid(
            expected[experiment_key(rid, job["task"], job["prompt_id"], config["split"])],
            config_sha256,
        )
    ]

    device = torch.device("cuda")
    model = None
    if pending_work:
        from models.qwen import QwenVL

        model = QwenVL(
            device=device,
            model_id=config["model_id"],
            revision=config["model_revision"],
            min_pixels=int(config.get("min_pixels", DEFAULT_MIN_PIXELS)),
            max_pixels=int(config.get("max_pixels", DEFAULT_MAX_PIXELS)),
            max_new_tokens=int(config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
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

    # 3. Inference, one atomic record per experiment key.
    verify_count = int(config.get("determinism_check_per_worker", 1))
    abort_after_empty = int(config.get("abort_after_consecutive_empty", 5))
    failures_path = shard_root / "failures.jsonl"
    consecutive_empty = 0

    for job, job_manifest, assigned in zip(config["jobs"], job_manifests, job_assigned):
        todo = [
            rid for rid in assigned
            if not record_is_valid(
                expected[experiment_key(rid, job["task"], job["prompt_id"], config["split"])],
                config_sha256,
            )
        ]
        if not todo:
            continue
        rows = load_rows(repo_root, config["split"], todo)
        prompt_text = job_prompt_text(job)

        for index, rid in enumerate(todo):
            key = experiment_key(rid, job["task"], job["prompt_id"], config["split"])
            manifest_row = job_manifest.loc[rid].to_dict()
            images = row_images(rows[rid])
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            raw_response, error, deterministic = None, None, None
            try:
                torch.cuda.reset_peak_memory_stats(device)
                raw_response = model.predict(images, prompt_text)
                if index < verify_count:
                    repeated = model.predict(images, prompt_text)
                    deterministic = repeated == raw_response
                    if not deterministic:
                        raise RuntimeError(f"non-deterministic Qwen output for {rid}")
            except Exception as exc:  # recorded explicitly, never silently dropped
                error = f"{type(exc).__name__}: {exc}"
                torch.cuda.empty_cache()
                with failures_path.open("a") as handle:
                    handle.write(json.dumps({"record_id": rid, "key": key, "error": error}) + "\n")

            record = build_record(
                config=config,
                config_sha256=config_sha256,
                job=job,
                manifest_row=manifest_row,
                raw_response=raw_response,
                runtime_seconds=time.monotonic() - started,
                peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                page_count=len(images),
                worker_id=worker_id,
                git_commit=git_commit,
                settings=model.settings,
                started_at=started_at,
                ended_at=datetime.now(timezone.utc).isoformat(),
                error=error,
                deterministic=deterministic,
                input_info=dict(model.last_input_info) if error is None else None,
            )
            _atomic_json(expected[key], record)

            consecutive_empty = (
                consecutive_empty + 1 if error is None and not (raw_response or "").strip() else 0
            )
            if consecutive_empty >= abort_after_empty:
                raise RuntimeError(
                    f"{consecutive_empty} consecutive empty Qwen responses; stopping"
                )

    if model is not None:
        model.unload()

    # 4. Completion marker only after every expected key has a final record.
    unfinished = [key for key, path in expected.items() if not record_is_final(path, config_sha256)]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} expected records are missing/invalid; not marking complete")
    failed = sum(
        json.loads(path.read_text())["status"] == "failed" for path in expected.values()
    )
    _atomic_json(
        shard_root / "WORKER_COMPLETE.json",
        {
            "run_id": config["run_id"],
            "stage": "baseline",
            "model": MODEL_TAG,
            "worker_id": worker_id,
            "user_id": user_id,
            "expected_records": len(expected),
            "failed_records": failed,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
