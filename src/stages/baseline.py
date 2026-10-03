"""Unified, checkpointed Qwen + Donut clean baseline."""

from __future__ import annotations

import fcntl
import gc
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpoints import (
    atomic_json,
    atomic_text,
    experiment_key,
    load_json,
    record_is_final,
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from data import (
    DATASETS,
    EXPECTED_COUNTS,
    RECEIPT_DATASETS,
    load_processed_rows,
    row_images,
    select_baseline_manifest,
)
from models.donut import DEFAULT_MODEL_ID as DONUT_MODEL_ID
from models.donut import DEFAULT_REVISION as DONUT_REVISION
from models.qwen import DEFAULT_MODEL_ID as QWEN_MODEL_ID
from models.qwen import DEFAULT_REVISION as QWEN_REVISION
from normalization import normalize_receipt_total, normalize_resume_degree
from sharding import assigned_record_ids
from team import get_team_member


PROMPTS = {
    "receipt_total": {
        "id": "receipt_total_v1",
        "text": (
            "What is the final total amount shown on this receipt? Return only the "
            "amount, with no currency symbol, label, or explanation."
        ),
    },
    "resume_highest_degree": {
        "id": "resume_degree_v1",
        "text": (
            "What is the highest degree level stated in this resume? Return exactly "
            "one label from: secondary, certificate_or_diploma, associate, bachelor, "
            "postgraduate, master, doctorate."
        ),
    },
}

MODEL_CONTRACT = {
    "qwen": (QWEN_MODEL_ID, QWEN_REVISION),
    "donut": (DONUT_MODEL_ID, DONUT_REVISION),
}


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "run_id", "stage", "split", "seed", "output_root", "datasets",
        "models", "prompts",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"baseline config is missing: {', '.join(missing)}")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be a non-empty string")
    if config["stage"] != "baseline":
        raise ValueError("stage must be 'baseline'")
    if config["output_root"] != "outputs/baseline":
        raise ValueError("output_root must be 'outputs/baseline'")
    if config["split"] not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if config["split"] == "test":
        if not config.get("frozen_for_test", False):
            raise ValueError("test execution requires frozen_for_test=true")
        if not config.get("validation_run_id"):
            raise ValueError("test execution requires validation_run_id")
    if set(config["datasets"]) != set(DATASETS) or len(config["datasets"]) != len(DATASETS):
        raise ValueError(f"datasets must contain exactly: {', '.join(DATASETS)}")
    if set(config["models"]) != set(MODEL_CONTRACT):
        raise ValueError("models must contain exactly qwen and donut")
    for model_key, (model_id, revision) in MODEL_CONTRACT.items():
        model = config["models"][model_key]
        if model.get("model_id") != model_id or model.get("revision") != revision:
            raise ValueError(f"{model_key} model ID/revision does not match the pinned contract")
    if config["prompts"] != PROMPTS:
        raise ValueError("prompt IDs and text must exactly match the frozen baseline prompts")
    if config.get("smoke_per_dataset") is not None and config["split"] != "validation":
        raise ValueError("smoke_per_dataset is allowed only on validation")


def build_experiments(manifest, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the exact expected experiment set before loading a model."""
    experiments: dict[str, dict[str, Any]] = {}
    for record_id, row in manifest.iterrows():
        dataset_name = row["dataset_name"]
        model_keys = ("qwen", "donut") if dataset_name in RECEIPT_DATASETS else ("qwen",)
        task = row["task"]
        prompt = config["prompts"][task]
        for model_key in model_keys:
            model = config["models"][model_key]
            key = experiment_key(
                record_id, model["model_id"], task, prompt["id"], config["split"]
            )
            if key in experiments:
                raise ValueError(f"duplicate expected experiment key: {key}")
            experiments[key] = {
                "experiment_key": key,
                "record_id": record_id,
                "dataset_name": dataset_name,
                "experiment_split": config["split"],
                "model_key": model_key,
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "task": task,
                "prompt_id": prompt["id"],
                "prompt_text": prompt["text"],
                "prompt_sha256": sha256_text(prompt["text"]),
            }
    expected_count = (
        2 * (
            EXPECTED_COUNTS[config["split"]]["sroie"]
            + EXPECTED_COUNTS[config["split"]]["cord_v2"]
        )
        + EXPECTED_COUNTS[config["split"]]["resume_parsing_vision"]
    )
    if config.get("smoke_per_dataset") is not None:
        expected_count = 5 * int(config["smoke_per_dataset"])
    if len(experiments) != expected_count:
        raise ValueError(f"built {len(experiments)} experiments; expected {expected_count}")
    return experiments


def _normalize(task: str, dataset_name: str, raw_response: str | None) -> str | None:
    if task == "receipt_total":
        return normalize_receipt_total(raw_response, dataset_name)
    if task == "resume_highest_degree":
        return normalize_resume_degree(raw_response)
    raise ValueError(f"unsupported task: {task}")


def _make_record(
    *,
    config: dict[str, Any],
    config_sha256: str,
    expected: dict[str, Any],
    manifest_row: dict[str, Any],
    raw_response: str | None,
    runtime_seconds: float,
    peak_gpu_memory_mb: float | None,
    page_count: int,
    worker_id: int,
    user_id: str,
    git_commit: str,
    processor_settings: dict[str, Any],
    input_info: dict[str, Any] | None,
    deterministic: bool | None,
    started_at: str,
    error: str | None,
) -> dict[str, Any]:
    prediction = None if error else _normalize(
        expected["task"], expected["dataset_name"], raw_response
    )
    target = _scalar(manifest_row["clean_target"])
    return {
        "run_id": config["run_id"],
        "stage": "baseline",
        **expected,
        "source_split": _scalar(manifest_row.get("source_split")),
        "source_index": _scalar(manifest_row.get("source_index")),
        "document_id": _scalar(manifest_row.get("document_id")),
        "clean_target": target,
        "attack_target": _scalar(manifest_row.get("attack_target")),
        "attack_eligible": bool(_scalar(manifest_row.get("attack_eligible"))),
        "raw_response": raw_response,
        "prediction_normalized": prediction,
        "parse_status": "ok" if prediction is not None else "failed",
        "clean_correct": prediction is not None and prediction == str(target),
        "runtime_seconds": round(float(runtime_seconds), 4),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "worker_id": worker_id,
        "user_id": user_id,
        "seed": int(config["seed"]),
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "input_page_count": page_count,
        "processor_settings": processor_settings,
        "input_info": input_info,
        "deterministic": deterministic,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if error else "complete",
        "error": error,
    }


def _record_path(shard_root: Path, key: str) -> Path:
    return shard_root / "records" / f"{key}.json"


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _append_failure(shard_root: Path, record: dict[str, Any]) -> None:
    path = shard_root / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "experiment_key": record["experiment_key"],
            "record_id": record["record_id"],
            "model_key": record["model_key"],
            "error": record["error"],
        }, sort_keys=True) + "\n")


def _write_progress(
    shard_root: Path, expected: list[dict[str, Any]], config_sha256: str
) -> None:
    final = sum(
        record_is_final(
            _record_path(shard_root, item["experiment_key"]),
            expected=item,
            config_sha256=config_sha256,
        )
        for item in expected
    )
    atomic_json(shard_root / "progress.json", {
        "final_records": final,
        "expected_records": len(expected),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    })


def _run_qwen(
    *, pending, rows, manifest, config, config_sha256, shard_root,
    worker_id, user_id, git_commit, device,
) -> None:
    import torch
    from models.qwen import QwenVL

    if not pending:
        return
    settings = config["models"]["qwen"]
    model = QwenVL(
        device=device,
        model_id=settings["model_id"],
        revision=settings["revision"],
        precision=settings.get("precision", "bfloat16"),
        min_pixels=int(settings.get("min_pixels", 256 * 28 * 28)),
        max_pixels=int(settings.get("max_pixels", 1280 * 28 * 28)),
        max_new_tokens=int(settings.get("max_new_tokens", 32)),
    )
    verify_count = int(config.get("determinism_check_per_model_per_worker", 1))
    abort_after_empty = int(config.get("abort_after_consecutive_empty", 5))
    consecutive_empty = 0
    try:
        for index, expected in enumerate(pending):
            images = row_images(rows[expected["record_id"]])
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            raw_response = None
            error = None
            deterministic = None
            input_info = None
            torch.cuda.reset_peak_memory_stats(device)
            try:
                raw_response = model.predict(images, expected["prompt_text"])
                input_info = dict(model.last_input_info)
                if index < verify_count:
                    repeated = model.predict(images, expected["prompt_text"])
                    deterministic = repeated == raw_response
                    if not deterministic:
                        raise RuntimeError("repeated Qwen response differed")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                torch.cuda.empty_cache()
            record = _make_record(
                config=config,
                config_sha256=config_sha256,
                expected=expected,
                manifest_row=manifest.loc[expected["record_id"]].to_dict(),
                raw_response=raw_response,
                runtime_seconds=time.monotonic() - started,
                peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                page_count=len(images),
                worker_id=worker_id,
                user_id=user_id,
                git_commit=git_commit,
                processor_settings=model.settings,
                input_info=input_info,
                deterministic=deterministic,
                started_at=started_at,
                error=error,
            )
            atomic_json(_record_path(shard_root, expected["experiment_key"]), record)
            if error:
                _append_failure(shard_root, record)
            consecutive_empty = (
                consecutive_empty + 1
                if not error and not (raw_response or "").strip()
                else 0
            )
            if consecutive_empty >= abort_after_empty:
                raise RuntimeError(f"{consecutive_empty} consecutive empty Qwen responses")
            _write_progress(shard_root, pending, config_sha256)
    finally:
        model.unload()


def _run_donut(
    *, pending, rows, manifest, config, config_sha256, shard_root,
    worker_id, user_id, git_commit, device,
) -> None:
    import torch
    from models.donut import DonutDocVQA

    if not pending:
        return
    settings = config["models"]["donut"]
    model = DonutDocVQA(
        device=device,
        model_id=settings["model_id"],
        revision=settings["revision"],
        precision=settings.get("precision", "float32"),
        max_length=int(settings.get("max_length", 64)),
    )
    verify_count = int(config.get("determinism_check_per_model_per_worker", 1))
    try:
        for index, expected in enumerate(pending):
            image = row_images(rows[expected["record_id"]])[0]
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            raw_response = None
            error = None
            deterministic = None
            torch.cuda.reset_peak_memory_stats(device)
            try:
                raw_response = model.predict_pil(image, expected["prompt_text"])
                if index < verify_count:
                    repeated = model.predict_pil(image, expected["prompt_text"])
                    deterministic = repeated == raw_response
                    if not deterministic:
                        raise RuntimeError("repeated Donut response differed")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                torch.cuda.empty_cache()
            model_settings = {
                "precision": settings.get("precision", "float32"),
                "height": model.settings.height,
                "width": model.settings.width,
                "image_mean": model.settings.image_mean,
                "image_std": model.settings.image_std,
                "align_long_axis": model.settings.align_long_axis,
                "max_length": model.max_length,
            }
            record = _make_record(
                config=config,
                config_sha256=config_sha256,
                expected=expected,
                manifest_row=manifest.loc[expected["record_id"]].to_dict(),
                raw_response=raw_response,
                runtime_seconds=time.monotonic() - started,
                peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                page_count=1,
                worker_id=worker_id,
                user_id=user_id,
                git_commit=git_commit,
                processor_settings=model_settings,
                input_info={"image_size": list(image.size)},
                deterministic=deterministic,
                started_at=started_at,
                error=error,
            )
            atomic_json(_record_path(shard_root, expected["experiment_key"]), record)
            if error:
                _append_failure(shard_root, record)
            _write_progress(shard_root, pending, config_sha256)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _load_plan(config: dict[str, Any], repo_root: Path):
    import pandas as pd

    manifest_path = repo_root / "data" / "processed" / "master_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"processed manifest not found: {manifest_path}")
    manifest = select_baseline_manifest(pd.read_parquet(manifest_path), config)
    return manifest, build_experiments(manifest, config)


def run(config, config_path, user_id, worker_id, num_workers):
    """Run one deterministic document shard; smoke may use one owner worker."""
    import datasets
    import PIL
    import torch
    import transformers

    validate_config(config)
    member = get_team_member(user_id)
    if member.worker_id != worker_id:
        raise ValueError(f"{user_id} must use worker {member.worker_id}, not {worker_id}")
    allowed_workers = {5}
    if config.get("smoke_per_dataset") is not None:
        allowed_workers.add(1)
    if num_workers not in allowed_workers:
        raise ValueError("baseline requires five workers; owner smoke may use one")
    if not torch.cuda.is_available():
        raise RuntimeError("baseline inference requires a CUDA GPU on CARC")

    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[2]
    if config.get("require_clean_git", True):
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        )
        if status.strip():
            raise RuntimeError("refusing to run baseline from a dirty worktree")
    git_commit = _git_commit(repo_root)
    config_bytes = config_path.read_bytes()
    config_sha256 = sha256_bytes(config_bytes)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest, experiments = _load_plan(config, repo_root)
    assigned_documents = assigned_record_ids(manifest.index, worker_id, num_workers)
    assigned_set = set(assigned_documents)
    expected = [
        value for value in experiments.values() if value["record_id"] in assigned_set
    ]
    expected = [{**value, "git_commit": git_commit} for value in expected]
    expected.sort(key=lambda value: (value["model_key"], value["record_id"]))

    output_root = repo_root / config["output_root"] / config["run_id"]
    shard_root = output_root / "shards" / f"worker_{worker_id:02d}"
    assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
    assignment_text = "".join(f"{value}\n" for value in assigned_documents)
    if assignment_path.exists() and assignment_path.read_text() != assignment_text:
        raise RuntimeError("existing assignment differs from this run")
    atomic_text(assignment_path, assignment_text)

    copied_config = output_root / "config.json"
    if copied_config.exists() and sha256_file(copied_config) != config_sha256:
        raise RuntimeError("run directory contains a different config")
    if not copied_config.exists():
        atomic_text(copied_config, config_bytes.decode("utf-8"))

    device = torch.device("cuda")
    atomic_json(shard_root / "environment.json", {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "pillow": PIL.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "git_commit": git_commit,
    })

    pending = [
        item for item in expected
        if not record_is_final(
            _record_path(shard_root, item["experiment_key"]),
            expected=item,
            config_sha256=config_sha256,
        )
    ]
    if pending:
        rows = load_processed_rows(
            repo_root, config["split"], {item["record_id"] for item in pending}
        )
        common = {
            "rows": rows,
            "manifest": manifest,
            "config": config,
            "config_sha256": config_sha256,
            "shard_root": shard_root,
            "worker_id": worker_id,
            "user_id": user_id,
            "git_commit": git_commit,
            "device": device,
        }
        _run_qwen(
            pending=[item for item in pending if item["model_key"] == "qwen"],
            **common,
        )
        _run_donut(
            pending=[item for item in pending if item["model_key"] == "donut"],
            **common,
        )

    unfinished = [
        item["experiment_key"] for item in expected
        if not record_is_final(
            _record_path(shard_root, item["experiment_key"]),
            expected=item,
            config_sha256=config_sha256,
        )
    ]
    if unfinished:
        raise RuntimeError(f"{len(unfinished)} expected records are missing or invalid")
    records = [
        load_json(_record_path(shard_root, item["experiment_key"])) for item in expected
    ]
    atomic_json(shard_root / "WORKER_COMPLETE.json", {
        "run_id": config["run_id"],
        "stage": "baseline",
        "worker_id": worker_id,
        "user_id": user_id,
        "expected_documents": len(assigned_documents),
        "expected_records": len(expected),
        "failed_records": sum(record["status"] == "failed" for record in records),
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    merge_if_complete(config, config_path, num_workers=num_workers)


def merge_if_complete(
    config: dict[str, Any], config_path: str | Path, *, num_workers: int = 5
) -> bool:
    """CPU-safe exact merge. Return False while another shard is unfinished."""
    import pandas as pd

    validate_config(config)
    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[2]
    config_sha256 = sha256_bytes(config_path.read_bytes())
    manifest, experiments = _load_plan(config, repo_root)
    output_root = repo_root / config["output_root"] / config["run_id"]
    if not output_root.exists():
        return False
    copied_config = output_root / "config.json"
    if not copied_config.is_file() or sha256_file(copied_config) != config_sha256:
        raise RuntimeError("run directory is missing the exact submitted config")
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
        records: dict[str, dict[str, Any]] = {}
        git_commits: set[str] = set()
        for worker_id, marker_path in enumerate(markers):
            marker = load_json(marker_path)
            if not marker:
                raise RuntimeError(f"invalid marker: {marker_path}")
            if (
                marker.get("run_id") != config["run_id"]
                or marker.get("worker_id") != worker_id
                or marker.get("config_sha256") != config_sha256
                or not marker.get("git_commit")
            ):
                raise RuntimeError(f"worker {worker_id} marker provenance mismatch")
            git_commits.add(marker.get("git_commit"))

            expected_documents = assigned_record_ids(manifest.index, worker_id, num_workers)
            assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
            assignment = assignment_path.read_text().splitlines() if assignment_path.exists() else []
            if assignment != expected_documents:
                raise RuntimeError(f"worker {worker_id} assignment is missing or incorrect")
            assigned_set = set(expected_documents)
            worker_expected = {
                key: value for key, value in experiments.items()
                if value["record_id"] in assigned_set
            }
            if marker.get("expected_records") != len(worker_expected):
                raise RuntimeError(f"worker {worker_id} marker count is incorrect")
            record_dir = output_root / "shards" / f"worker_{worker_id:02d}" / "records"
            actual_paths = {path.stem: path for path in record_dir.glob("*.json")}
            if set(actual_paths) != set(worker_expected):
                raise RuntimeError(f"worker {worker_id} record keys do not match its assignment")
            for key, expected in worker_expected.items():
                path = actual_paths[key]
                if not record_is_final(path, expected=expected, config_sha256=config_sha256):
                    raise RuntimeError(f"invalid record during merge: {path}")
                record = load_json(path)
                if record["worker_id"] != worker_id:
                    raise RuntimeError(f"wrong worker in record: {path}")
                if record["git_commit"] != marker["git_commit"]:
                    raise RuntimeError(f"Git commit mismatch in record: {path}")
                if key in records:
                    raise RuntimeError(f"duplicate experiment key: {key}")
                records[key] = record
        if len(git_commits) != 1:
            raise RuntimeError("workers did not run the same Git commit")
        if set(records) != set(experiments):
            raise RuntimeError("merged experiment keys do not match the exact expected set")

        fingerprint = sha256_text("".join(
            f"{key}:{sha256_text(json.dumps(records[key], sort_keys=True))}\n"
            for key in sorted(records)
        ))
        merged = output_root / "merged"
        merge_manifest = merged / "merge_manifest.json"
        previous = load_json(merge_manifest)
        if previous:
            if (
                previous.get("records_fingerprint") == fingerprint
                and previous.get("config_sha256") == config_sha256
            ):
                return True
            raise RuntimeError("merged outputs already exist for different inputs")
        if merged.exists() and any(merged.iterdir()):
            raise RuntimeError("merged directory exists without matching merge provenance")
        merged.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame([records[key] for key in sorted(records)]).sort_values(
            ["model_key", "dataset_name", "record_id"]
        )
        parquet_tmp = merged / "predictions.parquet.tmp"
        csv_tmp = merged / "predictions.csv.tmp"
        frame.to_parquet(parquet_tmp, index=False)
        frame.to_csv(csv_tmp, index=False)
        os.replace(parquet_tmp, merged / "predictions.parquet")
        os.replace(csv_tmp, merged / "predictions.csv")

        for model_key in ("qwen", "donut"):
            model_frame = frame[frame["model_key"].eq(model_key)]
            for correct, label in ((True, "correct"), (False, "incorrect")):
                ids = sorted(model_frame.loc[
                    model_frame["clean_correct"].eq(correct), "record_id"
                ].tolist())
                atomic_text(
                    merged / f"{model_key}_clean_{label}_ids.txt",
                    "".join(f"{record_id}\n" for record_id in ids),
                )
        failures = frame[
            frame["status"].eq("failed") | frame["parse_status"].eq("failed")
        ]
        failures_tmp = merged / "failures.csv.tmp"
        failures.to_csv(failures_tmp, index=False)
        os.replace(failures_tmp, merged / "failures.csv")

        grouped = frame.groupby(
            ["model_key", "dataset_name", "task", "experiment_split"]
        ).agg(
            documents=("record_id", "size"),
            clean_correct=("clean_correct", "sum"),
            parse_failures=("parse_status", lambda values: int((values == "failed").sum())),
            inference_failures=("status", lambda values: int((values == "failed").sum())),
            median_runtime_seconds=("runtime_seconds", "median"),
            peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
        ).reset_index()
        grouped["exact_match_accuracy"] = grouped["clean_correct"] / grouped["documents"]
        atomic_json(merged / "summary.json", {
            "records": len(frame),
            "expected_records": len(experiments),
            "inference_failures": int(frame["status"].eq("failed").sum()),
            "parse_failures": int(frame["parse_status"].eq("failed").sum()),
            "groups": json.loads(grouped.to_json(orient="records")),
            "merged_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(merge_manifest, {
            "config_sha256": config_sha256,
            "git_commit": next(iter(git_commits)),
            "records": len(records),
            "records_fingerprint": fingerprint,
        })
        return True
