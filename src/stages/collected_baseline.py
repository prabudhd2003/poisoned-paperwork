"""Clean Qwen and Donut inference on the frozen 100-receipt external set."""

from __future__ import annotations

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpoints import atomic_json, sha256_text
from experiment_utils import (
    initialize_run, load_collected_image, load_collected_manifest,
    merge_json_records, repo_root_from_config, write_worker_complete,
)
from normalization import normalize_receipt_total


PROMPT = (
    "What is the final total amount shown on this receipt? Return only the "
    "amount, with no currency symbol, label, or explanation."
)


def validate_config(config: dict[str, Any]) -> None:
    required = {"stage", "track", "run_id", "ready_for_production", "seed", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"collected baseline config is missing: {missing}")
    if config["stage"] != "baseline" or config["track"] != "collected_receipts":
        raise ValueError("collected baseline requires stage=baseline, track=collected_receipts")
    if config.get("split") != "external_test":
        raise ValueError("collected receipts must remain split=external_test")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("collected baseline is gated until all 100 labels and hashes are frozen")
    if set(config["models"]) != {"qwen", "donut"}:
        raise ValueError("collected baseline requires exactly Qwen and Donut")


def _key(record_id: str, model_key: str) -> str:
    return sha256_text(f"collected-baseline|{model_key}|{record_id}")


def _record(
    *, config, config_hash, commit, row, model_key, model, image,
    worker_id, user_id, device,
):
    import torch

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    error = None
    raw = None
    try:
        if model_key == "qwen":
            raw = model.predict([image], PROMPT)
            settings = model.settings
        else:
            raw = model.predict_pil(image, PROMPT)
            settings = {
                "precision": str(model.dtype).replace("torch.", ""),
                "height": model.settings.height, "width": model.settings.width,
                "max_length": model.max_length,
            }
    except Exception as exc:  # retained as an explicit denominator
        error = f"{type(exc).__name__}: {exc}"
        settings = {}
    normalized = None if error else normalize_receipt_total(raw, "collected_receipts")
    target = str(row["clean_target"])
    record_id = str(row["record_id"])
    return {
        "experiment_key": _key(record_id, model_key),
        "run_id": config["run_id"], "stage": "baseline",
        "track": "collected_receipts", "record_id": record_id,
        "dataset_name": "collected_receipts", "experiment_split": "external_test",
        "model_key": model_key, "model_id": model.model_id,
        "model_revision": model.revision, "prompt_id": "receipt_total_v1",
        "prompt_text": PROMPT, "clean_target": target,
        "attack_target": str(row["attack_target"]), "raw_response": raw,
        "prediction_normalized": normalized,
        "parse_status": "ok" if normalized is not None else "failed",
        "clean_correct": normalized == target, "status": "failed" if error else "complete",
        "error": error, "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "processor_settings": settings, "worker_id": worker_id, "user_id": user_id,
        "seed": int(config["seed"]), "config_sha256": config_hash,
        "git_commit": commit, "processed_sha256": str(row["processed_sha256"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def preflight(config: dict[str, Any], root: Path) -> None:
    validate_config(config)
    manifest, path = load_collected_manifest(root, config)
    print(f"COLLECTED BASELINE PREFLIGHT PASSED: {len(manifest)} records, manifest={path}")


def run(config, config_path, user_id, worker_id, num_workers):
    import torch
    from models.donut import DonutDocVQA
    from models.qwen import QwenVL

    validate_config(config)
    if num_workers != 5:
        raise ValueError("collected baseline requires five workers")
    if not torch.cuda.is_available():
        raise RuntimeError("collected baseline requires a CARC GPU")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    manifest, _ = load_collected_manifest(root, config)
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="baseline",
        worker_id=worker_id, num_workers=num_workers, record_ids=manifest.index,
    )
    device = torch.device("cuda")
    atomic_json(shard / "environment.json", {
        "python": os.sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        "git_commit": commit,
    })
    models = config["models"]
    for model_key in ("qwen", "donut"):
        settings = models[model_key]
        if model_key == "qwen":
            model = QwenVL(
                device=device, model_id=settings["model_id"], revision=settings["revision"],
                precision=settings.get("precision", "bfloat16"),
                min_pixels=int(settings.get("min_pixels", 200704)),
                max_pixels=int(settings.get("max_pixels", 1003520)),
                max_new_tokens=int(settings.get("max_new_tokens", 32)),
            )
        else:
            model = DonutDocVQA(
                device=device, model_id=settings["model_id"], revision=settings["revision"],
                precision=settings.get("precision", "float32"),
                max_length=int(settings.get("max_length", 64)),
            )
        try:
            for record_id in ids:
                key = _key(record_id, model_key)
                path = shard / "records" / f"{key}.json"
                if path.is_file():
                    continue
                row = manifest.loc[record_id]
                image = load_collected_image(root, row)
                atomic_json(path, _record(
                    config=config, config_hash=config_hash, commit=commit, row=row,
                    model_key=model_key, model=model, image=image, worker_id=worker_id,
                    user_id=user_id, device=device,
                ))
        finally:
            if hasattr(model, "unload"):
                model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_worker_complete(
        shard, config=config, config_hash=config_hash, commit=commit,
        worker_id=worker_id, expected_records=len(ids) * 2,
    )
    merged = merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=len(manifest) * 2,
        sort_columns=["model_key", "record_id"],
    )
    if merged:
        _write_baseline_indexes(output)


def _write_baseline_indexes(output: Path) -> None:
    import pandas as pd
    from checkpoints import atomic_text

    frame = pd.read_parquet(output / "merged" / "records.parquet")
    for model_key in ("qwen", "donut"):
        subset = frame.loc[frame.model_key.eq(model_key)]
        for correct, label in ((True, "correct"), (False, "incorrect")):
            ids = sorted(subset.loc[subset.clean_correct.eq(correct), "record_id"].tolist())
            atomic_text(
                output / "merged" / f"{model_key}_clean_{label}_ids.txt",
                "".join(f"{record_id}\n" for record_id in ids),
            )
    summary = json.loads((output / "merged" / "summary.json").read_text())
    summary["groups"] = [
        {
            "model_key": model,
            "documents": int(len(group)),
            "clean_correct": int(group.clean_correct.sum()),
            "clean_accuracy": float(group.clean_correct.mean()),
            "parse_failures": int(group.parse_status.eq("failed").sum()),
            "median_runtime_seconds": float(group.runtime_seconds.median()),
        }
        for model, group in frame.groupby("model_key")
    ]
    atomic_json(output / "merged" / "summary.json", summary)

