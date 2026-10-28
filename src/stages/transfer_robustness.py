"""Black-box model transfer and digital transformation robustness stage."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from checkpoints import atomic_json, sha256_file, sha256_text
from defenses import jpeg_roundtrip, resize_roundtrip, transform_pages
from experiment_utils import initialize_run, merge_json_records, repo_root_from_config, write_worker_complete
from normalization import normalize_receipt_total, normalize_resume_degree
from stages.defenses import _load_attack_sources, _load_attacked_pages, _load_clean_pages


TRANSFORMS = (
    "png_reload", "jpeg_q95", "jpeg_q85", "jpeg_q75", "resize_0.75",
    "resize_0.50", "blur_0.5", "blur_1.0", "brightness_0.90",
    "brightness_1.10", "common_pipeline",
)


def _scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "transfer_robustness":
        raise ValueError("transfer config requires stage=transfer_robustness")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("transfer run is gated until attack sources and hashes are frozen")
    if tuple(config.get("transform_ids", ())) != TRANSFORMS:
        raise ValueError(f"transform_ids must equal the frozen suite: {list(TRANSFORMS)}")
    if set(config.get("model_keys", ())) != {"qwen", "internvl"}:
        raise ValueError("transfer requires Qwen and InternVL")


def _normalizer(dataset: str):
    return normalize_resume_degree if dataset == "resume_parsing_vision" else (
        lambda value: normalize_receipt_total(
            value, "collected_receipts" if dataset == "collected_receipts" else dataset
        )
    )


def _transform(pages, transform_id: str):
    if transform_id == "png_reload":
        return transform_pages(pages, "identity")
    if transform_id == "common_pipeline":
        value = [jpeg_roundtrip(page, 85) for page in pages]
        value = [resize_roundtrip(page, 0.75) for page in value]
        return transform_pages(value, "blur_0.5")
    return transform_pages(pages, transform_id)


def _prompt(row, config) -> str:
    value = row.get("prompt_text")
    if isinstance(value, str) and value.strip():
        return value
    dataset = str(row["dataset_name"])
    return config["prompts"]["resume" if dataset == "resume_parsing_vision" else "receipt"]


def preflight(config: dict[str, Any], root: Path) -> None:
    validate_config(config)
    attacks = _load_attack_sources(root, config)
    print(
        f"TRANSFER PREFLIGHT PASSED: {attacks.record_id.nunique()} documents, "
        f"{len(attacks)} attack records, {len(TRANSFORMS)} transforms"
    )


def _evaluate(wrapper, model_key, pages, row, transform_id, example_type,
              config, config_hash, commit, worker_id, user_id):
    transformed = _transform(pages, transform_id)
    started = time.perf_counter()
    error = None
    raw = None
    try:
        raw = wrapper.predict(transformed, _prompt(row, config))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    normalize = _normalizer(str(row["dataset_name"]))
    prediction = None if error else normalize(raw)
    clean = str(row["clean_target"])
    target = str(row["attack_target"])
    source_key = str(row.get("source_experiment_key", "clean"))
    key = sha256_text(f"transfer|{example_type}|{source_key}|{model_key}|{transform_id}")
    return {
        "experiment_key": key, "run_id": config["run_id"],
        "stage": "transfer_robustness", "record_id": str(row["record_id"]),
        "dataset_name": str(row["dataset_name"]),
        "experiment_split": str(row["experiment_split"]),
        "source_name": str(row.get("source_name", "clean")),
        "source_experiment_key": source_key, "example_type": example_type,
        "epsilon_255": _scalar(row.get("epsilon_255")), "model_key": model_key,
        "model_id": wrapper.model_id, "model_revision": wrapper.revision,
        "transform_id": transform_id, "clean_target": clean,
        "attack_target": target, "raw_response": raw,
        "prediction_normalized": prediction, "clean_target_correct": prediction == clean,
        "adversarial_target_success": prediction == target,
        "source_attack_success": bool(row.get("saved_image_success", False)),
        "runtime_seconds": time.perf_counter() - started,
        "worker_id": worker_id, "user_id": user_id,
        "config_sha256": config_hash, "git_commit": commit,
        "status": "failed" if error else "complete", "error": error,
    }


def run(config, config_path, user_id, worker_id, num_workers):
    import torch
    from models.internvl import InternVL
    from models.qwen import QwenVL

    validate_config(config)
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("transfer requires five CARC GPU workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    attacks = _load_attack_sources(root, config)
    splits = attacks.groupby("record_id").experiment_split.first().to_dict()
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="transfer_robustness",
        worker_id=worker_id, num_workers=num_workers, record_ids=splits,
    )
    clean_pages = _load_clean_pages(root, config, {key: splits[key] for key in ids})
    model_classes = {"qwen": QwenVL, "internvl": InternVL}
    expected = 0
    for model_key in ("qwen", "internvl"):
        settings = config["models"][model_key]
        wrapper = model_classes[model_key](
            device=torch.device("cuda"), model_id=settings["model_id"],
            revision=settings["revision"], precision=settings.get("precision", "bfloat16"),
            max_new_tokens=int(settings.get("max_new_tokens", 32)),
            **({"min_pixels": int(settings.get("min_pixels", 200704)),
                "max_pixels": int(settings.get("max_pixels", 1003520))} if model_key == "qwen" else
               {"image_size": int(settings.get("image_size", 448)),
                "max_tiles": int(settings.get("max_tiles", 12))}),
        )
        try:
            for record_id in ids:
                rows = attacks.loc[attacks.record_id.eq(record_id)]
                representative = rows.iloc[0].copy()
                for transform_id in TRANSFORMS:
                    expected += 1
                    clean_key = sha256_text(
                        f"transfer|clean|{representative.source_experiment_key}|{model_key}|{transform_id}"
                    )
                    clean_path = shard / "records" / f"{clean_key}.json"
                    if clean_path.is_file():
                        continue
                    trial = _evaluate(
                        wrapper, model_key, clean_pages[record_id], representative,
                        transform_id, "clean", config, config_hash, commit, worker_id, user_id,
                    )
                    atomic_json(clean_path, trial)
                for _, row in rows.iterrows():
                    attacked = _load_attacked_pages(row, root)
                    for transform_id in TRANSFORMS:
                        expected += 1
                        key = sha256_text(
                            f"transfer|attack|{row.source_experiment_key}|{model_key}|{transform_id}"
                        )
                        path = shard / "records" / f"{key}.json"
                        if path.is_file():
                            continue
                        atomic_json(path, _evaluate(
                            wrapper, model_key, attacked, row, transform_id, "attack",
                            config, config_hash, commit, worker_id, user_id,
                        ))
        finally:
            wrapper.unload()
    write_worker_complete(
        shard, config=config, config_hash=config_hash, commit=commit,
        worker_id=worker_id, expected_records=expected,
    )
    total = 2 * len(TRANSFORMS) * (len(splits) + len(attacks))
    if merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=total,
        sort_columns=["example_type", "model_key", "record_id", "epsilon_255", "transform_id"],
    ):
        _summarize(output)


def _summarize(output: Path) -> None:
    import pandas as pd

    merged = output / "merged"
    frame = pd.read_parquet(merged / "records.parquet")
    attacks = frame.loc[frame.example_type.eq("attack")].copy()
    direct = attacks.loc[attacks.transform_id.eq("png_reload")].copy()
    direct.to_parquet(merged / "direct_transfer.parquet", index=False)
    attacks.to_parquet(merged / "transformation_trials.parquet", index=False)
    features = attacks.groupby([
        "record_id", "source_experiment_key", "model_key", "epsilon_255"
    ], dropna=False).agg(
        transform_success_fraction=("adversarial_target_success", "mean"),
        transforms=("experiment_key", "size"),
    ).reset_index()
    features.to_parquet(merged / "document_features.parquet", index=False)
    groups = frame.groupby([
        "example_type", "model_key", "transform_id", "epsilon_255"
    ], dropna=False).agg(
        trials=("experiment_key", "size"),
        clean_accuracy=("clean_target_correct", "mean"),
        attack_success=("adversarial_target_success", "mean"),
        failures=("status", lambda values: int((values != "complete").sum())),
    ).reset_index()
    atomic_json(merged / "transfer_summary.json", {
        "groups": json.loads(groups.to_json(orient="records")),
        "interpretation": "InternVL is evaluated only after optimization; no transfer-model feedback creates attacks.",
    })
