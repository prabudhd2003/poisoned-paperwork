"""Clean, non-adaptive, and adaptive evaluation of document defenses."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from checkpoints import atomic_json, sha256_file, sha256_text
from defenses import evaluate_defense
from experiment_utils import (
    initialize_run, load_collected_image, load_collected_manifest,
    merge_json_records, repo_root_from_config, write_worker_complete,
)
from normalization import normalize_receipt_total, normalize_resume_degree


DEFENSE_IDS = ("none", "jpeg", "resize_vote", "selective_multiview")


def _scalar(value: Any):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "defenses":
        raise ValueError("defense config requires stage=defenses")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("defense run is gated until all source hashes and validation choices are frozen")
    if tuple(config.get("defense_ids", ())) != DEFENSE_IDS:
        raise ValueError(f"defense_ids must equal {list(DEFENSE_IDS)}")
    settings = config.get("defense", {})
    if not 1 <= int(settings.get("jpeg_quality", 0)) <= 100:
        raise ValueError("invalid frozen JPEG quality")
    votes = int(settings.get("resize_votes", 0))
    if votes < 3 or votes % 2 == 0:
        raise ValueError("resize_votes must be an odd integer >=3")
    transforms = settings.get("multiview_transforms", [])
    if len(transforms) < 5 or "identity" not in transforms:
        raise ValueError("selective multiview needs identity plus at least four transformations")
    adaptive = config.get("adaptive", {})
    if adaptive.get("enabled"):
        if not adaptive.get("epsilons_255"):
            raise ValueError("adaptive defense attack requires at least one epsilon")
        if int(adaptive.get("document_limit", 0)) < 1:
            raise ValueError("adaptive document_limit must be positive")


def _normalizer(dataset: str):
    if dataset == "resume_parsing_vision":
        return normalize_resume_degree
    return lambda value: normalize_receipt_total(
        value, "collected_receipts" if dataset == "collected_receipts" else dataset
    )


def _parse_images(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    elif hasattr(value, "tolist"):
        parsed = value.tolist()
    else:
        parsed = value
    if isinstance(parsed, dict):
        parsed = [parsed]
    result = []
    for item in parsed or []:
        result.append(str(item.get("path")) if isinstance(item, dict) else str(item))
    return result


def _resolve_image(path: str, root: Path, source_root: Path) -> Path:
    candidate = Path(path)
    choices = [candidate] if candidate.is_absolute() else [source_root / candidate, root / candidate]
    for choice in choices:
        if choice.is_file():
            return choice
    raise FileNotFoundError(f"cannot resolve attack image: {path}")


def _load_attack_sources(root: Path, config: dict[str, Any]):
    import pandas as pd

    frames = []
    for source in config.get("attack_sources", []):
        path = root / source["records"] if not Path(source["records"]).is_absolute() else Path(source["records"])
        if not path.is_file():
            raise FileNotFoundError(f"defense attack source missing: {path}")
        expected = source.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
            raise RuntimeError(f"freeze the exact attack source hash: {path}")
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.status.eq("complete")].copy()
        frame["source_name"] = source["name"]
        frame["source_root"] = str(Path(source.get("artifact_root", root)).resolve())
        frames.append(frame)
    if not frames:
        raise RuntimeError("defense config has no attack sources")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {"record_id", "dataset_name", "experiment_split", "clean_target", "attack_target"}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f"defense sources are missing columns: {missing}")
    if "epsilon_255" not in combined:
        combined["epsilon_255"] = None
    def source_key(row):
        value = row.get("experiment_key")
        if isinstance(value, str) and value.strip():
            return value
        return sha256_text(f"{row.source_name}|{row.record_id}|{row.epsilon_255}")

    combined["source_experiment_key"] = combined.apply(source_key, axis=1)
    per_dataset = config.get("study_documents_per_dataset")
    if per_dataset is not None:
        limit = int(per_dataset)
        if limit < 1:
            raise ValueError("study_documents_per_dataset must be positive")
        selected_ids = set()
        for _, group in combined.groupby("dataset_name"):
            selected_ids.update(sorted(group.record_id.unique())[:limit])
        combined = combined.loc[combined.record_id.isin(selected_ids)].copy()
    return combined.sort_values(["record_id", "source_name", "source_experiment_key"])


def _load_clean_pages(root: Path, config: dict[str, Any], rows):
    from data import load_processed_rows, row_images

    pages = {}
    collected_ids = [rid for rid in rows if rid.startswith("collected_receipts/")]
    if collected_ids:
        manifest, _ = load_collected_manifest(root, config["collected_manifest_contract"])
        for record_id in collected_ids:
            pages[record_id] = [load_collected_image(root, manifest.loc[record_id])]
    by_split: dict[str, list[str]] = {}
    for record_id, split in rows.items():
        if not record_id.startswith("collected_receipts/"):
            by_split.setdefault(split, []).append(record_id)
    for split, ids in by_split.items():
        loaded = load_processed_rows(root, split, ids)
        pages.update({record_id: row_images(row) for record_id, row in loaded.items()})
    return pages


def _load_attacked_pages(row, root: Path) -> list[Image.Image]:
    paths = _parse_images(row.get("images"))
    image_path = row.get("image_path")
    if not paths and image_path is not None and not (
        isinstance(image_path, float) and image_path != image_path
    ):
        paths = [str(row.image_path)]
    source_root = Path(str(row.source_root))
    result = []
    for stored in paths:
        path = _resolve_image(stored, root, source_root)
        with Image.open(path) as source:
            result.append(source.convert("RGB").copy())
    if not result:
        raise RuntimeError(f"attack record has no resolvable images: {row.record_id}")
    return result


def _save_views(root: Path, views: dict[str, list[Image.Image]]) -> dict[str, list[str]]:
    saved = {}
    for view_id, pages in views.items():
        paths = []
        for index, page in enumerate(pages):
            path = root / view_id / f"page_{index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            page.save(path, format="PNG")
            paths.append(str(path))
        saved[view_id] = paths
    return saved


def _defense_record(
    *, wrapper, pages, row, defense_id, config, config_hash, commit,
    worker_id, user_id, example_type, key, audit_root: Path | None,
):
    dataset = str(row["dataset_name"])
    normalize = _normalizer(dataset)
    started = time.perf_counter()
    prompt = row.get("prompt_text")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = config["prompts"][
            "resume" if dataset == "resume_parsing_vision" else "receipt"
        ]
    result, views = evaluate_defense(
        pages=pages, defense_id=defense_id, record_id=str(row["record_id"]),
        predict=lambda value: wrapper.predict(value, prompt),
        normalize=normalize, config=config["defense"],
    )
    saved = _save_views(audit_root, views) if audit_root is not None else {}
    clean_target, attack_target = str(row["clean_target"]), str(row["attack_target"])
    return {
        "experiment_key": key, "run_id": config["run_id"], "stage": "defenses",
        "record_id": str(row["record_id"]), "dataset_name": dataset,
        "experiment_split": str(row["experiment_split"]),
        "example_type": example_type, "source_name": str(row.get("source_name", "clean")),
        "source_experiment_key": str(row.get("source_experiment_key", "clean")),
        "epsilon_255": _scalar(row.get("epsilon_255")), "defense_id": defense_id,
        "clean_target": clean_target, "attack_target": attack_target,
        "final_prediction": result.final_prediction,
        "clean_correct": result.final_prediction == clean_target,
        "adversarial_target_success": result.final_prediction == attack_target,
        "abstained": result.abstained, "agreement": result.agreement,
        "prediction_entropy": result.entropy,
        "view_ids": list(result.view_ids),
        "raw_predictions": list(result.raw_predictions),
        "normalized_predictions": list(result.normalized_predictions),
        "additional_vlm_passes": result.additional_vlm_passes,
        "runtime_seconds": time.perf_counter() - started,
        "audit_images": saved, "worker_id": worker_id, "user_id": user_id,
        "config_sha256": config_hash, "git_commit": commit,
        "status": "complete", "error": None,
    }


def preflight(config: dict[str, Any], root: Path) -> None:
    validate_config(config)
    attacks = _load_attack_sources(root, config)
    documents = attacks.record_id.nunique()
    datasets = sorted(attacks.dataset_name.unique())
    print(f"DEFENSE PREFLIGHT PASSED: {documents} documents, {len(attacks)} attacks, {datasets}")


def run(config, config_path, user_id, worker_id, num_workers):
    import pandas as pd
    import torch
    from models.qwen import QwenVL

    validate_config(config)
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("defenses require five CARC GPU workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    attacks = _load_attack_sources(root, config)
    document_splits = attacks.groupby("record_id").experiment_split.first().to_dict()
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="defenses",
        worker_id=worker_id, num_workers=num_workers, record_ids=document_splits,
    )
    clean_pages = _load_clean_pages(root, config, {record_id: document_splits[record_id] for record_id in ids})
    adaptive_settings = config.get("adaptive", {})
    adaptive_ids = (
        sorted(
            record_id for record_id in document_splits
            if record_id.startswith("collected_receipts/")
        )[:int(adaptive_settings.get("document_limit", 0))]
        if adaptive_settings.get("enabled") else []
    )
    if adaptive_settings.get("enabled") and not adaptive_ids:
        raise RuntimeError("adaptive defense evaluation requires a collected-receipt attack source")
    settings = config["model"]
    wrapper = QwenVL(
        device=torch.device("cuda"), model_id=settings["model_id"],
        revision=settings["revision"], precision=settings.get("precision", "bfloat16"),
        min_pixels=int(settings.get("min_pixels", 200704)),
        max_pixels=int(settings.get("max_pixels", 1003520)),
        max_new_tokens=int(settings.get("max_new_tokens", 32)),
    )
    expected = 0
    audit_limit = int(config.get("audit_documents_per_worker", 3))
    audit_ids = set(ids[:audit_limit])
    try:
        for record_id in ids:
            document_attacks = attacks.loc[attacks.record_id.eq(record_id)]
            representative = document_attacks.iloc[0].copy()
            for defense_id in DEFENSE_IDS:
                key = sha256_text(f"defense|clean|{record_id}|{defense_id}")
                path = shard / "records" / f"{key}.json"
                expected += 1
                if not path.is_file():
                    audit = shard / "defended_images" / "clean" / record_id.replace("/", "__") / defense_id if record_id in audit_ids else None
                    atomic_json(path, _defense_record(
                        wrapper=wrapper, pages=clean_pages[record_id], row=representative,
                        defense_id=defense_id, config=config, config_hash=config_hash,
                        commit=commit, worker_id=worker_id, user_id=user_id,
                        example_type="clean", key=key, audit_root=audit,
                    ))
            for _, attack_row in document_attacks.iterrows():
                attack_row = attack_row.copy()
                attacked_pages = _load_attacked_pages(attack_row, root)
                for defense_id in DEFENSE_IDS:
                    key = sha256_text(
                        f"defense|attack|{attack_row.source_name}|"
                        f"{attack_row.source_experiment_key}|{defense_id}"
                    )
                    path = shard / "records" / f"{key}.json"
                    expected += 1
                    if path.is_file():
                        continue
                    audit = shard / "defended_images" / "attacked" / record_id.replace("/", "__") / str(attack_row.source_experiment_key) / defense_id if record_id in audit_ids else None
                    atomic_json(path, _defense_record(
                        wrapper=wrapper, pages=attacked_pages, row=attack_row,
                        defense_id=defense_id, config=config, config_hash=config_hash,
                        commit=commit, worker_id=worker_id, user_id=user_id,
                        example_type="non_adaptive_attack", key=key, audit_root=audit,
                    ))
            if record_id in adaptive_ids:
                if len(clean_pages[record_id]) != 1:
                    raise RuntimeError("adaptive selective-defense evaluation currently requires one-page receipts")
                from attacks.adaptive_qwen import adaptive_multiview_attack
                prompt = representative.get("prompt_text")
                if not isinstance(prompt, str) or not prompt.strip():
                    prompt = config["prompts"][
                        "resume" if representative["dataset_name"] == "resume_parsing_vision" else "receipt"
                    ]
                for epsilon in adaptive_settings["epsilons_255"]:
                    epsilon = int(epsilon)
                    key = sha256_text(f"adaptive-defense|{record_id}|eps={epsilon}")
                    path = shard / "records" / f"{key}.json"
                    expected += 1
                    if path.is_file():
                        continue
                    optimized = adaptive_multiview_attack(
                        wrapper, clean_pages[record_id][0], prompt,
                        str(representative["attack_target"]), adaptive_settings,
                        config["defense"], epsilon,
                        shard / "defended_images" / "adaptive" / record_id.replace("/", "__") / f"eps_{epsilon}",
                    )
                    adaptive_row = representative.copy()
                    adaptive_row["source_name"] = "adaptive_selective_multiview"
                    adaptive_row["source_experiment_key"] = key
                    adaptive_row["epsilon_255"] = epsilon
                    record = _defense_record(
                        wrapper=wrapper, pages=optimized["pages"], row=adaptive_row,
                        defense_id="selective_multiview", config=config,
                        config_hash=config_hash, commit=commit, worker_id=worker_id,
                        user_id=user_id, example_type="adaptive_attack", key=key,
                        audit_root=None,
                    )
                    record.update({
                        "adaptive_attack_trace": optimized["trace"],
                        "adaptive_linf_255": optimized["linf_255"],
                        "adaptive_runtime_seconds": optimized["runtime_seconds"],
                        "adaptive_images": optimized["images"],
                        "adaptive_bpda_contract": optimized["bpda"],
                    })
                    atomic_json(path, record)
    finally:
        wrapper.unload()
    write_worker_complete(
        shard, config=config, config_hash=config_hash, commit=commit,
        worker_id=worker_id, expected_records=expected,
    )
    total_expected = (
        len(document_splits) * len(DEFENSE_IDS)
        + len(attacks) * len(DEFENSE_IDS)
        + len(adaptive_ids) * len(adaptive_settings.get("epsilons_255", []))
    )
    if merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=total_expected,
        sort_columns=["example_type", "dataset_name", "record_id", "defense_id"],
    ):
        _summarize(output)


def _summarize(output: Path) -> None:
    import pandas as pd
    from checkpoints import atomic_json

    frame = pd.read_parquet(output / "merged" / "records.parquet")
    groups = frame.groupby(["example_type", "dataset_name", "defense_id"]).agg(
        trials=("experiment_key", "size"),
        clean_accuracy=("clean_correct", "mean"),
        attack_success=("adversarial_target_success", "mean"),
        abstention_rate=("abstained", "mean"),
        mean_agreement=("agreement", "mean"),
        median_runtime_seconds=("runtime_seconds", "median"),
        mean_extra_passes=("additional_vlm_passes", "mean"),
    ).reset_index()
    atomic_json(output / "merged" / "defense_summary.json", {
        "groups": json.loads(groups.to_json(orient="records")),
        "novel_contribution": (
            "Selective multi-view consistency jointly reports robustness, clean coverage, "
            "answer stability, abstention, and compute cost."
        ),
    })

