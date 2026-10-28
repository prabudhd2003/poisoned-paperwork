"""Terminal-driven localized patch and EOT-patch experiment on collected receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attacks.eot_patch import optimize_patch
from checkpoints import atomic_json, sha256_file, sha256_text
from defenses import transform_pages
from experiment_utils import (
    initialize_run, load_collected_image, load_collected_manifest,
    merge_json_records, repo_root_from_config, write_worker_complete,
)
from normalization import normalize_receipt_total


PROMPT = "What is the final total amount shown on this receipt? Return only the amount, with no currency symbol, label, or explanation."


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "eot_patch" or config.get("track") != "collected_receipts":
        raise ValueError("EOT patch config requires stage=eot_patch and track=collected_receipts")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("EOT patch is gated until collected baseline is frozen")
    if config.get("controls") != ["no_eot", "eot"]:
        raise ValueError("EOT stage requires the no_eot control and eot treatment")


def _candidates(root: Path, config: dict):
    import pandas as pd

    manifest, _ = load_collected_manifest(root, config["collected_manifest_contract"])
    path = root / config["baseline_predictions"]
    if sha256_file(path) != config["baseline_predictions_sha256"]:
        raise RuntimeError("EOT baseline hash mismatch")
    baseline = pd.read_parquet(path)
    selected = baseline.loc[
        baseline.model_key.eq("qwen") & baseline.clean_correct.eq(True)
        & baseline.status.eq("complete")
    ].set_index("record_id", drop=False)
    limit = int(config.get("document_limit", 25))
    return manifest, selected.sort_index().head(limit)


def preflight(config: dict[str, Any], root: Path) -> None:
    validate_config(config)
    _, selected = _candidates(root, config)
    print(f"EOT PATCH PREFLIGHT PASSED: {len(selected)} frozen clean-correct receipts")


def run(config, config_path, user_id, worker_id, num_workers):
    import torch
    from models.qwen import QwenVL

    validate_config(config)
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("EOT patch requires five CARC GPU workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    manifest, candidates = _candidates(root, config)
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="eot_patch",
        worker_id=worker_id, num_workers=num_workers, record_ids=candidates.index,
    )
    settings = config["model"]
    wrapper = QwenVL(
        device=torch.device("cuda"), model_id=settings["model_id"],
        revision=settings["revision"], precision=settings.get("precision", "bfloat16"),
        min_pixels=int(settings.get("min_pixels", 200704)),
        max_pixels=int(settings.get("max_pixels", 1003520)),
        max_new_tokens=int(settings.get("max_new_tokens", 32)),
    )
    expected = 0
    try:
        for record_id in ids:
            row = manifest.loc[record_id]
            page = load_collected_image(root, row)
            for control in config["controls"]:
                key = sha256_text(f"patch|{record_id}|{control}|{config['seed']}")
                path = shard / "records" / f"{key}.json"
                expected += 1
                if path.is_file():
                    continue
                attack = dict(config["patch_attack"])
                attack["use_eot"] = control == "eot"
                result = optimize_patch(
                    wrapper, page, PROMPT, str(row.attack_target), attack,
                    shard / "images" / key, int(config["seed"]) + int(key[:8], 16),
                )
                raw = wrapper.predict([result["page"]], PROMPT)
                prediction = normalize_receipt_total(raw, "collected_receipts")
                transform_results = {}
                for transform_id in config["final_transform_ids"]:
                    pages = transform_pages([result["page"]], transform_id)
                    transformed_raw = wrapper.predict(pages, PROMPT)
                    transform_results[transform_id] = {
                        "prediction": normalize_receipt_total(transformed_raw, "collected_receipts"),
                        "target_success": normalize_receipt_total(transformed_raw, "collected_receipts") == str(row.attack_target),
                    }
                atomic_json(path, {
                    "experiment_key": key, "run_id": config["run_id"], "stage": "eot_patch",
                    "record_id": record_id, "dataset_name": "collected_receipts",
                    "experiment_split": "external_test", "control": control,
                    "clean_target": str(row.clean_target), "attack_target": str(row.attack_target),
                    "raw_response": raw, "prediction_normalized": prediction,
                    "adversarial_target_success": prediction == str(row.attack_target),
                    "digital_survival_fraction": sum(v["target_success"] for v in transform_results.values()) / len(transform_results),
                    "transform_results": transform_results,
                    "image_path": result["image_path"], "image_sha256": result["image_sha256"],
                    "patch_box": result["box"], "outside_patch_exact": result["outside_patch_exact"],
                    "optimization_trace": result["trace"], "runtime_seconds": result["runtime_seconds"],
                    "worker_id": worker_id, "user_id": user_id, "config_sha256": config_hash,
                    "git_commit": commit, "status": "complete", "error": None,
                })
    finally:
        wrapper.unload()
    write_worker_complete(shard, config=config, config_hash=config_hash, commit=commit,
                          worker_id=worker_id, expected_records=expected)
    if merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=len(candidates) * 2,
        sort_columns=["record_id", "control"],
    ):
        _summarize(output)


def _summarize(output: Path) -> None:
    import pandas as pd

    merged = output / "merged"
    frame = pd.read_parquet(merged / "records.parquet")
    groups = frame.groupby("control").agg(
        trials=("experiment_key", "size"),
        targeted_asr=("adversarial_target_success", "mean"),
        mean_digital_survival=("digital_survival_fraction", "mean"),
        median_runtime_seconds=("runtime_seconds", "median"),
    ).reset_index()
    atomic_json(merged / "patch_summary.json", {
        "groups": json.loads(groups.to_json(orient="records")),
        "warning": "Visible patch ASR is a separate threat model and must not be pooled with imperceptible PGD.",
    })
