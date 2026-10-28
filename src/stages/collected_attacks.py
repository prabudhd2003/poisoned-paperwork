"""Qwen and Donut attacks for the frozen collected-receipt external test."""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from checkpoints import atomic_json, sha256_file, sha256_text
from experiment_utils import (
    initialize_run, load_collected_image, load_collected_manifest,
    merge_json_records, repo_root_from_config, write_worker_complete,
)
from normalization import normalize_receipt_total


PROMPT = (
    "What is the final total amount shown on this receipt? Return only the "
    "amount, with no currency symbol, label, or explanation."
)


def _validate(config: dict[str, Any], model_key: str) -> None:
    if config.get("track") != "collected_receipts":
        raise ValueError("collected attack requires track=collected_receipts")
    if config.get("split") != "external_test":
        raise ValueError("collected attacks require split=external_test")
    if config.get("ready_for_production") is not True:
        raise RuntimeError("collected attack remains gated")
    if config.get("model_key") != model_key:
        raise ValueError(f"config model_key must be {model_key}")
    attack = config.get("attack", {})
    if attack.get("epsilons_255") != [2, 4, 8, 16]:
        raise ValueError("external test must use the frozen epsilon grid [2,4,8,16]")
    if int(attack.get("max_steps", 0)) < 1:
        raise ValueError("max_steps must be positive")


def _load_candidates(root: Path, config: dict[str, Any], model_key: str):
    import pandas as pd

    manifest, _ = load_collected_manifest(root, config)
    path = root / str(config["baseline_predictions"])
    if not path.is_file():
        raise FileNotFoundError(f"collected baseline predictions missing: {path}")
    expected_hash = config.get("baseline_predictions_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RuntimeError("freeze baseline_predictions_sha256 before attacking")
    if sha256_file(path) != expected_hash:
        raise RuntimeError("collected baseline predictions hash mismatch")
    baseline = pd.read_parquet(path)
    selected = baseline.loc[
        baseline.model_key.eq(model_key)
        & baseline.dataset_name.eq("collected_receipts")
        & baseline.experiment_split.eq("external_test")
        & baseline.clean_correct.eq(True)
        & baseline.status.eq("complete")
    ].copy().set_index("record_id", drop=False)
    if selected.empty:
        raise RuntimeError(f"no clean-correct collected receipts for {model_key}")
    if not set(selected.index) <= set(manifest.index):
        raise RuntimeError("baseline contains collected IDs outside the frozen manifest")
    for record_id in selected.index:
        source = manifest.loc[record_id]
        for field in ("clean_target", "attack_target", "processed_sha256"):
            if str(selected.loc[record_id].get(field)) != str(source[field]):
                raise RuntimeError(f"baseline/manifest drift for {record_id}: {field}")
    return manifest, selected.sort_index()


def _key(model_key: str, record_id: str, epsilon: int, seed: int) -> str:
    return sha256_text(f"{model_key}|collected|{record_id}|eps={epsilon}|seed={seed}")


def preflight(config: dict[str, Any], root: Path, model_key: str) -> None:
    _validate(config, model_key)
    manifest, selected = _load_candidates(root, config, model_key)
    print(
        f"COLLECTED {model_key.upper()} ATTACK PREFLIGHT PASSED: "
        f"{len(selected)}/{len(manifest)} clean-correct candidates"
    )


def run_qwen(config, config_path, user_id, worker_id, num_workers):
    import torch
    from attacks.qwen import attack
    from models.qwen import QwenVL

    _validate(config, "qwen")
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("collected Qwen attack requires five CARC GPU workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    manifest, candidates = _load_candidates(root, config, "qwen")
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="qwen_attack",
        worker_id=worker_id, num_workers=num_workers, record_ids=candidates.index,
    )
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    settings = config["model"]
    wrapper = QwenVL(
        device=torch.device("cuda"), model_id=settings["model_id"],
        revision=settings["revision"], precision=settings.get("precision", "bfloat16"),
        min_pixels=int(settings.get("min_pixels", 200704)),
        max_pixels=int(settings.get("max_pixels", 1003520)),
        max_new_tokens=int(settings.get("max_new_tokens", 32)),
    )
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(wrapper.device).eval()
        for record_id in ids:
            source = manifest.loc[record_id]
            page = load_collected_image(root, source)
            baseline = candidates.loc[record_id]
            row = {
                "record_id": record_id, "prompt_text": PROMPT,
                "attack_target": str(source.attack_target),
                "clean_target": str(source.clean_target),
                "raw_response": baseline.raw_response,
                "prediction_normalized": baseline.prediction_normalized,
            }
            for epsilon in config["attack"]["epsilons_255"]:
                epsilon = int(epsilon)
                key = _key("qwen", record_id, epsilon, seed)
                record_path = shard / "records" / f"{key}.json"
                if record_path.is_file():
                    continue
                provenance = {
                    "run_id": config["run_id"], "stage": "qwen_attack",
                    "track": "collected_receipts", "experiment_key": key,
                    "config_sha256": config_hash, "git_commit": commit,
                    "model_id": wrapper.model_id, "model_revision": wrapper.revision,
                    "worker_id": worker_id, "user_id": user_id,
                    "dataset_name": "collected_receipts", "experiment_split": "external_test",
                    "clean_target": str(source.clean_target),
                    "attack_target": str(source.attack_target), "seed": seed,
                }
                try:
                    result = attack(
                        wrapper, [page], row, config, epsilon, shard, key, provenance,
                        lpips_model, lambda value: normalize_receipt_total(value, "collected_receipts"),
                    )
                    record = {**provenance, **result, "epsilon_255": epsilon,
                              "status": "complete", "error": None}
                except Exception as exc:
                    record = {**provenance, "epsilon_255": epsilon,
                              "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                atomic_json(record_path, record)
    finally:
        wrapper.unload()
    expected = len(ids) * len(config["attack"]["epsilons_255"])
    write_worker_complete(shard, config=config, config_hash=config_hash, commit=commit,
                          worker_id=worker_id, expected_records=expected)
    if merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=len(candidates) * len(config["attack"]["epsilons_255"]),
        sort_columns=["record_id", "epsilon_255"],
    ):
        _write_attack_summary(output, "qwen")


def run_donut(config, config_path, user_id, worker_id, num_workers):
    import torch
    from attacks.donut import targeted_image_attack
    from models.donut import DonutDocVQA, pil_to_rgb_tensor, rgb_tensor_to_pil
    from skimage.metrics import structural_similarity

    _validate(config, "donut")
    if num_workers != 5 or not torch.cuda.is_available():
        raise RuntimeError("collected Donut attack requires five CARC GPU workers")
    config_path = Path(config_path).resolve()
    root = repo_root_from_config(config_path)
    manifest, candidates = _load_candidates(root, config, "donut")
    output, shard, config_hash, commit, ids = initialize_run(
        root=root, config_path=config_path, config=config, stage="donut_attack",
        worker_id=worker_id, num_workers=num_workers, record_ids=candidates.index,
    )
    seed = int(config["seed"])
    settings = config["model"]
    device = torch.device("cuda")
    model = DonutDocVQA(
        device=device, model_id=settings["model_id"], revision=settings["revision"],
        precision=settings.get("precision", "float32"),
        max_length=int(settings.get("max_length", 64)),
    )
    try:
        import lpips
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
        for record_id in ids:
            source = manifest.loc[record_id]
            clean_pil = load_collected_image(root, source)
            clean = pil_to_rgb_tensor(clean_pil, device)
            target = str(source.attack_target)
            for epsilon in config["attack"]["epsilons_255"]:
                epsilon = int(epsilon)
                key = _key("donut", record_id, epsilon, seed)
                record_path = shard / "records" / f"{key}.json"
                image_path = shard / "images" / f"{key}.png"
                if record_path.is_file():
                    continue
                started = time.perf_counter()
                common = {
                    "experiment_key": key, "run_id": config["run_id"],
                    "stage": "donut_attack", "track": "collected_receipts",
                    "record_id": record_id, "dataset_name": "collected_receipts",
                    "experiment_split": "external_test", "epsilon_255": epsilon,
                    "clean_target": str(source.clean_target), "attack_target": target,
                    "worker_id": worker_id, "user_id": user_id, "seed": seed,
                    "config_sha256": config_hash, "git_commit": commit,
                    "model_id": model.model_id, "model_revision": model.revision,
                }
                try:
                    result = targeted_image_attack(
                        clean,
                        lambda x: model.attack_objective(
                            x, question=PROMPT, target_text=target,
                            loss_type=config["attack"].get("loss_type", "cross_entropy"),
                        ),
                        epsilon_255=epsilon,
                        step_size_255=float(config["attack"]["step_size_255"][str(epsilon)]),
                        max_steps=int(config["attack"]["max_steps"]),
                        optimizer_name=config["attack"].get("optimizer", "adam"),
                        checkpoint_interval=int(config["attack"].get("checkpoint_interval", 10)),
                    )
                    adv = rgb_tensor_to_pil(result.adversarial)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    adv.save(image_path, format="PNG")
                    raw = model.predict_pil(adv, PROMPT)
                    normalized = normalize_receipt_total(raw, "collected_receipts")
                    a = clean.detach().permute(1, 2, 0).cpu().numpy()
                    b = result.adversarial.detach().permute(1, 2, 0).cpu().numpy()
                    with torch.inference_mode():
                        lpips_value = float(perceptual(
                            clean.unsqueeze(0).mul(2).sub(1),
                            result.adversarial.unsqueeze(0).mul(2).sub(1),
                        ).item())
                    record = {
                        **common, "status": "complete", "error": None,
                        "raw_response": raw, "prediction_normalized": normalized,
                        "saved_image_success": normalized == target,
                        "completed_steps": result.completed_steps,
                        "best_loss": result.best_loss, "final_loss": result.final_loss,
                        "linf_255": float((result.adversarial - clean).abs().max() * 255),
                        "lpips": lpips_value,
                        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
                        "image_path": str(image_path.relative_to(root)),
                        "image_sha256": sha256_file(image_path),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                except Exception as exc:
                    record = {**common, "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}",
                              "runtime_seconds": time.perf_counter() - started}
                atomic_json(record_path, record)
    finally:
        del model
        torch.cuda.empty_cache()
    expected = len(ids) * len(config["attack"]["epsilons_255"])
    write_worker_complete(shard, config=config, config_hash=config_hash, commit=commit,
                          worker_id=worker_id, expected_records=expected)
    if merge_json_records(
        output, config=config, config_hash=config_hash, num_workers=num_workers,
        expected_records=len(candidates) * len(config["attack"]["epsilons_255"]),
        sort_columns=["record_id", "epsilon_255"],
    ):
        _write_attack_summary(output, "donut")


def _write_attack_summary(output: Path, model_key: str) -> None:
    import pandas as pd

    path = output / "merged" / "records.parquet"
    frame = pd.read_parquet(path)
    complete = frame.loc[frame.status.eq("complete")].copy()
    groups = []
    for epsilon, group in complete.groupby("epsilon_255"):
        groups.append({
            "model_key": model_key, "epsilon_255": int(epsilon),
            "attempted": int(len(group)),
            "targeted_successes": int(group.saved_image_success.sum()),
            "conditional_asr": float(group.saved_image_success.mean()),
            "median_lpips": float(group.lpips.median()),
            "median_ssim": float(group.ssim.median()),
            "median_runtime_seconds": float(group.runtime_seconds.median()),
        })
    summary_path = output / "merged" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["groups"] = groups
    summary["failed_records"] = int(frame.status.ne("complete").sum())
    atomic_json(summary_path, summary)

