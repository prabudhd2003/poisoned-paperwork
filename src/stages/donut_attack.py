"""Checkpointed Donut targeted-attack worker for SROIE and CORD receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attacks.donut import checkpoint_state, load_checkpoint_state, targeted_image_attack
from models.donut import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    DonutDocVQA,
    pil_to_rgb_tensor,
    rgb_tensor_to_pil,
)
from normalization import normalize_receipt_total
from sharding import assigned_record_ids


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_key(record_id: str, epsilon: float, seed: int) -> str:
    base = f"{record_id}__donut__receipt_total__targeted__eps-{epsilon:g}__seed-{seed}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "stage", "mode", "run_id", "seed", "split", "output_root", "datasets",
        "baseline_predictions", "baseline_merge_manifest",
        "baseline_predictions_sha256", "baseline_config_sha256",
        "baseline_git_commit", "attack",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Donut attack config is missing: {', '.join(missing)}")
    if config["stage"] != "donut_attack":
        raise ValueError("stage must be 'donut_attack'")
    if config["mode"] not in {"smoke", "validation", "test"}:
        raise ValueError("mode must be smoke, validation, or test")
    if config["output_root"] != "outputs/donut_attack":
        raise ValueError("output_root must be 'outputs/donut_attack'")
    if set(config["datasets"]) != {"sroie", "cord_v2"}:
        raise ValueError("datasets must contain exactly sroie and cord_v2")
    if config["split"] not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if config["mode"] == "smoke" and config["split"] != "validation":
        raise ValueError("smoke mode must use the validation split")
    if config["mode"] != "smoke" and config.get("smoke_per_dataset") is not None:
        raise ValueError("smoke_per_dataset is allowed only in smoke mode")
    if config["mode"] != "smoke" and not config.get("ready_for_production", False):
        raise ValueError(
            "production config is gated; set ready_for_production=true only after smoke review"
        )
    if config["split"] == "test" and not config.get("frozen_for_test", False):
        raise ValueError("test attacks require frozen_for_test=true")
    if config["mode"] == "test" and not config.get("validation_run_id"):
        raise ValueError("test attacks require validation_run_id")
    for field in (
        "baseline_predictions_sha256", "baseline_config_sha256", "baseline_git_commit"
    ):
        value = config[field]
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40,64}", value):
            raise ValueError(f"{field} must be a lowercase Git/SHA-256 hexadecimal value")
    attack = config["attack"]
    for field in ("epsilons_255", "step_size_255", "max_steps", "loss_type"):
        if field not in attack:
            raise ValueError(f"attack config is missing {field!r}")
    if not attack["epsilons_255"]:
        raise ValueError("at least one epsilon is required")
    epsilons = [float(value) for value in attack["epsilons_255"]]
    if any(value < 0 or value > 255 for value in epsilons):
        raise ValueError("every epsilon must be between 0 and 255")
    if any(not value.is_integer() for value in epsilons):
        raise ValueError("epsilon values must be integer 0-255 pixel levels")
    if int(attack["max_steps"]) < 1:
        raise ValueError("max_steps must be positive")
    if attack["loss_type"] not in {"paper_margin", "cross_entropy"}:
        raise ValueError("unsupported targeted loss")
    if attack.get("optimizer", "adam") not in {"adam", "sign"}:
        raise ValueError("optimizer must be adam or sign")
    missing_steps = [
        value for value in epsilons if str(int(value)) not in attack["step_size_255"]
    ]
    if missing_steps:
        raise ValueError(f"step_size_255 is missing epsilon keys: {missing_steps}")
    if any(
        float(attack["step_size_255"][str(int(value))]) <= 0 for value in epsilons
    ):
        raise ValueError("every step size must be positive")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_eligible_baseline(config: dict[str, Any], repo_root: Path):
    import pandas as pd

    path = _resolve_repo_path(repo_root, config["baseline_predictions"])
    if not path.is_file():
        raise FileNotFoundError(
            f"baseline predictions not found: {path}. Formal attacks require Donut "
            "clean-correct IDs from the baseline stage."
        )
    observed_predictions_sha256 = _sha256_file(path)
    if observed_predictions_sha256 != config["baseline_predictions_sha256"]:
        raise ValueError(
            "baseline predictions SHA-256 does not match the frozen config: "
            f"{observed_predictions_sha256}"
        )
    provenance_path = _resolve_repo_path(repo_root, config["baseline_merge_manifest"])
    provenance = _load_json(provenance_path)
    if provenance is None:
        raise FileNotFoundError(f"baseline merge provenance not found: {provenance_path}")
    expected_provenance = {
        "config_sha256": config["baseline_config_sha256"],
        "git_commit": config["baseline_git_commit"],
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ValueError(f"baseline merge {field} does not match the frozen config")
    if int(provenance.get("records", -1)) != 527:
        raise ValueError("baseline merge manifest must describe the accepted 527-record run")

    frame = pd.read_parquet(path)
    needed = {
        "record_id",
        "dataset_name",
        "experiment_split",
        "model_id",
        "model_revision",
        "prompt_id",
        "prompt_text",
        "clean_target",
        "attack_target",
        "clean_correct",
        "attack_eligible",
        "status",
        "parse_status",
        "config_sha256",
        "git_commit",
    }
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"baseline file is missing columns: {', '.join(missing)}")
    prediction_column = (
        "prediction_normalized"
        if "prediction_normalized" in frame.columns
        else "clean_prediction"
        if "clean_prediction" in frame.columns
        else None
    )
    if prediction_column is None:
        raise ValueError(
            "baseline file must contain prediction_normalized (or legacy clean_prediction)"
        )
    frame["clean_prediction"] = frame[prediction_column]
    selected = frame[
        frame["dataset_name"].isin(config.get("datasets", ["sroie", "cord_v2"]))
        & frame["experiment_split"].eq(config["split"])
        & frame["model_id"].eq(config.get("model_id", DEFAULT_MODEL_ID))
        & frame["model_revision"].eq(config.get("model_revision", DEFAULT_REVISION))
        & frame["prompt_id"].eq(config.get("prompt_id", "receipt_total_v1"))
        & frame["prompt_text"].eq(
            config.get(
                "question",
                "What is the final total amount shown on this receipt? Return only the amount, "
                "with no currency symbol, label, or explanation.",
            )
        )
        & frame["config_sha256"].eq(config["baseline_config_sha256"])
        & frame["git_commit"].eq(config["baseline_git_commit"])
        & frame["status"].eq("complete")
        & frame["parse_status"].eq("ok")
        & frame["clean_correct"].eq(True)
        & frame["attack_eligible"].eq(True)
    ].copy()
    if selected["record_id"].duplicated().any():
        raise ValueError("baseline contains duplicate eligible record IDs")
    smoke_per_dataset = config.get("smoke_per_dataset")
    if smoke_per_dataset is not None:
        selected = (
            selected.sort_values("record_id")
            .groupby("dataset_name", group_keys=False)
            .head(int(smoke_per_dataset))
        )
    if selected.empty:
        raise ValueError("no clean-correct attack-eligible Donut receipts were selected")
    selected.attrs["predictions_sha256"] = observed_predictions_sha256
    return selected.set_index("record_id", drop=False)


def _load_rows(repo_root: Path, split: str, assigned: list[str]) -> dict[str, dict[str, Any]]:
    from datasets import load_from_disk

    wanted = set(assigned)
    rows: dict[str, dict[str, Any]] = {}
    for dataset_name in ("sroie", "cord_v2"):
        dataset = load_from_disk(str(repo_root / "data" / "processed" / dataset_name))[split]
        for row in dataset:
            record_id = f"{dataset_name}/{row['source_split']}/{row['document_id']}"
            if record_id in wanted:
                rows[record_id] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"processed datasets are missing {len(missing)} assigned records")
    return rows


def _linf_255(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    return float((clean - adversarial).abs().max().item() * 255.0)


def _perceptibility(clean: torch.Tensor, adversarial: torch.Tensor, lpips_model) -> tuple[float, float]:
    from skimage.metrics import structural_similarity

    clean_np = clean.detach().permute(1, 2, 0).cpu().numpy()
    adversarial_np = adversarial.detach().permute(1, 2, 0).cpu().numpy()
    ssim = float(
        structural_similarity(clean_np, adversarial_np, channel_axis=2, data_range=1.0)
    )
    with torch.inference_mode():
        lpips_value = float(
            lpips_model(
                clean.unsqueeze(0).mul(2).sub(1),
                adversarial.unsqueeze(0).mul(2).sub(1),
            ).item()
        )
    return lpips_value, ssim


DONUT_REQUIRED_RECORD_FIELDS = {
    "run_id", "stage", "experiment_key", "record_id", "dataset_name",
    "experiment_split", "model_id", "model_revision", "git_commit",
    "config_sha256", "baseline_predictions_sha256", "user_id", "worker_id",
    "seed", "prompt_id", "clean_target", "target_text", "clean_prediction",
    "clean_prediction_reverified", "epsilon_255", "step_size_255", "max_steps",
    "completed_steps", "loss_type", "optimizer", "final_raw_response",
    "final_prediction_normalized", "targeted_success_memory",
    "targeted_success_reloaded", "linf_255", "lpips", "ssim",
    "clean_image_sha256", "attacked_image_sha256", "image_width", "image_height",
    "final_target_loss", "best_target_loss", "gradient_l2_final",
    "gradient_linf_final",
    "early_stop_reason", "checkpoint_resumed", "attack_runtime_seconds",
    "status", "error",
}


def _record_is_final(
    path: Path,
    *,
    config: dict[str, Any],
    config_sha256: str,
    git_commit: str,
    record_id: str,
    epsilon: float,
    worker_id: int,
) -> bool:
    record = _load_json(path)
    if record is None or not DONUT_REQUIRED_RECORD_FIELDS <= set(record):
        return False
    expected = {
        "run_id": config["run_id"],
        "stage": "donut_attack",
        "experiment_key": path.stem,
        "record_id": record_id,
        "experiment_split": config["split"],
        "model_id": config.get("model_id", DEFAULT_MODEL_ID),
        "model_revision": config.get("model_revision", DEFAULT_REVISION),
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "baseline_predictions_sha256": config["baseline_predictions_sha256"],
        "worker_id": worker_id,
        "seed": int(config["seed"]),
        "epsilon_255": float(epsilon),
        "status": "complete",
        "error": None,
    }
    return all(record.get(field) == value for field, value in expected.items())


def merge_if_complete(
    config: dict[str, Any], config_path: str | Path, *, num_workers: int = 5
) -> bool:
    """Strict CPU-safe merge. Return False while worker markers are incomplete."""
    _validate_config(config)
    config_path = Path(config_path).resolve()
    repo_root = config_path.parents[2]
    config_sha256 = _sha256_bytes(config_path.read_bytes())
    baseline = _load_eligible_baseline(config, repo_root)
    expected_record_ids = set(baseline.index)
    epsilons = [float(value) for value in config["attack"]["epsilons_255"]]
    seed = int(config["seed"])
    output_root = repo_root / config["output_root"] / config["run_id"]
    if not output_root.exists():
        return False
    copied_config = output_root / "config.json"
    if not copied_config.is_file() or _sha256_file(copied_config) != config_sha256:
        raise RuntimeError("run directory is missing the exact submitted Donut config")
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
        git_commits: set[str] = set()
        records: dict[str, dict[str, Any]] = {}
        for worker_id in range(num_workers):
            marker = _load_json(markers[worker_id])
            if marker is None:
                raise RuntimeError(f"invalid worker marker: {markers[worker_id]}")
            if (
                marker.get("run_id") != config["run_id"]
                or marker.get("stage") != "donut_attack"
                or marker.get("worker_id") != worker_id
                or marker.get("config_sha256") != config_sha256
                or marker.get("baseline_predictions_sha256")
                != config["baseline_predictions_sha256"]
                or not marker.get("git_commit")
            ):
                raise RuntimeError(f"worker {worker_id} marker provenance mismatch")
            git_commits.add(marker["git_commit"])

            expected_assignment = assigned_record_ids(
                baseline.index, worker_id, num_workers
            )
            path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
            assignment = path.read_text().splitlines() if path.is_file() else []
            if assignment != expected_assignment:
                raise RuntimeError(f"worker {worker_id} assignment is missing or incorrect")
            expected_keys = {
                _safe_key(record_id, epsilon, seed): (record_id, epsilon)
                for record_id in expected_assignment
                for epsilon in epsilons
            }
            if marker.get("expected_records") != len(expected_keys):
                raise RuntimeError(f"worker {worker_id} marker count is incorrect")
            record_dir = output_root / "shards" / f"worker_{worker_id:02d}" / "records"
            actual_paths = {path.stem: path for path in record_dir.glob("*.json")}
            if set(actual_paths) != set(expected_keys):
                raise RuntimeError(f"worker {worker_id} record keys do not match assignment")
            for key, (record_id, epsilon) in expected_keys.items():
                path = actual_paths[key]
                if not _record_is_final(
                    path,
                    config=config,
                    config_sha256=config_sha256,
                    git_commit=marker["git_commit"],
                    record_id=record_id,
                    epsilon=epsilon,
                    worker_id=worker_id,
                ):
                    raise RuntimeError(f"invalid final record during merge: {path}")
                record = _load_json(path)
                numeric_fields = (
                    "linf_255", "lpips", "ssim", "final_target_loss",
                    "best_target_loss", "gradient_l2_final", "gradient_linf_final",
                    "attack_runtime_seconds",
                )
                if any(
                    record.get(field) is None
                    or not math.isfinite(float(record[field]))
                    for field in numeric_fields
                ):
                    raise RuntimeError(f"non-finite metric in final record: {path}")
                if float(record["linf_255"]) > epsilon + 1e-4:
                    raise RuntimeError(f"epsilon violation in final record: {path}")
                image_path = record_dir.parent / "images" / f"{key}.png"
                if (
                    not image_path.is_file()
                    or _sha256_file(image_path) != record.get("attacked_image_sha256")
                ):
                    raise RuntimeError(f"missing or changed attacked image: {image_path}")
                if key in records:
                    raise RuntimeError(f"duplicate experiment key during merge: {key}")
                records[key] = record
        if len(git_commits) != 1:
            raise RuntimeError("Donut workers did not run the same Git commit")
        expected_count = len(expected_record_ids) * len(epsilons)
        if len(records) != expected_count:
            raise RuntimeError(
                f"merge found {len(records)} records but expected {expected_count}"
            )

        import pandas as pd

        fingerprint = _sha256_bytes("".join(
            f"{key}:{_sha256_bytes(json.dumps(records[key], sort_keys=True).encode())}\n"
            for key in sorted(records)
        ).encode())
        merged = output_root / "merged"
        merge_manifest_path = merged / "merge_manifest.json"
        previous = _load_json(merge_manifest_path)
        if previous:
            if (
                previous.get("records_fingerprint") == fingerprint
                and previous.get("config_sha256") == config_sha256
            ):
                return True
            raise RuntimeError("merged Donut outputs already exist for different inputs")
        if merged.exists() and any(merged.iterdir()):
            raise RuntimeError("merged directory exists without matching provenance")

        frame = pd.DataFrame([records[key] for key in sorted(records)]).sort_values(
            ["dataset_name", "record_id", "epsilon_255"]
        )
        merged.mkdir(parents=True, exist_ok=True)
        parquet_tmp = merged / "attacks.parquet.tmp"
        csv_tmp = merged / "attacks.csv.tmp"
        # Nested trace data belongs in JSON records, not the flat analysis table.
        flat = frame.drop(columns=["optimization_trace"], errors="ignore")
        flat.to_parquet(parquet_tmp, index=False)
        flat.to_csv(csv_tmp, index=False)
        os.replace(parquet_tmp, merged / "attacks.parquet")
        os.replace(csv_tmp, merged / "attacks.csv")

        failures: list[dict[str, Any]] = []
        for worker_id in range(num_workers):
            path = output_root / "shards" / f"worker_{worker_id:02d}" / "failures.jsonl"
            if path.is_file():
                failures.extend(
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                )
        failure_frame = pd.DataFrame(
            failures, columns=["experiment_key", "record_id", "error"]
        )
        failures_tmp = merged / "failures.csv.tmp"
        failure_frame.to_csv(failures_tmp, index=False)
        os.replace(failures_tmp, merged / "failures.csv")

        successes = sorted(
            frame.loc[frame["targeted_success_reloaded"].eq(True), "record_id"].unique()
        )
        _atomic_text(
            merged / "successful_attack_ids.txt", "".join(f"{value}\n" for value in successes)
        )
        grouped = (
            frame.groupby(["dataset_name", "epsilon_255"], dropna=False)
            .agg(
                attempted=("record_id", "size"),
                reloaded_successes=("targeted_success_reloaded", "sum"),
                median_lpips=("lpips", "median"),
                median_ssim=("ssim", "median"),
                median_runtime_seconds=("attack_runtime_seconds", "median"),
            )
            .reset_index()
        )
        grouped["conditional_asr"] = (
            grouped["reloaded_successes"] / grouped["attempted"]
        )
        _atomic_json(
            merged / "summary.json",
            {
                "records": len(records),
                "expected_records": expected_count,
                "unique_documents": int(frame["record_id"].nunique()),
                "failures_logged": len(failures),
                "config_sha256": config_sha256,
                "git_commit": next(iter(git_commits)),
                "baseline_predictions_sha256": config["baseline_predictions_sha256"],
                "groups": json.loads(grouped.to_json(orient="records")),
                "merged_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _atomic_json(
            merge_manifest_path,
            {
                "config_sha256": config_sha256,
                "git_commit": next(iter(git_commits)),
                "baseline_predictions_sha256": config["baseline_predictions_sha256"],
                "records": len(records),
                "records_fingerprint": fingerprint,
            },
        )
        return True


def _processor_parity(model: DonutDocVQA, image, device: torch.device) -> dict[str, Any]:
    """Compare the differentiable approximation with the official processor."""
    clean = pil_to_rgb_tensor(image, device=device)
    differentiable = model.preprocess_for_attack(clean).detach()
    official = model.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    if differentiable.shape != official.shape:
        raise RuntimeError(
            f"Donut processor shape mismatch: {tuple(differentiable.shape)} versus "
            f"{tuple(official.shape)}"
        )
    difference = (differentiable - official).abs()
    return {
        "shape": list(differentiable.shape),
        "mean_absolute_error": float(difference.mean()),
        "max_absolute_error": float(difference.max()),
    }


def run(config, config_path, user_id, worker_id, num_workers):
    """Run one deterministic worker shard."""
    _validate_config(config)
    if config["mode"] == "smoke" and num_workers != 1:
        raise ValueError("Donut smoke mode requires exactly one worker")
    if config["mode"] != "smoke" and num_workers != 5:
        raise ValueError("Donut validation and test runs require exactly five workers")
    if not torch.cuda.is_available():
        raise RuntimeError("The real Donut attack requires a CUDA GPU; submit it through CARC Slurm.")

    repo_root = Path(config_path).resolve().parents[2]
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if config.get("require_clean_git", True) and git_status.strip():
        raise RuntimeError(
            "refusing to run from a dirty worktree; commit the reviewed attack and config first"
        )
    config_bytes = Path(config_path).read_bytes()
    config_sha256 = _sha256_bytes(config_bytes)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    baseline = _load_eligible_baseline(config, repo_root)
    assignments = assigned_record_ids(baseline.index, worker_id, num_workers)
    output_root = repo_root / config["output_root"] / config["run_id"]
    shard_root = output_root / "shards" / f"worker_{worker_id:02d}"
    assignment_path = output_root / "assignments" / f"worker_{worker_id:02d}.txt"
    expected_assignment = "".join(f"{record_id}\n" for record_id in assignments)
    if assignment_path.exists() and assignment_path.read_text() != expected_assignment:
        raise RuntimeError("existing worker assignment differs from this frozen run")
    _atomic_text(assignment_path, expected_assignment)

    copied_config = output_root / "config.json"
    if copied_config.exists() and _sha256_file(copied_config) != config_sha256:
        raise RuntimeError("run output already contains a different configuration")
    if not copied_config.exists():
        _atomic_text(copied_config, config_bytes.decode())

    rows = _load_rows(repo_root, config["split"], assignments)
    device = torch.device("cuda")
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    model_revision = config.get("model_revision", DEFAULT_REVISION)
    model = DonutDocVQA(
        device=device,
        model_id=model_id,
        revision=model_revision,
        max_length=int(config.get("max_length", 64)),
    )
    parity = None
    if assignments:
        parity = _processor_parity(model, rows[assignments[0]]["image"].convert("RGB"), device)
        threshold = float(config.get("processor_parity_mean_tolerance", 0.05))
        if parity["mean_absolute_error"] > threshold:
            raise RuntimeError(
                "differentiable Donut preprocessing does not match the official processor: "
                f"mean error {parity['mean_absolute_error']:.6f} > {threshold:.6f}"
            )
    try:
        import lpips

        lpips_model = lpips.LPIPS(net=config.get("lpips_backbone", "alex")).to(device).eval()
    except Exception as exc:
        raise RuntimeError("LPIPS initialization failed; ensure its weights are cached on CARC") from exc

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    try:
        import datasets
        import PIL
        import transformers

        environment = {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "pillow": PIL.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "processor_parity": parity,
        }
        environment_path = output_root / "environment.json"
        if not environment_path.exists():
            _atomic_json(environment_path, environment)
    except Exception as exc:
        raise RuntimeError("failed to record the required run environment") from exc
    attack_config = config["attack"]
    trace_interval = int(attack_config.get("checkpoint_interval", 10))
    failures: list[dict[str, str]] = []
    completed_keys: list[str] = []

    for record_id in assignments:
        row = rows[record_id]
        dataset_name = row["dataset_name"]
        question = config.get(
            "question",
            "What is the final total amount shown on this receipt? Return only the amount, "
            "with no currency symbol, label, or explanation.",
        )
        target_text = row["attack_target"]
        baseline_target = str(baseline.loc[record_id, "attack_target"])
        if str(target_text) != baseline_target:
            raise RuntimeError(f"attack target drift for {record_id}")
        if str(row["clean_target"]) != str(baseline.loc[record_id, "clean_target"]):
            raise RuntimeError(f"clean target drift for {record_id}")
        clean_pil = row["image"].convert("RGB")
        clean = pil_to_rgb_tensor(clean_pil, device=device)
        clean_bytes = clean.detach().mul(255).round().to(torch.uint8).cpu().numpy().tobytes()
        clean_sha256 = _sha256_bytes(clean_bytes)
        clean_prediction_reverified_raw = model.predict_pil(clean_pil, question)
        clean_prediction_reverified = normalize_receipt_total(
            clean_prediction_reverified_raw, dataset_name
        )
        stored_clean_prediction = str(baseline.loc[record_id, "clean_prediction"])
        if clean_prediction_reverified != stored_clean_prediction:
            raise RuntimeError(
                f"clean prediction drift for {record_id}: stored={stored_clean_prediction!r}, "
                f"current={clean_prediction_reverified!r}"
            )

        for epsilon in attack_config["epsilons_255"]:
            epsilon = float(epsilon)
            key = _safe_key(record_id, epsilon, seed)
            record_path = shard_root / "records" / f"{key}.json"
            image_path = shard_root / "images" / f"{key}.png"
            checkpoint_path = shard_root / "checkpoints" / key / "state.pt"
            if _record_is_final(
                record_path,
                config=config,
                config_sha256=config_sha256,
                git_commit=git_commit,
                record_id=record_id,
                epsilon=epsilon,
                worker_id=worker_id,
            ):
                completed_keys.append(key)
                continue

            started = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            trace: list[dict[str, Any]] = []
            resumed = checkpoint_path.is_file()
            start_adversarial = None
            start_step = 0
            optimizer_state = None
            best_loss = float("inf")
            if resumed:
                state = load_checkpoint_state(
                    checkpoint_path,
                    clean_sha256=clean_sha256,
                    config_sha256=config_sha256,
                    model_revision=model_revision,
                    device=device,
                )
                start_adversarial = state["adversarial"]
                start_step = int(state["completed_steps"])
                optimizer_state = state.get("optimizer_state")
                best_loss = float(state["best_loss"])
                trace = list(state.get("extra_state", {}).get("optimization_trace", []))
                torch.set_rng_state(state["torch_rng_state"])
                np.random.set_state(state["numpy_rng_state"])
                random.setstate(state["python_rng_state"])

            def objective(adversarial):
                return model.attack_objective(
                    adversarial,
                    question=question,
                    target_text=target_text,
                    loss_type=attack_config["loss_type"],
                )

            def save_checkpoint(step, adversarial, current_best, loss, metadata, optimizer):
                current_loss, current_metadata = objective(adversarial)
                checkpoint_prediction = model.predict_pil(
                    rgb_tensor_to_pil(adversarial), question
                )
                trace.append(
                    {
                        "step": step,
                        "target_loss": float(current_loss.detach()),
                        "mean_target_log_probability": current_metadata.get(
                            "mean_target_log_probability"
                        ),
                        "prediction": checkpoint_prediction,
                        "exact_target": normalize_receipt_total(
                            checkpoint_prediction, dataset_name
                        )
                        == target_text,
                        "linf_255": _linf_255(clean, adversarial),
                        "gradient_l2": metadata.get("gradient_l2"),
                        "gradient_linf": metadata.get("gradient_linf"),
                    }
                )
                checkpoint_state(
                    checkpoint_path,
                    adversarial=adversarial,
                    clean_sha256=clean_sha256,
                    completed_steps=step,
                    best_loss=min(best_loss, current_best),
                    optimizer=optimizer,
                    config_sha256=config_sha256,
                    model_revision=model_revision,
                    extra_state={"optimization_trace": trace},
                )

            try:
                result = targeted_image_attack(
                    clean,
                    objective,
                    epsilon_255=epsilon,
                    step_size_255=float(attack_config["step_size_255"][str(int(epsilon))]),
                    max_steps=int(attack_config["max_steps"]),
                    optimizer_name=attack_config.get("optimizer", "adam"),
                    quantize_each_step=bool(attack_config.get("quantize_each_step", True)),
                    start_adversarial=start_adversarial,
                    start_step=start_step,
                    start_best_loss=best_loss,
                    optimizer_state=optimizer_state,
                    checkpoint_interval=trace_interval,
                    checkpoint_callback=save_checkpoint,
                    early_stop=False,
                )
                adversarial_pil = rgb_tensor_to_pil(result.adversarial)
                memory_prediction = model.predict_pil(adversarial_pil, question)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_image = image_path.with_suffix(".png.tmp")
                adversarial_pil.save(temporary_image, format="PNG")
                os.replace(temporary_image, image_path)
                from PIL import Image

                with Image.open(image_path) as reopened:
                    reloaded_pil = reopened.convert("RGB")
                    reloaded_prediction = model.predict_pil(reloaded_pil, question)
                    reloaded = pil_to_rgb_tensor(reloaded_pil, device=device)

                observed_linf = _linf_255(clean, reloaded)
                if observed_linf > epsilon + 1e-4:
                    raise RuntimeError(
                        f"saved image violates epsilon: {observed_linf:.6f} > {epsilon}"
                    )
                lpips_value, ssim = _perceptibility(clean, reloaded, lpips_model)
                memory_normalized = normalize_receipt_total(memory_prediction, dataset_name)
                reloaded_normalized = normalize_receipt_total(reloaded_prediction, dataset_name)
                record = {
                    "run_id": config["run_id"],
                    "stage": "donut_attack",
                    "experiment_key": key,
                    "record_id": record_id,
                    "dataset_name": dataset_name,
                    "experiment_split": config["split"],
                    "source_split": row["source_split"],
                    "source_index": row["source_index"],
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "git_commit": git_commit,
                    "config_sha256": config_sha256,
                    "baseline_predictions_sha256": config[
                        "baseline_predictions_sha256"
                    ],
                    "user_id": user_id,
                    "worker_id": worker_id,
                    "seed": seed,
                    "question": question,
                    "prompt_id": config.get("prompt_id", "receipt_total_v1"),
                    "clean_target": row["clean_target"],
                    "target_text": target_text,
                    "clean_prediction": (
                        None
                        if baseline.loc[record_id, "clean_prediction"] is None
                        else str(baseline.loc[record_id, "clean_prediction"])
                    ),
                    "clean_prediction_reverified_raw": clean_prediction_reverified_raw,
                    "clean_prediction_reverified": clean_prediction_reverified,
                    "epsilon_255": epsilon,
                    "step_size_255": float(
                        attack_config["step_size_255"][str(int(epsilon))]
                    ),
                    "max_steps": int(attack_config["max_steps"]),
                    "completed_steps": result.completed_steps,
                    "loss_type": attack_config["loss_type"],
                    "optimizer": attack_config.get("optimizer", "adam"),
                    "quantize_each_step": bool(
                        attack_config.get("quantize_each_step", True)
                    ),
                    "final_raw_response": reloaded_prediction,
                    "final_prediction_normalized": reloaded_normalized,
                    "targeted_success_memory": memory_normalized == target_text,
                    "targeted_success_reloaded": reloaded_normalized == target_text,
                    "linf_255": observed_linf,
                    "lpips": lpips_value,
                    "ssim": ssim,
                    "clean_image_sha256": clean_sha256,
                    "attacked_image_sha256": _sha256_file(image_path),
                    "image_width": reloaded_pil.width,
                    "image_height": reloaded_pil.height,
                    "final_target_loss": (
                        trace[-1]["target_loss"] if trace else result.final_loss
                    ),
                    "best_target_loss": min(
                        [result.best_loss]
                        + [float(item["target_loss"]) for item in trace]
                    ),
                    "gradient_l2_final": (
                        trace[-1].get("gradient_l2") if trace else None
                    ),
                    "gradient_linf_final": (
                        trace[-1].get("gradient_linf") if trace else None
                    ),
                    "early_stop_reason": result.early_stop_reason,
                    "checkpoint_resumed": resumed,
                    "optimization_trace": trace,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "attack_runtime_seconds": time.monotonic() - started,
                    "status": "complete",
                    "error": None,
                }
                _atomic_json(record_path, record)
                completed_keys.append(key)
            except Exception as exc:
                failure = {"experiment_key": key, "record_id": record_id, "error": repr(exc)}
                failures.append(failure)
                failure_path = shard_root / "failures.jsonl"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("a") as handle:
                    handle.write(json.dumps(failure, sort_keys=True) + "\n")

            _atomic_json(
                shard_root / "progress.json",
                {
                    "completed_experiments": len(completed_keys),
                    "failures_this_process": len(failures),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
            )

    expected = len(assignments) * len(attack_config["epsilons_255"])
    expected_records = {
        _safe_key(record_id, float(epsilon), seed): (record_id, float(epsilon))
        for record_id in assignments
        for epsilon in attack_config["epsilons_255"]
    }
    actual_paths = {
        path.stem: path for path in (shard_root / "records").glob("*.json")
    }
    if set(actual_paths) != set(expected_records):
        raise RuntimeError(
            f"worker record keys differ from the expected set ({len(actual_paths)} vs {expected})"
        )
    for key, (record_id, epsilon) in expected_records.items():
        if not _record_is_final(
            actual_paths[key],
            config=config,
            config_sha256=config_sha256,
            git_commit=git_commit,
            record_id=record_id,
            epsilon=epsilon,
            worker_id=worker_id,
        ):
            raise RuntimeError(f"worker has invalid final record: {actual_paths[key]}")
    actual = len(actual_paths)
    _atomic_json(
        shard_root / "WORKER_COMPLETE.json",
        {
            "run_id": config["run_id"],
            "stage": "donut_attack",
            "worker_id": worker_id,
            "user_id": user_id,
            "expected_records": expected,
            "completed_records": actual,
            "config_sha256": config_sha256,
            "git_commit": git_commit,
            "baseline_predictions_sha256": config[
                "baseline_predictions_sha256"
            ],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    merge_if_complete(config, config_path, num_workers=num_workers)
