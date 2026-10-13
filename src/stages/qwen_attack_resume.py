"""Resume-degree track for the shared Qwen adversarial attack core."""
from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checkpoints import atomic_json, atomic_text, sha256_bytes, sha256_file, sha256_text
from data import load_processed_rows, row_images
from models.qwen import DEFAULT_MODEL_ID, DEFAULT_REVISION, QwenVL
from normalization import RESUME_DEGREE_LABELS, normalize_resume_degree
from sharding import assigned_record_ids
from team import get_team_member

PROMPT_ID = "resume_degree_v1"
TASK = "resume_highest_degree"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _validate_attack_config(config: dict[str, Any]) -> None:
    required = {
        "stage", "track", "mode", "run_id", "split", "datasets", "seed",
        "baseline_run_id", "baseline_config", "baseline_config_sha256",
        "baseline_predictions", "baseline_predictions_sha256",
        "baseline_manifest", "baseline_summary", "baseline_correct_ids",
        "baseline_environment", "model_id", "model_revision", "prompt_id",
        "question", "processor_tolerance", "attack",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"resume config missing required fields: {missing}")
    if config.get("stage") != "qwen_attack" or config.get("track") != "resume":
        raise ValueError("resume config requires stage='qwen_attack' and track='resume'")
    if config.get("datasets") != ["resume_parsing_vision"]:
        raise ValueError("resume track datasets must be exactly ['resume_parsing_vision']")
    if config.get("split") not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if config["split"] == "test" and not config.get("frozen_for_test", False):
        raise ValueError("test attacks require frozen_for_test=true")
    if config.get("mode") not in {"smoke", "validation", "test"}:
        raise ValueError("mode must be smoke, validation, or test")
    if config["mode"] == "smoke" and config["split"] != "validation":
        raise ValueError("smoke mode must use the validation split")
    if config["mode"] == "validation" and config["split"] != "validation":
        raise ValueError("validation mode must use the validation split")
    if config["mode"] == "test" and config["split"] != "test":
        raise ValueError("test mode must use the test split")
    if config["mode"] != "smoke" and config.get("ready_for_production") is not True:
        raise ValueError(
            "Qwen resume production is gated; set ready_for_production=true only "
            "after the CARC smoke is reviewed"
        )
    if config["mode"] == "test" and not config.get("validation_run_id"):
        raise ValueError("test mode requires the frozen validation_run_id")
    if config["mode"] == "smoke" and not config.get("owner_user_id"):
        raise ValueError("smoke mode requires owner_user_id")
    if config.get("model_id") != DEFAULT_MODEL_ID or config.get("model_revision") != DEFAULT_REVISION:
        raise ValueError("resume attack must use the pinned Qwen2.5-VL-3B-Instruct revision")
    if config.get("prompt_id") != PROMPT_ID:
        raise ValueError("resume attack prompt_id must match the frozen resume prompt")
    attack = config.get("attack") or {}
    epsilons = attack.get("epsilons_255", [])
    if not epsilons or len(epsilons) != len(set(epsilons)) or any(type(x) is not int or not 0 < x <= 255 for x in epsilons):
        raise ValueError("epsilons_255 must contain unique positive integer pixel budgets")
    if int(attack.get("max_steps", 0)) < 1 or int(attack.get("checkpoint_interval", 0)) < 1:
        raise ValueError("max_steps and checkpoint_interval must be positive")
    if any(float(attack.get("step_size_255", {}).get(str(x), 0)) <= 0 for x in epsilons):
        raise ValueError("step_size_255 must provide a positive step for every epsilon")
    if any(float(attack["step_size_255"][str(x)]) > x for x in epsilons):
        raise ValueError("step_size_255 may not exceed its epsilon")
    if attack.get("loss_type") != "cross_entropy":
        raise ValueError("the shared Qwen attack currently requires loss_type='cross_entropy'")
    if attack.get("multi_page_fallback") not in {"gradient_accumulation_across_pages", "none"}:
        raise ValueError("multi_page_fallback must be gradient_accumulation_across_pages or none")
    tolerance = config.get("processor_tolerance", {})
    if any(float(tolerance.get(field, -1)) < 0 for field in ("mean", "max", "surrogate_mean")):
        raise ValueError(
            "processor_tolerance must define nonnegative mean, max, and surrogate_mean thresholds"
        )
    if float(tolerance["mean"]) > float(tolerance["max"]):
        raise ValueError("processor mean tolerance may not exceed max tolerance")
    for field in ("baseline_config_sha256", "baseline_predictions_sha256"):
        value = config.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{field} must be a lowercase SHA-256 value")


def select_candidates(predictions, manifest, baseline):
    """Verify all frozen resume baseline rows, then return eligible correct candidates."""
    import pandas as pd
    required_prediction = {"record_id", "model_key", "dataset_name", "experiment_split", "model_id",
        "model_revision", "task", "prompt_id", "prompt_text", "clean_target", "attack_target",
        "attack_eligible", "clean_correct", "parse_status", "status", "prediction_normalized",
        "processor_settings", "raw_response", "input_page_count"}
    required_manifest = {"record_id", "dataset_name", "experiment_split", "usable", "attack_eligible",
        "clean_target", "attack_target"}
    missing = sorted(required_prediction - set(predictions.columns))
    if missing:
        raise ValueError(f"baseline predictions missing fields: {missing}")
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"master manifest missing fields: {missing}")
    p = predictions.loc[predictions.model_key.eq("qwen") & predictions.dataset_name.eq("resume_parsing_vision") &
                        predictions.experiment_split.eq(baseline.get("split", "validation"))].copy()
    m = manifest.loc[manifest.dataset_name.eq("resume_parsing_vision") &
                     manifest.experiment_split.eq(baseline.get("split", "validation"))].copy()
    if p.empty or p.record_id.duplicated().any() or m.record_id.duplicated().any():
        raise ValueError("empty or duplicate resume baseline/manifest rows")
    m = m.set_index("record_id", drop=False)
    usable_ids = set(m.loc[m.usable.eq(True), "record_id"])
    if set(p.record_id) != usable_ids:
        raise ValueError("Qwen baseline IDs do not equal usable resume manifest IDs")
    model = baseline["models"]["qwen"]
    prompt = baseline["prompts"]["resume_highest_degree"]
    expected_fields = {"run_id": baseline["run_id"], "model_id": model["model_id"],
        "model_revision": model["revision"], "task": TASK, "prompt_id": prompt["id"], "prompt_text": prompt["text"]}
    for field, expected in expected_fields.items():
        if field not in p or not p[field].eq(expected).all():
            raise ValueError(f"baseline drift in {field}")
    for record in p.to_dict("records"):
        meta = m.loc[record["record_id"]]
        for field in ("clean_target", "attack_target", "attack_eligible"):
            if record[field] != meta[field]:
                raise ValueError(f"baseline/manifest mismatch: {record['record_id']} {field}")
        settings = record["processor_settings"]
        if isinstance(settings, str):
            settings = json.loads(settings)
        for field in ("precision", "min_pixels", "max_pixels", "max_new_tokens"):
            if settings.get(field) != model[field]:
                raise ValueError(f"baseline processor setting drift: {field}")
        if settings.get("do_sample") is not False or settings.get("batch_size") != 1:
            raise ValueError("baseline Qwen generation must be greedy with batch size one")
    chosen = p.loc[p.clean_correct.eq(True) & p.attack_eligible.eq(True) & p.status.eq("complete") &
                   p.parse_status.eq("ok")].copy()
    for record in chosen.to_dict("records"):
        if record["prediction_normalized"] != record["clean_target"]:
            raise ValueError(f"clean_correct contradicts prediction for {record['record_id']}")
        if record["clean_target"] not in RESUME_DEGREE_LABELS or record["attack_target"] not in RESUME_DEGREE_LABELS:
            raise ValueError(f"noncanonical degree target for {record['record_id']}")
        if record["attack_target"] == record["clean_target"]:
            raise ValueError(f"attack target equals clean target for {record['record_id']}")
    return chosen.sort_values("record_id").set_index("record_id", drop=False)


def select_smoke_candidates(candidates, count_per_page_group: int):
    """Choose deterministic single- and multi-page smoke rows with an unambiguous index."""
    import pandas as pd
    n = int(count_per_page_group)
    one = candidates.loc[candidates.input_page_count.eq(1)].head(n)
    many = candidates.loc[candidates.input_page_count.gt(1)].head(n)
    if n < 1 or len(one) != n or len(many) != n:
        raise ValueError(
            f"smoke needs {n} single-page and {n} multi-page candidates; "
            f"found {len(one)} and {len(many)}"
        )
    # select_candidates intentionally retains record_id as both index and column.
    # Drop the inherited index before sorting so pandas never has to guess which
    # record_id the operation refers to, then restore the runtime index contract.
    return (pd.concat([one, many], ignore_index=True)
              .sort_values("record_id")
              .set_index("record_id", drop=False))


def _load_plan(config: dict[str, Any], repo_root: Path):
    import pandas as pd
    required = ("baseline_config", "baseline_predictions", "baseline_manifest", "baseline_summary",
                "baseline_correct_ids", "baseline_environment")
    missing_cfg = [key for key in required if not config.get(key)]
    if missing_cfg:
        raise ValueError(f"resume config missing baseline artifact paths: {missing_cfg}")
    paths = {key: _path(repo_root, config[key]) for key in required}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    for extra in (repo_root / "data/processed/master_manifest.parquet", repo_root / "data/processed/resume_parsing_vision"):
        if not extra.exists():
            missing_files.append(str(extra))
    if missing_files:
        raise FileNotFoundError("missing frozen CARC artifacts:\n" + "\n".join(missing_files))
    if sha256_file(paths["baseline_config"]) != config.get("baseline_config_sha256"):
        frozen_copy = _path(repo_root, config.get("baseline_frozen_config", "outputs/baseline/baseline_validation_v1/config.json"))
        if not frozen_copy.is_file() or sha256_file(frozen_copy) != config.get("baseline_config_sha256"):
            raise ValueError(f"frozen baseline config SHA-256 mismatch: {paths['baseline_config']}")
        paths["baseline_config"] = frozen_copy
    if sha256_file(paths["baseline_predictions"]) != config.get("baseline_predictions_sha256"):
        raise ValueError(f"frozen SHA-256 mismatch for {paths['baseline_predictions']}")
    baseline = _read_json(paths["baseline_config"])
    if baseline.get("run_id") != config.get("baseline_run_id"):
        raise ValueError("frozen baseline run_id differs from baseline_run_id")
    frozen_prompt = baseline.get("prompts", {}).get("resume_highest_degree", {})
    if config.get("question") != frozen_prompt.get("text") or config.get("prompt_id") != frozen_prompt.get("id"):
        raise ValueError("resume prompt differs from the frozen baseline prompt")
    manifest_meta = _read_json(paths["baseline_manifest"])
    summary = _read_json(paths["baseline_summary"])
    if manifest_meta.get("config_sha256") != config["baseline_config_sha256"]:
        raise ValueError("baseline merge/config hash mismatch")
    predictions = pd.read_parquet(paths["baseline_predictions"])
    if len(predictions) != summary.get("records") or len(predictions) != manifest_meta.get("records"):
        raise ValueError("baseline prediction row count differs from merge metadata")
    if not predictions.config_sha256.eq(manifest_meta["config_sha256"]).all():
        raise ValueError("baseline rows have a different config hash")
    if not predictions.git_commit.eq(manifest_meta["git_commit"]).all():
        raise ValueError("baseline rows have mixed Git commits")
    correct_path = paths["baseline_correct_ids"]
    saved_ids = correct_path.read_text(encoding="utf-8").splitlines()
    actual_ids = predictions.loc[predictions.model_key.eq("qwen") & predictions.clean_correct.eq(True), "record_id"].tolist()
    if sorted(saved_ids) != sorted(actual_ids):
        raise ValueError("saved Qwen clean-correct IDs disagree with frozen predictions")
    manifest = pd.read_parquet(repo_root / "data/processed/master_manifest.parquet")
    baseline["split"] = config["split"]
    candidates = select_candidates(predictions, manifest, baseline)
    if config["split"] == "validation" and len(candidates) != int(config.get("expected_validation_candidates", 74)):
        raise ValueError(f"accepted baseline should have {config.get('expected_validation_candidates', 74)} resume candidates; found {len(candidates)}")
    if config["mode"] == "smoke":
        candidates = select_smoke_candidates(
            candidates, int(config.get("smoke_per_page_group", 5))
        )
    return baseline, candidates, paths


def preflight(config, repo_root):
    """Perform CPU-only frozen-artifact checks and print exact selected records."""
    _validate_attack_config(config)
    root = Path(repo_root).resolve()
    baseline, candidates, _ = _load_plan(config, root)
    print(f"Selected {len(candidates)} Qwen clean-correct attack candidates:")
    for rid, row in candidates.iterrows():
        print(f"  {rid} | clean={row.clean_target} | target={row.attack_target}")
    return candidates


def _record_key(record_id: str, epsilon: int, seed: int) -> str:
    safe = "_".join(record_id.split("/"))
    return f"{safe}__qwen__resume_highest_degree__pgd__eps-{epsilon}__seed-{seed}"


def _immutable(path: Path, value: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise RuntimeError(f"existing run metadata differs: {path}")
    if not path.exists():
        atomic_text(path, value)


def _runtime_versions() -> dict[str, str]:
    import datasets
    import PIL
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "pillow": PIL.__version__,
    }


def _validate_environment(versions: dict[str, str], baseline_environment: Path) -> None:
    frozen = _read_json(baseline_environment)
    for key, version in versions.items():
        if frozen.get(key) != version:
            raise RuntimeError(
                f"baseline environment drift: {key} current={version}, "
                f"frozen={frozen.get(key)}"
            )


def _build_provenance(
    config: dict[str, Any], config_path: Path, root: Path, versions: dict[str, str]
) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "run_id": config["run_id"],
        "stage": "qwen_attack",
        "track": "resume",
        "config_sha256": sha256_file(config_path),
        "git_commit": commit,
        "baseline_config_sha256": config["baseline_config_sha256"],
        "baseline_predictions_sha256": config["baseline_predictions_sha256"],
        "processed_manifest_sha256": sha256_file(
            root / "data/processed/master_manifest.parquet"
        ),
        "environment_hash": sha256_text(json.dumps(versions, sort_keys=True)),
    }


def _validate_saved_record(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    record = _read_json(path)
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"final record provenance mismatch in {key}: {path}")
    if record.get("status") != "complete":
        raise RuntimeError(f"final record is not complete: {path}")
    numeric_fields = (
        "linf_255", "lpips", "ssim", "initial_loss", "final_loss", "best_loss",
        "mean_target_log_probability", "gradient_norm_final", "runtime_seconds",
        "peak_gpu_memory_mb",
    )
    for key in numeric_fields:
        if record.get(key) is None or not math.isfinite(float(record[key])):
            raise RuntimeError(f"nonfinite final metric {key}: {path}")
    if not 0 <= record["linf_255"] <= record["epsilon_255"] + 1e-4 or not -1 <= record["ssim"] <= 1:
        raise RuntimeError(f"out-of-range final metric: {path}")
    if record.get("targeted_success_reloaded") != record.get("saved_image_success"):
        raise RuntimeError(f"saved-image success fields disagree: {path}")
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"final record has no saved page images: {path}")
    expected_image_dir = (path.parent.parent / "images" / path.stem).resolve()
    if record.get("document_page_count") is not None and len(images) != int(record["document_page_count"]):
        raise RuntimeError(f"saved page count differs from document page count: {path}")
    for page_index, image in enumerate(images):
        image_path = Path(image["path"])
        if image_path.resolve().parent != expected_image_dir:
            raise RuntimeError(f"saved attack image is outside its experiment directory: {image_path}")
        if image.get("page") != page_index:
            raise RuntimeError(f"saved attack page order is invalid: {image_path}")
        if not image_path.is_file() or sha256_file(image_path) != image.get("sha256"):
            raise RuntimeError(f"saved attack image is missing or changed: {image_path}")
        for metric in ("linf_255", "lpips", "ssim"):
            if image.get(metric) is None or not math.isfinite(float(image[metric])):
                raise RuntimeError(f"nonfinite page metric {metric}: {image_path}")
        if not 0 <= float(image["linf_255"]) <= float(record["epsilon_255"]) + 1e-4:
            raise RuntimeError(f"page exceeds epsilon bound: {image_path}")
        if not -1 <= float(image["ssim"]) <= 1:
            raise RuntimeError(f"page SSIM is out of range: {image_path}")
    if record.get("attacked_image_sha256") != [image["sha256"] for image in images]:
        raise RuntimeError(f"record image-hash list disagrees with page metadata: {path}")
    trace = record.get("optimization_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"final record has no optimization trace: {path}")
    for point in trace:
        for key in ("target_loss", "mean_target_log_probability", "linf_255", "gradient_norm", "peak_gpu_memory_mb"):
            if point.get(key) is None or not math.isfinite(float(point[key])):
                raise RuntimeError(f"nonfinite trace field {key}: {path}")
        if not 0 <= float(point["linf_255"]) <= float(record["epsilon_255"]) + 1e-4:
            raise RuntimeError(f"trace exceeds epsilon bound: {path}")
    return True


def _merge_if_complete(output: Path, *, expected_ids: set[str], epsilons: list[int],
                       num_workers: int, provenance: dict[str, Any]) -> bool:
    markers = [output / "shards" / f"worker_{i:02d}" / "WORKER_COMPLETE.json" for i in range(num_workers)]
    if not all(marker.is_file() for marker in markers):
        return False
    import fcntl
    import pandas as pd
    lock_path = output / ".merge.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assigned = []
        for worker_id, marker_path in enumerate(markers):
            marker = _read_json(marker_path)
            if marker.get("worker_id") != worker_id or any(marker.get(key) != value for key, value in provenance.items()):
                raise RuntimeError(f"worker {worker_id} completion provenance mismatch")
            path = output / "assignments" / f"worker_{worker_id:02d}.txt"
            ids = path.read_text(encoding="utf-8").splitlines()
            expected_assignment = assigned_record_ids(expected_ids, worker_id, num_workers)
            if ids != expected_assignment or marker.get("record_ids") != ids:
                raise RuntimeError(f"worker {worker_id} assignment/marker mismatch")
            expected_worker_count = len(ids) * len(epsilons)
            if marker.get("num_experiments") != expected_worker_count:
                raise RuntimeError(f"worker {worker_id} completion count mismatch")
            assigned.extend(ids)
        if len(assigned) != len(set(assigned)):
            raise RuntimeError("worker assignments overlap; refusing merge")
        if set(assigned) != expected_ids:
            raise RuntimeError("worker assignments do not cover the expected candidate IDs")
        records, keys = [], set()
        for worker_id in range(num_workers):
            directory = output / "shards" / f"worker_{worker_id:02d}" / "records"
            files = sorted(directory.glob("*.json"))
            expected_count = len((output / "assignments" / f"worker_{worker_id:02d}.txt").read_text().splitlines()) * len(epsilons)
            if len(files) != expected_count:
                raise RuntimeError(f"worker {worker_id} has missing record keys")
            for path in files:
                record = _read_json(path)
                required_fields = {"run_id", "stage", "track", "experiment_key", "record_id", "dataset_name",
                    "experiment_split", "model_key", "model_id", "model_revision", "task", "prompt_id",
                    "prompt_sha256", "clean_target", "attack_target", "attack_eligible", "clean_correct",
                    "epsilon_255", "targeted_success_reloaded", "linf_255", "lpips", "ssim",
                    "target_token_ids_hash", "official_processor_shape", "differentiable_processor_shape",
                    "processor_max_abs_difference", "processor_mean_abs_difference", "mean_target_log_probability",
                    "processor_surrogate_max_abs_difference", "processor_surrogate_mean_abs_difference",
                    "processor_gradient_method",
                    "gradient_norm_final", "peak_gpu_memory_mb", "worker_id", "seed", "git_commit",
                    "config_sha256", "document_page_count", "attacked_page_count", "status", "error"}
                missing_fields = required_fields - record.keys()
                if missing_fields:
                    raise RuntimeError(f"record {path} missing fields: {sorted(missing_fields)}")
                if record.get("worker_id") != worker_id or any(record.get(key) != value for key, value in provenance.items()):
                    raise RuntimeError(f"record provenance mismatch: {path}")
                if record["record_id"] not in expected_ids or record["epsilon_255"] not in epsilons:
                    raise RuntimeError(f"unexpected record key: {path}")
                expected_record = {**provenance, "experiment_key": path.stem,
                    "record_id": record["record_id"], "epsilon_255": record["epsilon_255"],
                    "worker_id": worker_id}
                if not _validate_saved_record(path, expected_record):
                    raise RuntimeError(f"invalid final record during merge: {path}")
                key = (record["record_id"], record["epsilon_255"])
                if key in keys:
                    raise RuntimeError(f"duplicate record key: {key}")
                keys.add(key)
                records.append(record)
        expected_keys = {(rid, eps) for rid in expected_ids for eps in epsilons}
        if keys != expected_keys:
            raise RuntimeError("merged experiment keys do not equal the expected set")
        fingerprint = sha256_text("".join(
            f"{rid}:{eps}:{sha256_text(json.dumps(next(r for r in records if r['record_id'] == rid and r['epsilon_255'] == eps), sort_keys=True, default=str))}\n"
            for rid, eps in sorted(expected_keys)
        ))
        merged = output / "merged"
        manifest_path = merged / "merge_manifest.json"
        if manifest_path.is_file():
            previous = _read_json(manifest_path)
            if previous.get("records_fingerprint") == fingerprint and all(previous.get(k) == v for k, v in provenance.items()):
                return True
            raise RuntimeError("merged outputs already exist for different inputs")
        if merged.exists() and any(merged.iterdir()):
            raise RuntimeError("merged directory exists without matching merge provenance")
        merged.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(records).sort_values(["dataset_name", "record_id", "epsilon_255"])
        flat = frame.drop(columns=["optimization_trace", "images"], errors="ignore")
        tmp = merged / "attacks.parquet.tmp"
        flat.to_parquet(tmp, index=False)
        os.replace(tmp, merged / "attacks.parquet")
        tmp = merged / "attacks.csv.tmp"
        flat.to_csv(tmp, index=False)
        os.replace(tmp, merged / "attacks.csv")
        page_rows = []
        for rec in records:
            for page in rec["images"]:
                page_rows.append({"run_id": rec["run_id"], "record_id": rec["record_id"],
                    "epsilon_255": rec["epsilon_255"], **page})
        page_frame = pd.DataFrame(page_rows)
        tmp = merged / "page_metrics.parquet.tmp"
        page_frame.to_parquet(tmp, index=False)
        os.replace(tmp, merged / "page_metrics.parquet")
        failures = []
        for worker_id in range(num_workers):
            failure_dir = output / "shards" / f"worker_{worker_id:02d}" / "failures"
            failures.extend(_read_json(path) for path in failure_dir.glob("*.json"))
        tmp = merged / "failures.csv.tmp"
        pd.DataFrame(failures).to_csv(tmp, index=False)
        os.replace(tmp, merged / "failures.csv")
        successes = sorted(frame.loc[frame.saved_image_success.eq(True), "record_id"].unique())
        atomic_text(merged / "successful_attack_ids.txt", "".join(f"{rid}\n" for rid in successes))
        groups = frame.groupby(["dataset_name", "epsilon_255"], dropna=False).agg(
            attempted=("record_id", "size"), reloaded_successes=("saved_image_success", "sum"),
            median_lpips=("lpips", "median"), median_ssim=("ssim", "median"),
            median_runtime_seconds=("runtime_seconds", "median")).reset_index()
        groups["conditional_asr"] = groups.reloaded_successes / groups.attempted
        atomic_json(merged / "summary.json", {"records": len(records), "unique_documents": int(frame.record_id.nunique()),
            "failures_logged": len(failures), "groups": json.loads(groups.to_json(orient="records")), "merged_at": _now()})
        atomic_json(manifest_path, {**provenance, "records": len(records),
            "records_fingerprint": fingerprint, "merged_at": _now()})
        return True


def merge_if_complete(
    config: dict[str, Any], config_path: str | Path, *, num_workers: int | None = None
) -> bool:
    """Re-run the strict Qwen-resume merge safely from a CARC CPU login node."""
    _validate_attack_config(config)
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    _, selected, paths = _load_plan(config, root)
    versions = _runtime_versions()
    _validate_environment(versions, paths["baseline_environment"])
    provenance = _build_provenance(config, config_path, root, versions)
    if num_workers is None:
        num_workers = 1 if config["mode"] == "smoke" else 5
    expected_workers = 1 if config["mode"] == "smoke" else 5
    if num_workers != expected_workers:
        raise ValueError(
            f"mode={config['mode']!r} requires num_workers={expected_workers}"
        )
    output = root / "outputs/qwen_attack" / config["run_id"]
    return _merge_if_complete(
        output,
        expected_ids=set(selected.index),
        epsilons=config["attack"]["epsilons_255"],
        num_workers=num_workers,
        provenance=provenance,
    )


def run(config, config_path, user_id, worker_id, num_workers):
    _validate_attack_config(config)
    expected_workers = 1 if config["mode"] == "smoke" else 5
    if num_workers != expected_workers:
        raise ValueError(f"mode={config['mode']!r} requires num_workers={expected_workers}")
    member = get_team_member(user_id)
    if num_workers == 5 and member.worker_id != worker_id:
        raise ValueError(f"{user_id} must use worker {member.worker_id}, not {worker_id}")
    if num_workers == 1 and (worker_id != 0 or user_id != config.get("owner_user_id")):
        raise ValueError("one-worker smoke execution requires configured owner_user_id and worker 0")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Qwen attacks must run inside a CARC Slurm GPU job")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen attacks require a CUDA GPU")
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    baseline, selected, paths = _load_plan(config, root)
    if config.get("require_clean_git", True):
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        if status.strip():
            raise RuntimeError("refusing to run from a dirty git worktree")
    import lpips
    versions = _runtime_versions()
    _validate_environment(versions, paths["baseline_environment"])
    provenance = _build_provenance(config, config_path, root, versions)
    ids = selected.index.tolist() if num_workers == 1 else assigned_record_ids(selected.index, worker_id, num_workers)
    output = root / "outputs/qwen_attack" / config["run_id"]
    shard = output / "shards" / f"worker_{worker_id:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (shard / ".worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable(output / "config.json", config_path.read_text(encoding="utf-8"))
        assignment_path = output / "assignments" / f"worker_{worker_id:02d}.txt"
        _immutable(assignment_path, "".join(rid + "\n" for rid in ids))
        environment = {**versions, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"], **provenance}
        environment_path = shard / "environment.json"
        if environment_path.exists():
            previous_environment = _read_json(environment_path)
            for field, value in {**versions, **provenance}.items():
                if previous_environment.get(field) != value:
                    raise RuntimeError(f"worker environment/provenance changed in {field}")
        else:
            atomic_json(environment_path, environment)
        atomic_json(shard / "progress.json", {"assigned_records": len(ids), "started_at": _now()})
        import random
        import numpy as np
        random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        rows = load_processed_rows(root, config["split"], ids)
        wrapper = metric = None
        failures = []
        try:
            for rid in ids:
                selected_row = selected.loc[rid].to_dict()
                row = rows[rid]
                for field in ("clean_target", "attack_target", "usable", "attack_eligible"):
                    expected_value = True if field in {"usable", "attack_eligible"} else selected_row[field]
                    if row[field] != expected_value:
                        raise RuntimeError(f"processed row differs from frozen manifest: {rid} {field}")
                pages = [image.convert("RGB") for image in row_images(row)]
                if len(pages) != int(selected_row["input_page_count"]):
                    raise RuntimeError(f"page count drift for {rid}")
                for epsilon in config["attack"]["epsilons_255"]:
                    key = _record_key(rid, epsilon, config["seed"])
                    expected_record = {**provenance, "experiment_key": key, "record_id": rid,
                        "epsilon_255": epsilon, "worker_id": worker_id, "run_id": config["run_id"],
                        "source_image_hashes": [sha256_bytes(image.tobytes()) for image in pages]}
                    record_path = shard / "records" / f"{key}.json"
                    if _validate_saved_record(record_path, expected_record):
                        continue
                    if wrapper is None:
                        wrapper = QwenVL(device=torch.device("cuda"), **baseline["models"]["qwen"])
                        metric = lpips.LPIPS(net="alex").to(wrapper.device).eval()
                        metric.requires_grad_(False)
                    started = _now()
                    try:
                        from attacks.qwen import attack
                        result = attack(wrapper, pages, selected_row, config, epsilon, shard, key,
                                        provenance, metric, normalize_resume_degree)
                    except Exception as exc:
                        failure = {**expected_record, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "ended_at": _now()}
                        atomic_json(shard / "failures" / f"{key}.json", failure)
                        failures.append(failure)
                        raise
                    record = {**expected_record, **result, "stage": "qwen_attack", "track": "resume",
                        "mode": config["mode"], "model_key": "qwen", "model_id": wrapper.model_id,
                        "model_revision": wrapper.revision, "dataset_name": "resume_parsing_vision",
                        "experiment_split": config["split"], "source_split": selected_row["source_split"],
                        "source_index": int(selected_row["source_index"]), "task": selected_row["task"],
                        "clean_target": selected_row["clean_target"], "attack_target": selected_row["attack_target"],
                        "target_text": selected_row["attack_target"],
                        "usable": True, "attack_eligible": True, "clean_correct": True,
                        "clean_prediction": selected_row["prediction_normalized"],
                        "prompt_id": selected_row["prompt_id"], "prompt_text": selected_row["prompt_text"],
                        "question": selected_row["prompt_text"],
                        "prompt_sha256": sha256_text(selected_row["prompt_text"]),
                        "processor_settings": wrapper.settings, "document_page_count": len(pages),
                        "attacked_page_count": len(pages), "input_page_count": len(pages),
                        "seed": int(config["seed"]), "user_id": user_id,
                        "epsilon_255": epsilon,
                        "step_size_255": float(config["attack"]["step_size_255"][str(int(epsilon))]),
                        "max_steps": int(config["attack"]["max_steps"]),
                        "loss_type": "cross_entropy", "optimizer": "sign",
                        "quantize_each_step": True,
                        "final_raw_response": result["raw_response"],
                        "raw_response": result["raw_response"],
                        "prediction_normalized": result["prediction_normalized"],
                        "parse_status": "ok" if result["prediction_normalized"] is not None else "failed",
                        "final_prediction_normalized": result["prediction_normalized"],
                        "targeted_success_memory": result["in_memory_success"],
                        "targeted_success_reloaded": result["saved_image_success"],
                        "clean_image_sha256": sha256_text(json.dumps(expected_record["source_image_hashes"])),
                        "attacked_image_sha256": [image["sha256"] for image in result["images"]],
                        "image_widths": [image["width"] for image in result["images"]],
                        "image_heights": [image["height"] for image in result["images"]],
                        "final_target_loss": result["final_loss"],
                        "best_target_loss": result["best_loss"],
                        "checkpoint_resumed": result["checkpoint_resumed"],
                        "attack_runtime_seconds": result["runtime_seconds"],
                        "started_at": started, "ended_at": _now(), "status": "complete", "error": None,
                        "attack_settings": config["attack"]}
                    atomic_json(record_path, record)
                    _validate_saved_record(record_path, expected_record)
                    atomic_json(shard / "progress.json", {"last_completed": key, "updated_at": _now(),
                        "completed_experiments": len(list((shard / "records").glob("*.json")))})
        finally:
            if wrapper is not None:
                wrapper.unload()
        for rid in ids:
            for epsilon in config["attack"]["epsilons_255"]:
                key = _record_key(rid, epsilon, config["seed"])
                expected = {**provenance, "record_id": rid, "experiment_key": key,
                    "epsilon_255": epsilon, "worker_id": worker_id, "run_id": config["run_id"]}
                if not _validate_saved_record(shard / "records" / f"{key}.json", expected):
                    raise RuntimeError(f"missing final attack record {key}")
        atomic_json(shard / "WORKER_COMPLETE.json", {**provenance, "worker_id": worker_id,
            "record_ids": ids, "completed_at": _now(), "num_experiments": len(ids) * len(config["attack"]["epsilons_255"])})
    _merge_if_complete(output, expected_ids=set(selected.index), epsilons=config["attack"]["epsilons_255"],
                       num_workers=num_workers, provenance=provenance)
    if failures:
        raise RuntimeError(f"{len(failures)} attack experiment(s) failed; see {shard / 'failures'}")
