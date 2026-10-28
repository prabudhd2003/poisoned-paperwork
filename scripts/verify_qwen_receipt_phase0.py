#!/usr/bin/env python3
"""Verify the frozen inputs for the Qwen receipt attack without loading a model.

This is a Phase-0 integrity check.  It reads only JSON, text, and Parquet
artifacts produced by the clean baseline and the processed-data pipeline.  It
does not import torch, initialize CUDA, download Qwen, or run inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from checkpoints import sha256_file, sha256_text  # noqa: E402
from normalization import normalize_receipt_total  # noqa: E402


RECEIPT_DATASETS = ("sroie", "cord_v2")
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
RECEIPT_TASK = "receipt_total"
RECEIPT_PROMPT_ID = "receipt_total_v1"
RECEIPT_PROMPT_TEXT = (
    "What is the final total amount shown on this receipt? Return only the "
    "amount, with no currency symbol, label, or explanation."
)
EXPECTED_BASELINE_RECORDS = 527
EXPECTED_BASELINE_DOCUMENTS = 301
EXPECTED_BASELINE_GROUP_COUNTS = {
    ("qwen", "sroie"): 126,
    ("qwen", "cord_v2"): 100,
    ("qwen", "resume_parsing_vision"): 75,
    ("donut", "sroie"): 126,
    ("donut", "cord_v2"): 100,
}

DEFAULT_PATHS = {
    "reference_config": "configs/qwen_attack/resume_active.json",
    "baseline_config": "configs/baseline/active.json",
    "baseline_frozen_config": "outputs/baseline/baseline_validation_v1/config.json",
    "baseline_predictions": "outputs/baseline/baseline_validation_v1/merged/predictions.parquet",
    "baseline_merge_manifest": "outputs/baseline/baseline_validation_v1/merged/merge_manifest.json",
    "baseline_summary": "outputs/baseline/baseline_validation_v1/merged/summary.json",
    "baseline_correct_ids": "outputs/baseline/baseline_validation_v1/merged/qwen_clean_correct_ids.txt",
    "baseline_environment": "outputs/baseline/baseline_validation_v1/shards/worker_00/environment.json",
    "master_manifest": "data/processed/master_manifest.parquet",
}


class VerificationError(RuntimeError):
    """Raised when a frozen artifact or receipt-candidate invariant fails."""


def _scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalars to stable Python values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object in {path}")
    return value


def _require_columns(frame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise VerificationError(f"{label} is missing columns: {missing}")


def _require_text(value: Any, *, field: str, record_id: str) -> str:
    value = _scalar(value)
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{record_id}: {field} must be a nonempty string")
    return value


def canonical_receipt_amount(
    value: Any, dataset_name: str, *, field: str, record_id: str
) -> tuple[str, Decimal]:
    """Return a strict canonical amount and its exact Decimal value.

    The project normalizer deliberately rejects negative receipt outputs.  We
    still recognize a strictly formatted leading-minus value here so Phase 0
    can classify it as a non-positive policy exclusion instead of hiding it as
    a parsing error.  Positive values must round-trip through the production
    normalizer exactly.
    """
    if dataset_name not in RECEIPT_DATASETS:
        raise VerificationError(f"{record_id}: unsupported receipt dataset {dataset_name!r}")
    text = _require_text(value, field=field, record_id=record_id)
    negative = text.startswith("-")
    magnitude = text[1:] if negative else text
    normalized = normalize_receipt_total(magnitude, dataset_name)
    if normalized is None:
        raise VerificationError(
            f"{record_id}: {field}={text!r} is not a parseable {dataset_name} amount"
        )
    try:
        numeric = Decimal(text)
    except InvalidOperation as exc:
        raise VerificationError(
            f"{record_id}: {field}={text!r} is not an exact decimal"
        ) from exc
    expected = f"-{normalized}" if negative and numeric != 0 else normalized
    if text != expected:
        raise VerificationError(
            f"{record_id}: {field}={text!r} is not canonical; expected {expected!r}"
        )
    return text, numeric


def evaluate_target_policy(
    *, record_id: str, dataset_name: str, clean_target: Any, attack_target: Any
) -> dict[str, Any]:
    """Evaluate the proposed positive, distinct, exact-ten-times policy."""
    clean_text, clean_value = canonical_receipt_amount(
        clean_target, dataset_name, field="clean_target", record_id=record_id
    )
    attack_text, attack_value = canonical_receipt_amount(
        attack_target, dataset_name, field="attack_target", record_id=record_id
    )
    ten_x_exact = attack_value == clean_value * Decimal(10)
    reasons: list[str] = []
    if clean_value <= 0:
        reasons.append("non_positive_clean_total")
    if attack_value == clean_value:
        reasons.append("equal_target")

    # Equal-target rows are intentionally reported as policy exclusions.  For
    # every other row, a non-10x target is provenance/data corruption.
    if not ten_x_exact and "equal_target" not in reasons:
        raise VerificationError(
            f"{record_id}: attack_target={attack_text!r} is not exactly 10x "
            f"clean_target={clean_text!r}"
        )
    return {
        "clean_target": clean_text,
        "attack_target": attack_text,
        "ten_x_exact": ten_x_exact,
        "eligible_under_policy": not reasons and ten_x_exact,
        "reasons": reasons,
    }


def _parse_processor_settings(value: Any, record_id: str) -> dict[str, Any]:
    value = _scalar(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                f"{record_id}: processor_settings is invalid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{record_id}: processor_settings is not an object")
    return value


def _counts_by_dataset(values) -> dict[str, int]:
    counts = Counter(values)
    result = {dataset: int(counts.get(dataset, 0)) for dataset in RECEIPT_DATASETS}
    result["total"] = sum(result.values())
    return result


def analyze_receipt_candidates(predictions, manifest, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Verify Qwen receipt coverage and compute the Phase-0 candidate report."""
    prediction_fields = {
        "record_id", "run_id", "stage", "model_key", "dataset_name",
        "experiment_split", "model_id", "model_revision", "task", "prompt_id",
        "prompt_text", "prompt_sha256", "clean_target", "attack_target", "attack_eligible",
        "clean_correct", "parse_status", "status", "prediction_normalized",
        "processor_settings", "input_page_count",
    }
    manifest_fields = {
        "record_id", "dataset_name", "experiment_split", "task", "usable",
        "attack_eligible", "clean_target", "attack_target", "num_pages",
    }
    _require_columns(predictions, prediction_fields, "baseline predictions")
    _require_columns(manifest, manifest_fields, "master manifest")

    split = baseline.get("split")
    if split != "validation":
        raise VerificationError(f"Phase 0 requires baseline split='validation'; found {split!r}")
    receipt_manifest = manifest.loc[
        manifest["dataset_name"].isin(RECEIPT_DATASETS)
        & manifest["experiment_split"].eq(split)
        & manifest["usable"].eq(True)
    ].copy()
    receipt_predictions = predictions.loc[
        predictions["model_key"].eq("qwen")
        & predictions["dataset_name"].isin(RECEIPT_DATASETS)
        & predictions["experiment_split"].eq(split)
    ].copy()
    if receipt_manifest.empty or receipt_predictions.empty:
        raise VerificationError("usable manifest or Qwen receipt baseline selection is empty")
    for frame, label in (
        (receipt_manifest, "usable receipt manifest"),
        (receipt_predictions, "Qwen receipt baseline"),
    ):
        duplicates = sorted(frame.loc[frame["record_id"].duplicated(), "record_id"].astype(str).unique())
        if duplicates:
            raise VerificationError(f"{label} has duplicate record IDs: {duplicates[:10]}")

    manifest_ids = set(receipt_manifest["record_id"])
    prediction_ids = set(receipt_predictions["record_id"])
    if manifest_ids != prediction_ids:
        missing = sorted(manifest_ids - prediction_ids)
        extra = sorted(prediction_ids - manifest_ids)
        raise VerificationError(
            "Qwen receipt baseline coverage differs from usable validation manifest: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    model = baseline.get("models", {}).get("qwen", {})
    prompt = baseline.get("prompts", {}).get(RECEIPT_TASK, {})
    expected_fields = {
        "run_id": baseline.get("run_id"),
        "stage": "baseline",
        "model_id": QWEN_MODEL_ID,
        "model_revision": QWEN_MODEL_REVISION,
        "task": RECEIPT_TASK,
        "prompt_id": RECEIPT_PROMPT_ID,
        "prompt_text": RECEIPT_PROMPT_TEXT,
        "prompt_sha256": sha256_text(RECEIPT_PROMPT_TEXT),
    }
    manifest_by_id = receipt_manifest.set_index("record_id", drop=False)
    for record in receipt_predictions.to_dict("records"):
        record_id = str(record["record_id"])
        for field, expected in expected_fields.items():
            if _scalar(record.get(field)) != expected:
                raise VerificationError(
                    f"{record_id}: baseline drift in {field}; "
                    f"found={_scalar(record.get(field))!r} expected={expected!r}"
                )
        row = manifest_by_id.loc[record["record_id"]]
        for field in ("dataset_name", "experiment_split", "task"):
            if _scalar(record[field]) != _scalar(row[field]):
                raise VerificationError(f"{record_id}: baseline/manifest mismatch in {field}")
        if _scalar(row["task"]) != RECEIPT_TASK:
            raise VerificationError(f"{record_id}: manifest task is not {RECEIPT_TASK!r}")
        for field in ("clean_target", "attack_target", "attack_eligible"):
            if _scalar(record[field]) != _scalar(row[field]):
                raise VerificationError(f"{record_id}: baseline/manifest mismatch in {field}")
        input_page_count = _scalar(record["input_page_count"])
        manifest_page_count = _scalar(row["num_pages"])
        if input_page_count != manifest_page_count:
            raise VerificationError(
                f"{record_id}: baseline input_page_count differs from manifest num_pages"
            )
        if input_page_count != 1:
            raise VerificationError(
                f"{record_id}: receipt baseline must contain exactly one input page"
            )
        settings = _parse_processor_settings(record["processor_settings"], record_id)
        for field in ("precision", "min_pixels", "max_pixels", "max_new_tokens"):
            if settings.get(field) != model.get(field):
                raise VerificationError(f"{record_id}: processor setting drift in {field}")
        if settings.get("do_sample") is not False or settings.get("batch_size") != 1:
            raise VerificationError(
                f"{record_id}: Qwen baseline must use greedy batch-size-one generation"
            )

    pre_policy = receipt_predictions.loc[
        receipt_predictions["clean_correct"].eq(True)
        & receipt_predictions["attack_eligible"].eq(True)
        & receipt_predictions["status"].eq("complete")
        & receipt_predictions["parse_status"].eq("ok")
    ].copy()
    exclusions: list[dict[str, Any]] = []
    valid_datasets: list[str] = []
    for record in pre_policy.sort_values("record_id").to_dict("records"):
        record_id = str(record["record_id"])
        dataset_name = str(record["dataset_name"])
        if _scalar(record["prediction_normalized"]) != _scalar(record["clean_target"]):
            raise VerificationError(
                f"{record_id}: clean_correct contradicts prediction_normalized"
            )
        policy = evaluate_target_policy(
            record_id=record_id,
            dataset_name=dataset_name,
            clean_target=record["clean_target"],
            attack_target=record["attack_target"],
        )
        if policy["eligible_under_policy"]:
            valid_datasets.append(dataset_name)
        else:
            exclusions.append({
                "record_id": record_id,
                "dataset_name": dataset_name,
                **policy,
            })

    reason_counts = Counter(
        reason for exclusion in exclusions for reason in exclusion["reasons"]
    )
    manifest_counts = _counts_by_dataset(receipt_manifest["dataset_name"])
    baseline_counts = _counts_by_dataset(receipt_predictions["dataset_name"])
    if manifest_counts != baseline_counts:
        raise VerificationError("internal error: equal receipt ID sets produced different counts")
    return {
        "coverage": {
            "usable_validation_manifest": manifest_counts,
            "qwen_validation_baseline": baseline_counts,
            "record_ids_match_exactly": True,
            "metadata_page_count_verified": True,
        },
        "candidate_counts": {
            "pre_policy": _counts_by_dataset(pre_policy["dataset_name"]),
            "target_valid": _counts_by_dataset(valid_datasets),
        },
        "exclusion_counts": {
            "records": len(exclusions),
            "by_reason": dict(sorted(reason_counts.items())),
        },
        "exclusions": exclusions,
    }


def _resolve_paths(repo_root: Path, overrides: Mapping[str, str | Path] | None) -> dict[str, Path]:
    configured: dict[str, str | Path] = dict(DEFAULT_PATHS)
    if overrides:
        configured.update({key: value for key, value in overrides.items() if value is not None})
    paths: dict[str, Path] = {}
    for key, value in configured.items():
        path = Path(value)
        path = path if path.is_absolute() else repo_root / path
        path = path.resolve()
        if not path.is_file():
            raise VerificationError(f"missing Phase-0 artifact {key}: {path}")
        paths[key] = path
    return paths


def _artifact_entry(path: Path, repo_root: Path) -> dict[str, str]:
    try:
        displayed = str(path.relative_to(repo_root))
    except ValueError:
        displayed = str(path)
    return {"path": displayed, "sha256": sha256_file(path)}


def _git_info(repo_root: Path, *, expected_branch: str, allow_dirty: bool) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise VerificationError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc

    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    if branch != expected_branch:
        raise VerificationError(
            f"Phase 0 must run from branch {expected_branch!r}; found {branch!r}"
        )
    if status and not allow_dirty:
        raise VerificationError("Phase 0 requires a clean Git worktree")
    return {"branch": branch, "commit": commit, "dirty": bool(status)}


def _verify_baseline_contract(baseline: Mapping[str, Any]) -> None:
    expected_top = {
        "run_id": "baseline_validation_v1",
        "stage": "baseline",
        "split": "validation",
        "seed": 566,
    }
    for field, expected in expected_top.items():
        if baseline.get(field) != expected:
            raise VerificationError(
                f"baseline config drift in {field}: {baseline.get(field)!r} != {expected!r}"
            )
    if set(baseline.get("datasets", [])) != {
        "sroie", "cord_v2", "resume_parsing_vision"
    } or len(baseline.get("datasets", [])) != 3:
        raise VerificationError("baseline datasets do not match the frozen three-dataset contract")
    qwen = baseline.get("models", {}).get("qwen", {})
    if qwen.get("model_id") != QWEN_MODEL_ID or qwen.get("revision") != QWEN_MODEL_REVISION:
        raise VerificationError("baseline Qwen model ID/revision differs from the frozen contract")
    prompt = baseline.get("prompts", {}).get(RECEIPT_TASK, {})
    if prompt != {"id": RECEIPT_PROMPT_ID, "text": RECEIPT_PROMPT_TEXT}:
        raise VerificationError("baseline receipt prompt differs from the frozen contract")


def verify_phase0(
    repo_root: str | Path,
    *,
    path_overrides: Mapping[str, str | Path] | None = None,
    enforce_git: bool = False,
    expected_branch: str = "qwen-receipt",
    allow_dirty: bool = False,
    enforce_accepted_counts: bool = True,
) -> dict[str, Any]:
    """Read and verify every frozen Phase-0 artifact, returning a JSON-ready report."""
    import pandas as pd

    root = Path(repo_root).resolve()
    paths = _resolve_paths(root, path_overrides)
    artifacts = {
        key: _artifact_entry(path, root) for key, path in sorted(paths.items())
    }
    reference = _read_object(paths["reference_config"])
    baseline = _read_object(paths["baseline_config"])
    frozen_baseline = _read_object(paths["baseline_frozen_config"])
    merge_manifest = _read_object(paths["baseline_merge_manifest"])
    summary = _read_object(paths["baseline_summary"])
    environment = _read_object(paths["baseline_environment"])
    _verify_baseline_contract(baseline)
    _verify_baseline_contract(frozen_baseline)

    expected_config_hash = reference.get("baseline_config_sha256")
    expected_predictions_hash = reference.get("baseline_predictions_sha256")
    if reference.get("baseline_run_id") != baseline["run_id"]:
        raise VerificationError("reference config points to a different baseline run")
    for key in ("baseline_config", "baseline_frozen_config"):
        if artifacts[key]["sha256"] != expected_config_hash:
            raise VerificationError(
                f"{key} SHA-256 differs from the frozen reference: "
                f"{artifacts[key]['sha256']} != {expected_config_hash}"
            )
    if artifacts["baseline_predictions"]["sha256"] != expected_predictions_hash:
        raise VerificationError("baseline predictions SHA-256 differs from the frozen reference")
    if baseline != frozen_baseline:
        raise VerificationError("tracked and run-copied baseline configs are not identical")
    if merge_manifest.get("config_sha256") != expected_config_hash:
        raise VerificationError("baseline merge manifest/config hash mismatch")

    predictions = pd.read_parquet(paths["baseline_predictions"])
    manifest = pd.read_parquet(paths["master_manifest"])
    _require_columns(
        predictions,
        {
            "run_id", "stage", "experiment_key", "model_key", "record_id",
            "dataset_name", "worker_id", "config_sha256", "git_commit",
            "clean_correct",
        },
        "baseline predictions",
    )
    record_count = len(predictions)
    if merge_manifest.get("records") != record_count or summary.get("records") != record_count:
        raise VerificationError(
            "baseline prediction row count differs from merge manifest or summary"
        )
    if summary.get("expected_records") != record_count:
        raise VerificationError("baseline summary expected_records differs from predictions")
    fingerprint = merge_manifest.get("records_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise VerificationError("baseline merge manifest has no valid records fingerprint")
    if predictions["experiment_key"].duplicated().any():
        raise VerificationError("baseline predictions contain duplicate experiment keys")
    if predictions.duplicated(["model_key", "record_id"]).any():
        raise VerificationError("baseline predictions contain duplicate model/record pairs")

    unique_documents = int(predictions["record_id"].nunique())
    worker_ids = sorted(int(_scalar(value)) for value in predictions["worker_id"].unique())
    observed_groups = Counter(
        (str(model), str(dataset))
        for model, dataset in zip(
            predictions["model_key"], predictions["dataset_name"]
        )
    )
    if enforce_accepted_counts:
        if record_count != EXPECTED_BASELINE_RECORDS:
            raise VerificationError(
                f"accepted baseline must contain {EXPECTED_BASELINE_RECORDS} records; "
                f"found {record_count}"
            )
        if unique_documents != EXPECTED_BASELINE_DOCUMENTS:
            raise VerificationError(
                f"accepted baseline must contain {EXPECTED_BASELINE_DOCUMENTS} documents; "
                f"found {unique_documents}"
            )
        if worker_ids != [0, 1, 2, 3, 4]:
            raise VerificationError(
                f"accepted baseline must contain workers 0-4; found {worker_ids}"
            )
        if observed_groups != Counter(EXPECTED_BASELINE_GROUP_COUNTS):
            raise VerificationError(
                "accepted baseline model/dataset counts differ from the frozen contract"
            )
        if summary.get("inference_failures") != 0:
            raise VerificationError("accepted baseline summary must report zero inference failures")
    for field, expected in (
        ("run_id", baseline["run_id"]),
        ("stage", "baseline"),
        ("config_sha256", expected_config_hash),
    ):
        if not predictions[field].eq(expected).all():
            raise VerificationError(f"baseline prediction provenance drift in {field}")
    git_commits = {_scalar(value) for value in predictions["git_commit"]}
    if None in git_commits or len(git_commits) != 1:
        raise VerificationError("baseline predictions contain mixed or empty Git commits")
    baseline_git_commit = next(iter(git_commits))
    if merge_manifest.get("git_commit") != baseline_git_commit:
        raise VerificationError("baseline merge manifest Git commit differs from predictions")
    if environment.get("git_commit") != baseline_git_commit:
        raise VerificationError("baseline environment Git commit differs from predictions")
    for field in ("python", "torch", "cuda", "transformers", "datasets", "pillow", "gpu"):
        if not environment.get(field):
            raise VerificationError(f"baseline environment is missing {field}")

    saved_correct_ids = paths["baseline_correct_ids"].read_text(encoding="utf-8").splitlines()
    if len(saved_correct_ids) != len(set(saved_correct_ids)):
        raise VerificationError("qwen_clean_correct_ids.txt contains duplicates")
    actual_correct_ids = predictions.loc[
        predictions["model_key"].eq("qwen") & predictions["clean_correct"].eq(True),
        "record_id",
    ].astype(str).tolist()
    if sorted(saved_correct_ids) != sorted(actual_correct_ids):
        raise VerificationError(
            "saved Qwen clean-correct IDs disagree with frozen baseline predictions"
        )

    receipt_report = analyze_receipt_candidates(predictions, manifest, baseline)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "verification": "qwen_receipt_phase0",
        "artifacts": artifacts,
        "provenance": {
            "baseline_run_id": baseline["run_id"],
            "baseline_git_commit": baseline_git_commit,
            "baseline_records": record_count,
            "baseline_unique_documents": unique_documents,
            "baseline_worker_ids": worker_ids,
            "baseline_group_counts": {
                model: {
                    dataset: int(observed_groups.get((model, dataset), 0))
                    for expected_model, dataset in EXPECTED_BASELINE_GROUP_COUNTS
                    if expected_model == model
                }
                for model in sorted({key[0] for key in EXPECTED_BASELINE_GROUP_COUNTS})
            },
            "qwen_model_id": QWEN_MODEL_ID,
            "qwen_model_revision": QWEN_MODEL_REVISION,
            "receipt_prompt_id": RECEIPT_PROMPT_ID,
            "receipt_prompt_text": RECEIPT_PROMPT_TEXT,
            "baseline_environment_scope": "worker_00_representative",
        },
        "target_policy": {
            "clean_total_must_be_positive": True,
            "target_must_be_distinct": True,
            "target_must_equal_exactly_ten_times_clean": True,
            "decimal_arithmetic": True,
        },
        **receipt_report,
    }
    if enforce_git:
        report["git"] = _git_info(
            root, expected_branch=expected_branch, allow_dirty=allow_dirty
        )
    return report


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only integrity verification for Qwen receipt Phase 0."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-branch", default="qwen-receipt")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="report a dirty worktree instead of rejecting it (not for accepted Phase 0)",
    )
    for key, default in DEFAULT_PATHS.items():
        parser.add_argument(
            "--" + key.replace("_", "-"),
            dest=key,
            default=default,
            help=f"artifact path relative to --repo-root (default: {default})",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {key: getattr(args, key) for key in DEFAULT_PATHS}
    try:
        report = verify_phase0(
            args.repo_root,
            path_overrides=overrides,
            enforce_git=True,
            expected_branch=args.expected_branch,
            allow_dirty=args.allow_dirty,
        )
    except Exception as exc:
        failure = {
            "status": "fail",
            "verification": "qwen_receipt_phase0",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    if args.output is not None:
        _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
