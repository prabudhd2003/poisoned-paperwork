#!/usr/bin/env python3
"""CPU preflight and strict acceptance checks for a frozen Donut run."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def preflight(config_path):
    from stages.donut_attack import _load_eligible_baseline, _validate_config
    config = json.loads(config_path.read_text())
    _validate_config(config)
    if config["split"] != "validation":
        raise ValueError("This readiness workflow is validation-only")
    baseline = _load_eligible_baseline(config, ROOT)
    if not baseline.status.eq("complete").all() or not baseline.parse_status.eq("ok").all():
        raise ValueError("Selected baseline contains inference/parse failures")
    if baseline.deterministic.eq(False).any():
        raise ValueError("Selected baseline contains a failed determinism check")
    if len(set(baseline.config_sha256)) != 1 or len(set(baseline.git_commit)) != 1:
        raise ValueError("Baseline contains mixed provenance")
    for _, row in baseline.iterrows():
        if Decimal(str(row.clean_target)) <= 0 or Decimal(str(row.attack_target)) != 10 * Decimal(str(row.clean_target)):
            raise ValueError("Invalid receipt target")
    count = config.get("smoke_per_dataset")
    if count and baseline.groupby("dataset_name").size().to_dict() != {"cord_v2": count, "sroie": count}:
        raise ValueError("Smoke selection is missing requested examples")
    print(json.dumps(dict(run_id=config["run_id"], selected=baseline.groupby("dataset_name").size().to_dict(),
                          expected_results=len(baseline)*len(config["attack"]["epsilons_255"])), indent=2), flush=True)
    return config, baseline


def validate_results(config_path, *, expected_workers):
    from PIL import Image
    import numpy as np
    from data import load_processed_rows
    from stages.donut_attack import _safe_key
    config, baseline = preflight(config_path)
    root = ROOT / "outputs/donut_attack" / config["run_id"]
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if hashlib.sha256((root / "config.json").read_bytes()).hexdigest() != config_hash:
        raise ValueError("Output configuration differs from frozen config")
    expected = {(rid, float(eps)) for rid in baseline.index for eps in config["attack"]["epsilons_255"]}
    records = []
    assignments = []
    for worker in range(expected_workers):
        shard = root / "shards" / f"worker_{worker:02d}"
        marker = json.loads((shard / "WORKER_COMPLETE.json").read_text())
        assigned = (root / "assignments" / f"worker_{worker:02d}.txt").read_text().splitlines()
        assignments.extend(assigned)
        paths = sorted((shard / "records").glob("*.json"))
        if marker["completed_records"] != len(paths) or len(paths) != len(assigned)*len(config["attack"]["epsilons_255"]):
            raise ValueError("Worker completion counts mismatch")
        for path in paths:
            record = json.loads(path.read_text())
            if record["worker_id"] != worker or record["record_id"] not in assigned:
                raise ValueError("Record is assigned to the wrong worker")
            expected_key = _safe_key(record["record_id"], record["epsilon_255"], config["seed"])
            if path.stem != expected_key:
                raise ValueError("Unexpected experiment filename")
            records.append((record, shard / "images" / f"{path.stem}.png"))
    actual = [(r["record_id"], float(r["epsilon_255"])) for r, _ in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicate, or unexpected experiment results")
    if len(assignments) != len(set(assignments)) or set(assignments) != set(baseline.index):
        raise ValueError("Worker assignments overlap or omit eligible receipts")
    commits = {r["git_commit"] for r, _ in records}
    if len(commits) != 1:
        raise ValueError("Mixed code commits")
    rows = load_processed_rows(ROOT, "validation", set(baseline.index))
    for r, path in records:
        if (r["config_sha256"] != config_hash or r["model_revision"] != config["model_revision"]
                or r["model_id"] != config["model_id"] or r["run_id"] != config["run_id"]
                or r["experiment_split"] != "validation" or r["status"] != "complete"):
            raise ValueError("Invalid record provenance/status")
        for field in ("linf_255", "lpips", "ssim", "attack_runtime_seconds", "final_target_loss"):
            if not math.isfinite(float(r[field])):
                raise ValueError(f"Nonfinite {field}")
        if r["linf_255"] < 0 or r["linf_255"] > r["epsilon_255"]+1e-4 or not -1 <= r["ssim"] <= 1 or r["lpips"] < 0:
            raise ValueError("Invalid perceptibility metrics")
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["attacked_image_sha256"]:
            raise ValueError("Saved PNG checksum mismatch")
        clean = np.asarray(rows[r["record_id"]]["image"].convert("RGB"), dtype=np.uint8)
        if hashlib.sha256(clean.transpose(2, 0, 1).tobytes()).hexdigest() != r["clean_image_sha256"]:
            raise ValueError("Clean image checksum mismatch")
        with Image.open(path) as image:
            attacked = np.asarray(image.convert("RGB"), dtype=np.int16)
        if attacked.shape != clean.shape or np.abs(attacked-clean.astype(np.int16)).max() > r["epsilon_255"]:
            raise ValueError("Saved image dimensions or perturbation bound invalid")
        if r["targeted_success_reloaded"] != (r["final_prediction_normalized"] == r["target_text"]):
            raise ValueError("Inconsistent success flag")
    failures = sum(len(p.read_text().splitlines()) for p in root.glob("shards/*/failures.jsonl"))
    report = dict(passed=True, run_id=config["run_id"], records=len(records),
                  code_commit=next(iter(commits)), config_sha256=config_hash,
                  failures_logged=failures, reloaded_successes=sum(r["targeted_success_reloaded"] for r, _ in records),
                  visual_review_required=True)
    (root / "readiness_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    path = args.config.resolve()
    if args.results:
        validate_results(path, expected_workers=args.workers)
    else:
        preflight(path)
