#!/usr/bin/env python3
"""Run an isolated one-worker Donut attack smoke test on CARC."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/donut_attack/smoke_validation.json",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Recorded submitter identity (default: config owner_user_id)",
    )
    parser.add_argument("--preflight", action="store_true", help="Validate selection on CPU; no inference")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    from stages.donut_attack import _validate_config, _load_eligible_baseline, run
    _validate_config(config)
    if config["mode"] != "smoke" or config["split"] != "validation":
        raise ValueError("This runner accepts only validation-split smoke configs")
    smoke_per_dataset = int(config.get("smoke_per_dataset", 0))
    if smoke_per_dataset < 1:
        raise ValueError("A smoke config must select at least one receipt per dataset")
    user_id = args.user_id or config.get("owner_user_id")
    if not user_id:
        raise ValueError("Set owner_user_id in the config or pass --user-id")
    baseline = _load_eligible_baseline(config, ROOT)
    expected_counts = {"cord_v2": smoke_per_dataset, "sroie": smoke_per_dataset}
    if baseline.groupby("dataset_name").size().to_dict() != expected_counts:
        raise ValueError(f"Expected smoke counts {expected_counts}")
    if not baseline["status"].eq("complete").all() or not baseline["parse_status"].eq("ok").all():
        raise ValueError("Selected baseline records must have successful inference and parsing")
    from decimal import Decimal
    for _, row in baseline.iterrows():
        if Decimal(str(row["clean_target"])) <= 0:
            raise ValueError("Smoke test excludes non-positive receipt totals")
        if Decimal(str(row["attack_target"])) != Decimal(str(row["clean_target"])) * 10:
            raise ValueError("Expected the frozen ten-times-total target")
    print(baseline[["dataset_name", "clean_target", "attack_target", "clean_prediction"]].to_string(), flush=True)
    print(
        f"Isolated smoke run: owner {user_id}, single smoke worker 0; "
        "production identity remains unchanged.",
        flush=True,
    )
    if args.preflight:
        print("CPU PREFLIGHT PASSED")
        return
    run(config, config_path, user_id=user_id, worker_id=0, num_workers=1)
    import pandas as pd
    output = ROOT / "outputs/donut_attack" / config["run_id"]
    frame = pd.read_parquet(output / "merged/attacks.parquet")
    expected_records = len(baseline) * len(config["attack"]["epsilons_255"])
    if len(frame) != expected_records or set(frame.record_id) != set(baseline.index):
        raise RuntimeError("Smoke results do not cover the selected receipts")
    if not frame.status.eq("complete").all() or not (frame.linf_255 <= frame.epsilon_255 + 1e-4).all():
        raise RuntimeError("Smoke execution or saved-image perturbation check failed")
    finite_columns = [
        "linf_255", "lpips", "ssim", "final_target_loss", "best_target_loss",
        "gradient_l2_final", "gradient_linf_final",
    ]
    for column in finite_columns:
        if column not in frame or not frame[column].map(
            lambda value: value is not None and math.isfinite(float(value))
        ).all():
            raise RuntimeError(f"Smoke output has a missing or non-finite {column}")
    manifest = json.loads((output / "merged/merge_manifest.json").read_text())
    if (
        manifest.get("records") != expected_records
        or manifest.get("baseline_predictions_sha256")
        != config["baseline_predictions_sha256"]
    ):
        raise RuntimeError("Smoke merge manifest failed its provenance check")
    record_paths = sorted((output / "shards/worker_00/records").glob("*.json"))
    if len(record_paths) != expected_records:
        raise RuntimeError("Smoke output does not have the expected JSON traces")
    for record_path in record_paths:
        trace = json.loads(record_path.read_text()).get("optimization_trace")
        if not trace:
            raise RuntimeError(f"Missing optimization trace: {record_path}")
        for point in trace:
            for field in (
                "target_loss", "mean_target_log_probability", "linf_255",
                "gradient_l2", "gradient_linf",
            ):
                value = point.get(field)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"Invalid {field} in trace: {record_path}")
            if not isinstance(point.get("prediction"), str) or not isinstance(
                point.get("exact_target"), bool
            ):
                raise RuntimeError(f"Invalid decoded prediction trace: {record_path}")
    print(frame[["record_id", "target_text", "final_prediction_normalized", "targeted_success_reloaded", "linf_255", "lpips", "ssim"]].to_string(), flush=True)
    print("SMOKE EXECUTION COMPLETED; inspect attack success and traces before expanding.", flush=True)


if __name__ == "__main__":
    main()
