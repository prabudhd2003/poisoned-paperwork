#!/usr/bin/env python3
"""Isolated one-worker smoke test; user4 owns smoke slot 0, not production slot 3."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate selection on CPU; no inference")
    args = parser.parse_args()
    config_path = ROOT / "configs/donut_attack/smoke_khalid.json"
    config = json.loads(config_path.read_text())
    from stages.donut_attack import _validate_config, _load_eligible_baseline, run
    _validate_config(config)
    if config["run_id"] != "donut_smoke_khalid_v1" or config["smoke_per_dataset"] != 1:
        raise ValueError("This runner is restricted to the two-document smoke configuration")
    baseline = _load_eligible_baseline(config, ROOT)
    if baseline.groupby("dataset_name").size().to_dict() != {"cord_v2": 1, "sroie": 1}:
        raise ValueError("Expected one clean-correct eligible receipt from each dataset")
    if not baseline["status"].eq("complete").all() or not baseline["parse_status"].eq("ok").all():
        raise ValueError("Selected baseline records must have successful inference and parsing")
    from decimal import Decimal
    for _, row in baseline.iterrows():
        if Decimal(str(row["clean_target"])) <= 0:
            raise ValueError("Smoke test excludes non-positive receipt totals")
        if Decimal(str(row["attack_target"])) != Decimal(str(row["clean_target"])) * 10:
            raise ValueError("Expected the frozen ten-times-total target")
    print(baseline[["dataset_name", "clean_target", "attack_target", "clean_prediction"]].to_string(), flush=True)
    print("Isolated smoke run: owner user4, single smoke slot 0; production identity unchanged.", flush=True)
    if args.preflight:
        print("CPU PREFLIGHT PASSED")
        return
    run(config, config_path, user_id="user4", worker_id=0, num_workers=1)
    import pandas as pd
    output = ROOT / "outputs/donut_attack" / config["run_id"]
    frame = pd.read_parquet(output / "merged/attacks.parquet")
    if len(frame) != 2 or set(frame.record_id) != set(baseline.index):
        raise RuntimeError("Smoke results do not cover the selected receipts")
    if not frame.status.eq("complete").all() or not (frame.linf_255 <= frame.epsilon_255 + 1e-4).all():
        raise RuntimeError("Smoke execution or saved-image perturbation check failed")
    print(frame[["record_id", "target_text", "final_prediction_normalized", "targeted_success_reloaded", "linf_255", "lpips", "ssim"]].to_string(), flush=True)
    print("SMOKE EXECUTION COMPLETED; inspect attack success and traces before expanding.", flush=True)


if __name__ == "__main__":
    main()
