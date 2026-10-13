#!/usr/bin/env python3
"""Isolated one-worker Qwen smoke run for either track (owner only).

Usage (from the repository root):
    python scripts/smoke_qwen_attack.py --config configs/qwen_attack/<smoke>.json --preflight
    sbatch scripts/slurm/qwen_smoke.sbatch configs/qwen_attack/<smoke>.json

Validation and test runs never use this script; they use
``bash scripts/submit_stage.sh qwen_attack userN <config>`` on five shards.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", default=None, help="Recorded submitter identity (default: config owner_user_id)")
    parser.add_argument("--preflight", action="store_true", help="CPU checks only; no model loading")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text())
    if config.get("mode") != "smoke" or config.get("split") != "validation" or config.get("frozen_for_test"):
        raise SystemExit("This runner only accepts validation-split smoke configs (mode='smoke').")

    from stages.qwen_attack import TRACKS, _validate, run
    track, module_name = _validate(config)
    user_id = args.user_id or config.get("owner_user_id")
    if not user_id:
        raise SystemExit("Set owner_user_id in the smoke config or pass --user-id.")
    print(f"Qwen {track} smoke run {config['run_id']} by {user_id}; single worker 0 of 1.", flush=True)

    if args.preflight:
        module = importlib.import_module(module_name)
        preflight = getattr(module, "preflight", None)
        if preflight is None:
            raise SystemExit(f"{module_name} must define preflight(config, repo_root) for CPU checks.")
        preflight(config, ROOT)
        print("CPU PREFLIGHT PASSED", flush=True)
        return
    run(config, config_path, user_id=user_id, worker_id=0, num_workers=1)
    output = ROOT / "outputs/qwen_attack" / config["run_id"]
    summary = json.loads((output / "merged/summary.json").read_text(encoding="utf-8"))
    expected_documents = int(config.get("smoke_per_page_group", 0)) * 2
    expected_records = expected_documents * len(config["attack"]["epsilons_255"])
    if (
        summary.get("records") != expected_records
        or summary.get("unique_documents") != expected_documents
        or summary.get("failures_logged") != 0
    ):
        raise RuntimeError(f"Smoke summary failed acceptance: {summary}")
    record_paths = sorted((output / "shards/worker_00/records").glob("*.json"))
    if len(record_paths) != expected_records:
        raise RuntimeError("Smoke shard does not contain the exact expected record count")
    resumed_records = 0
    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("checkpoint_resumed"):
            resumed_records += 1
        for field in (
            "linf_255", "lpips", "ssim", "initial_loss", "final_loss",
            "best_loss", "gradient_norm_final", "peak_gpu_memory_mb",
        ):
            value = record.get(field)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"Invalid {field} in {record_path}")
        if float(record["linf_255"]) > float(record["epsilon_255"]) + 1e-4:
            raise RuntimeError(f"Epsilon violation in {record_path}")
        if not record.get("optimization_trace") or not record.get("images"):
            raise RuntimeError(f"Missing trace or saved page metadata in {record_path}")
    print(
        f"SMOKE EXECUTION COMPLETED: {expected_records} records on "
        f"{expected_documents} documents; resumed records={resumed_records}. "
        "Inspect examples and prove checkpoint resume before unlocking production.",
        flush=True,
    )


if __name__ == "__main__":
    main()
