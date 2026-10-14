#!/usr/bin/env python3
"""Isolated one-worker Qwen smoke run for either track (owner only).

Usage (from the repository root):
    python scripts/smoke_qwen_attack.py --config configs/qwen_attack/<smoke>.json --preflight
    # Receipt GPU smoke: bash scripts/submit_qwen_smoke.sh <receipt-smoke-config>
    # Resume GPU smoke: sbatch scripts/slurm/qwen_smoke.sbatch <resume-smoke-config>

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


def _expected_smoke_documents(config):
    """Return the track-specific smoke document count without changing either contract."""
    track = config.get("track")
    if track == "receipt":
        count = config.get("smoke_per_dataset")
        groups = len(config.get("datasets") or [])
        field = "smoke_per_dataset"
    elif track == "resume":
        count = config.get("smoke_per_page_group")
        groups = 2  # one single-page group and one multi-page group
        field = "smoke_per_page_group"
    else:
        raise ValueError(f"unsupported Qwen smoke track: {track!r}")
    if type(count) is not int or count < 1:
        raise ValueError(f"{field} must be a positive integer")
    if groups < 1:
        raise ValueError("smoke config must declare at least one dataset")
    return count * groups


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
    if track == "receipt" and config.get("ready_for_gpu_smoke") is not True:
        raise SystemExit(
            "Receipt GPU smoke is gated until CARC source-pixel evidence is "
            "reviewed and pinned in a closure commit. Run --preflight only."
        )
    run(config, config_path, user_id=user_id, worker_id=0, num_workers=1)
    if track == "receipt":
        from stages.qwen_attack_receipt import resolve_output_root

        output = resolve_output_root(config, ROOT)
    else:
        output = ROOT / "outputs/qwen_attack" / config["run_id"]
    summary = json.loads((output / "merged/summary.json").read_text(encoding="utf-8"))
    expected_documents = _expected_smoke_documents(config)
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
