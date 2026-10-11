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
    print("SMOKE EXECUTION COMPLETED; inspect records, traces and saved PNGs before expanding.", flush=True)


if __name__ == "__main__":
    main()
