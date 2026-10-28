#!/usr/bin/env python3
"""CPU-only fail-fast validation run immediately before a Slurm submission."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("stage") != args.stage:
        raise SystemExit(f"config stage {config.get('stage')!r} != requested {args.stage!r}")
    if config.get("ready_for_production") is not True and args.stage != "baseline":
        raise SystemExit("ready_for_production must be true in the committed config")
    module_name = f"stages.{args.stage}"
    if args.stage == "baseline" and config.get("track") == "collected_receipts":
        module_name = "stages.collected_baseline"
    elif args.stage == "donut_attack" and config.get("track") == "collected_receipts":
        module_name = "stages.collected_attacks"
    elif args.stage == "qwen_attack":
        from stages.qwen_attack import _validate
        _, module_name = _validate(config)
    module = importlib.import_module(module_name)
    if args.stage == "donut_attack" and config.get("track") == "collected_receipts":
        module.preflight(config, ROOT, "donut")
    else:
        preflight = getattr(module, "preflight", None)
        if preflight is not None:
            preflight(config, ROOT)
        else:
            validator = getattr(module, "validate_config", None) or getattr(module, "_validate_config", None)
            if validator is not None:
                validator(config)
    print(f"{args.stage.upper()} CPU PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
