#!/usr/bin/env python3
"""Strictly merge completed Qwen-resume shards on a CARC CPU login node."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_attack/resume_active.json",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from stages.qwen_attack_resume import merge_if_complete

    if not merge_if_complete(config, config_path, num_workers=args.num_workers):
        raise SystemExit("Not all Qwen-resume worker markers exist; nothing was merged.")
    print(ROOT / "outputs/qwen_attack" / config["run_id"] / "merged/summary.json")


if __name__ == "__main__":
    main()
