#!/usr/bin/env python3
"""CPU-only merge recovery for an already completed Qwen-receipt validation run.

This command never launches attacks or loads Qwen/LPIPS models.  It verifies the
original attack commit and byte-identical frozen config, runs the existing strict
receipt merger, records the real merger commit separately, and writes before/after
source-artifact hash inventories outside the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_attack/receipt_active.json",
    )
    parser.add_argument("--expected-attack-git-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-merger-git-commit", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=5)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from stages.qwen_attack_receipt import recover_merge_if_complete

    previous_umask = os.umask(0o002)
    try:
        result = recover_merge_if_complete(
            config,
            config_path,
            expected_attack_git_commit=args.expected_attack_git_commit,
            expected_config_sha256=args.expected_config_sha256,
            expected_merger_git_commit=args.expected_merger_git_commit,
            evidence_dir=args.evidence_dir,
            num_workers=args.num_workers,
        )
    finally:
        os.umask(previous_umask)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
