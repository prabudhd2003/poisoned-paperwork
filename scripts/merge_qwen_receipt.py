#!/usr/bin/env python3
"""Strictly merge completed Qwen-receipt shards on a CARC CPU login node.

All assignment, Cartesian-grid, provenance, image, metric, locking,
fingerprint, idempotency, and receipt-denominator checks live in
``stages.qwen_attack_receipt.merge_if_complete``. This file is intentionally
only the operator-facing CLI so the merge contract has one implementation.
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
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from stages.qwen_attack_receipt import merge_if_complete, resolve_output_root

    previous_umask = os.umask(0o002)
    try:
        if not merge_if_complete(config, config_path, num_workers=args.num_workers):
            raise SystemExit("Not all Qwen-receipt worker markers exist; nothing was merged.")
    finally:
        os.umask(previous_umask)
    repo_root = config_path.parents[2]
    print(resolve_output_root(config, repo_root) / "merged/summary.json")


if __name__ == "__main__":
    main()
