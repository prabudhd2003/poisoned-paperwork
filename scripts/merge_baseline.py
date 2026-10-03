#!/usr/bin/env python3
"""Validate and merge completed baseline shards without requesting a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline" / "active.json",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 1 if config.get("smoke_per_dataset") else 5

    from stages.baseline import merge_if_complete

    if not merge_if_complete(config, config_path, num_workers=num_workers):
        raise SystemExit("Not all worker completion markers exist yet; nothing was merged.")
    print(
        REPO_ROOT / config["output_root"] / config["run_id"] / "merged" / "summary.json"
    )


if __name__ == "__main__":
    main()
