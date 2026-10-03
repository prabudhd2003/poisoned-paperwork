#!/usr/bin/env python3
"""Run the complete 10-inference baseline smoke test on one interactive GPU."""

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
        default=REPO_ROOT / "configs" / "baseline" / "smoke.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("smoke_per_dataset") != 2:
        raise SystemExit("Smoke config must select exactly two documents per dataset.")

    from stages.baseline import run

    run(config, config_path, user_id="user1", worker_id=0, num_workers=1)
    merged = REPO_ROOT / config["output_root"] / config["run_id"] / "merged"
    summary = json.loads((merged / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
    if summary["records"] != 10:
        raise SystemExit("Smoke merge did not contain the expected 10 prediction records.")
    if summary["inference_failures"]:
        raise SystemExit("Smoke test recorded inference failures; inspect failures.csv.")
    print(f"Smoke test passed. Inspect {merged / 'predictions.csv'}")


if __name__ == "__main__":
    main()
