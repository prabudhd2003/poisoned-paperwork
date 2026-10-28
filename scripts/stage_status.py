#!/usr/bin/env python3
"""Print a read-only progress and interim-result summary for any new stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("run_id")
    args = parser.parse_args()
    run = ROOT / "outputs" / args.stage / args.run_id
    if not run.is_dir():
        raise SystemExit(f"run not found: {run}")
    total = 0
    rows = []
    print(f"Run: {run}")
    for worker in range(5):
        shard = run / "shards" / f"worker_{worker:02d}"
        count = len(list(shard.joinpath("records").glob("*.json")))
        for path in shard.joinpath("records").glob("*.json"):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        complete = shard.joinpath("WORKER_COMPLETE.json").is_file()
        total += count
        print(f"worker_{worker:02d}: records={count} complete={complete}")
    print(f"records currently saved: {total}")
    if rows:
        try:
            import pandas as pd
            frame = pd.DataFrame(rows)
            outcome = next((name for name in (
                "saved_image_success", "adversarial_target_success", "clean_correct"
            ) if name in frame), None)
            groups = [name for name in ("model_key", "example_type", "defense_id", "control", "epsilon_255") if name in frame]
            if outcome:
                print(f"interim outcome: {outcome}")
                if groups:
                    print(frame.groupby(groups, dropna=False)[outcome].agg(["count", "sum", "mean"]).to_string())
                else:
                    print(frame[outcome].agg(["count", "sum", "mean"]).to_string())
        except Exception as exc:
            print(f"interim aggregation unavailable: {type(exc).__name__}: {exc}")
    merged = run / "merged" / "summary.json"
    if merged.is_file():
        print("MERGED COMPLETE")
        print(merged.read_text(encoding="utf-8"))
    else:
        print("INTERIM ONLY: strict merged summary does not exist yet")


if __name__ == "__main__":
    main()
