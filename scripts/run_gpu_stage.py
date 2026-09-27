#!/usr/bin/env python3
"""Dispatch one checkpointed GPU stage worker.

Stage owners implement ``run(...)`` in ``src/stages/<stage>.py``. The Slurm
launcher supplies the stage, config, user identity, and fixed worker ID.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


STAGES = {
    "baseline",
    "donut_attack",
    "qwen_attack",
    "transfer_robustness",
    "eot_patch",
    "defenses",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from team import get_team_member

    member = get_team_member(args.user_id)
    if member.worker_id != args.worker_id:
        raise SystemExit(
            f"Identity mismatch: {args.user_id} must use worker {member.worker_id}, "
            f"not {args.worker_id}."
        )
    if args.num_workers != 5:
        raise SystemExit("The team contract currently requires exactly five workers.")
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")

    config = json.loads(args.config.read_text())
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise SystemExit("Config must contain a non-empty string field named 'run_id'.")

    print(
        json.dumps(
            {
                "stage": args.stage,
                "run_id": run_id,
                "user_id": member.user_id,
                "name": member.name,
                "worker_id": member.worker_id,
                "num_workers": args.num_workers,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": os.uname().nodename,
            },
            indent=2,
        ),
        flush=True,
    )

    module_name = f"stages.{args.stage}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            expected = Path("src/stages") / f"{args.stage}.py"
            raise SystemExit(
                f"Stage '{args.stage}' is planned but not implemented yet. "
                f"The assigned stage owner must create {expected}."
            ) from exc
        raise

    run = getattr(module, "run", None)
    if run is None:
        raise SystemExit(f"{module_name} must define a callable run(...) function.")

    run(
        config=config,
        config_path=args.config.resolve(),
        user_id=member.user_id,
        worker_id=member.worker_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Worker interrupted; restart with the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
