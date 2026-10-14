#!/usr/bin/env python3
"""Freeze decoded RGB identities for all 219 Qwen receipt candidates on CPU.

This creates evidence only. It does not load Qwen, initialize CUDA, submit a
job, or authorize the gated validation config. The resulting report digest
must be reviewed and pinned in a later closure commit before GPU smoke.
"""

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
        default=ROOT / "configs/qwen_attack/receipt_smoke_saaketh.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise SystemExit("Source evidence output must stay below this checkout's outputs/")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing source evidence: {output}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    from checkpoints import atomic_json, sha256_file
    from stages.qwen_attack_receipt import build_source_evidence

    report = build_source_evidence(config, config_path)
    atomic_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "records": report["records"],
        "candidate_counts": report["candidate_counts"],
        "source_records_sha256": report["source_records_sha256"],
        "report_sha256": sha256_file(output),
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
