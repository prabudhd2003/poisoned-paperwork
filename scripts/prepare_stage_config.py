#!/usr/bin/env python3
"""Create a hash-frozen runtime config without dirtying the Git worktree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import sha256_file  # noqa: E402


TEMPLATES = {
    "baseline_collected": "configs/baseline/collected_external_test.json",
    "qwen_collected": "configs/qwen_attack/collected_external_test.json",
    "donut_collected": "configs/donut_attack/collected_external_test.json",
    "eot_patch": "configs/eot_patch/active.json",
    "transfer": "configs/transfer_robustness/active.json",
    "defenses": "configs/defenses/active.json",
    "defenses_public": "configs/defenses/qwen_public_final.json",
}


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(TEMPLATES))
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PARQUET")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    template = ROOT / TEMPLATES[args.stage]
    config = json.loads(template.read_text(encoding="utf-8"))
    contract_path = ROOT / "data/processed/collected_receipts/manifest_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else None
    if args.stage in {"baseline_collected", "qwen_collected", "donut_collected", "eot_patch"}:
        if contract is None:
            raise SystemExit("run scripts/freeze_collected_external.py first")
        target = config.get("collected_manifest_contract", config)
        target.update(contract)
    if args.stage in {"qwen_collected", "donut_collected", "eot_patch"}:
        baseline = ROOT / "outputs/baseline/baseline_collected_external_v1/merged/records.parquet"
        if not baseline.is_file():
            raise SystemExit(f"baseline output missing: {baseline}")
        config["baseline_predictions"] = relative_or_absolute(baseline)
        config["baseline_predictions_sha256"] = sha256_file(baseline)
    if args.stage in {"transfer", "defenses", "defenses_public"}:
        if not args.source:
            raise SystemExit("provide every frozen attack table as --source NAME=PATH")
        sources = []
        for item in args.source:
            if "=" not in item:
                raise SystemExit(f"invalid --source: {item}")
            name, raw_path = item.split("=", 1)
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise SystemExit(f"attack source missing: {path}")
            sources.append({
                "name": name, "records": relative_or_absolute(path),
                "artifact_root": str(path.parent.parent.resolve()),
                "sha256": sha256_file(path),
            })
        config["attack_sources"] = sources
        if contract is not None and "collected_manifest_contract" in config:
            config["collected_manifest_contract"].update(contract)
    config["ready_for_production"] = True
    output = args.output or ROOT / "outputs/runtime_configs" / f"{args.stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(f"sha256={sha256_file(output)}")
    print("Runtime config frozen. Its exact copy will also be saved inside the run output.")


if __name__ == "__main__":
    main()
