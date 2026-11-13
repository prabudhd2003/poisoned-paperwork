#!/usr/bin/env python3
"""Prepare, run and analyze the report's fixed CSEV study (no historical mutation)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import atomic_json, sha256_file
from csev.study import prepare, read_config, preflight
from csev.runner import run, merge, output, tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "prepare",
            "preflight",
            "run",
            "merge",
            "analyze",
            "status",
            "archive",
        ],
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--run-id", default="csev_receipts_v1")
    p.add_argument(
        "--public-config",
        type=Path,
        default=ROOT / "outputs/runtime_configs/evidence_qwen_public_standard_v2.json",
    )
    p.add_argument("--baseline", type=Path)
    p.add_argument("--standard", type=Path)
    p.add_argument("--uniform-eot", type=Path)
    p.add_argument("--hardview", type=Path)
    p.add_argument(
        "--scratch",
        type=Path,
        default=Path("/scratch1")
        / os.environ.get("USER", "unknown")
        / "poisoned-paperwork/csev",
    )
    p.add_argument(
        "--phase", choices=["development", "evaluation", "adaptive", "replay"]
    )
    p.add_argument("--worker", type=int)
    args = p.parse_args()
    if args.action == "prepare":

        def pick(explicit, folder, names):
            if explicit:
                return explicit
            for name in names:
                candidate = ROOT / folder / "merged" / name
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"No source in {folder}; supply the explicit path")

        baseline = pick(
            args.baseline,
            "outputs/baseline/baseline_collected_external_v1",
            ["predictions.parquet", "records.parquet"],
        )
        source_paths = dict(
            standard=pick(
                args.standard,
                "outputs/qwen_attack/qwen_collected_external_v1",
                ["records.parquet", "attacks.parquet"],
            ),
            uniform_eot=pick(
                args.uniform_eot,
                "outputs/jpeg_eot_attack/qwen_collected_jpeg_eot_full_v1",
                ["records.parquet"],
            ),
            hardview=pick(
                args.hardview,
                "outputs/jpeg_eot_attack/qwen_collected_frequency_hardview_full_v1",
                ["records.parquet"],
            ),
        )
        dest = prepare(
            ROOT, args.run_id, args.public_config, baseline, source_paths, args.scratch
        )
        print(dest)
        c, plan = read_config(dest, ROOT)
        preflight(c, plan, ROOT)
        return
    if not args.config:
        p.error("--config is required")
    c, plan = read_config(
        args.config, ROOT, check_runtime=args.action not in {"status", "archive"}
    )
    if args.action == "preflight":
        preflight(c, plan, ROOT)
    elif args.action == "run":
        if args.phase is None or args.worker is None:
            p.error("--phase and --worker required")
        run(c, plan, ROOT, args.phase, args.worker)
    elif args.action == "merge":
        if args.phase is None:
            p.error("--phase required")
        merge(c, plan, args.phase)
    elif args.action == "analyze":
        from csev.reporting import analyze

        analyze(c, plan)
    elif args.action == "status":
        for phase in ("development", "evaluation", "adaptive", "replay"):
            d = output(c) / phase
            print(
                phase,
                "records:",
                len(list((d / "records").glob("*.json"))),
                "workers:",
                len(list(d.glob("worker_*_complete.json"))),
                "merged:",
                (d / "merged.json").exists(),
            )
    elif args.action == "archive":
        if not (output(c) / "analysis/analysis_complete.json").exists():
            raise ValueError("analyze first")
        for phase in ("development", "evaluation", "adaptive", "replay"):
            merge(c, plan, phase)
        dest = ROOT / "outputs/final/csev" / c["run_id"]
        entries = []
        for src in sorted(output(c).rglob("*")):
            if not src.is_file() or src.suffix in {".pkl", ".tmp", ".lock"}:
                continue
            relative = src.relative_to(output(c))
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = sha256_file(src)
            if target.exists() and sha256_file(target) != checksum:
                raise ValueError(f"archive conflict: {target}")
            if not target.exists():
                shutil.copy2(src, target)
            if sha256_file(target) != checksum:
                raise IOError("archive verification failed")
            entries.append(
                dict(original=str(src), archived=str(target), sha256=checksum)
            )
        shutil.copy2(args.config, dest / "runtime_config.json")
        atomic_json(
            dest / "ARCHIVE_COMPLETE.json",
            dict(
                files=entries,
                input_bundle=c["plan"],
                restoration="Restore paths in this manifest or remap original prefixes; frozen inputs remain under outputs/csev_inputs.",
            ),
        )
        print("Verified permanent archive:", dest)


if __name__ == "__main__":
    main()
