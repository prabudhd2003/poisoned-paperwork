#!/usr/bin/env python3
"""Archive completed final runs plus their frozen inputs and verify every copy."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    archive_root = (ROOT / "outputs/final").resolve()
    if str(archive_root).startswith("/scratch"):
        raise RuntimeError("permanent archive root resolves into scratch")
    run = ROOT / "outputs/final_protocol" / args.run_id
    summary = json.loads((run / "merged/summary.json").read_text())
    if summary["failures"] != 0 or not (run / "merged/final_summary.csv").is_file():
        raise RuntimeError("final analysis summary missing or failures present")
    from stages.final_protocol import read_plan, load_pages, valid_record

    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    for document in read_plan(config, ROOT):
        load_pages(document["images"], ROOT)
        for attack in document["attacks"]:
            load_pages(attack["images"], ROOT)
    records = list(run.glob("shards/worker_*/records/*.json"))
    if len(records) != summary["records"]:
        raise RuntimeError("archive record count differs")
    for record in records:
        valid_record(
            record,
            dict(
                config_sha256=sha256_file(config_path),
                git_commit=summary["git_commit"],
                experiment_key=record.stem,
            ),
            ROOT,
        )
    command = [
        sys.executable,
        str(ROOT / "scripts/manage_carc_storage.py"),
        "archive-run",
        "--stage",
        "final_protocol",
        "--run-id",
        args.run_id,
        "--expected-workers",
        "5",
    ]
    subprocess.run(command + (["--apply"] if args.apply else []), check=True, cwd=ROOT)
    source = ROOT / "outputs/final_inputs" / args.run_id
    destination = ROOT / "outputs/final/final_protocol" / args.run_id / "frozen_inputs"
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    manifest = {str(path.relative_to(source)): sha256_file(path) for path in files}
    print(f"Input bundle: {len(files)} files from {source} to {destination}")
    if args.apply:
        for path in files:
            target = destination / path.relative_to(source)
            if (
                target.exists()
                and sha256_file(target) != manifest[str(path.relative_to(source))]
            ):
                raise RuntimeError(f"conflicting archive file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            if sha256_file(target) != manifest[str(path.relative_to(source))]:
                raise RuntimeError("archive checksum mismatch")
        atomic_json(destination / "INPUT_ARCHIVE_VERIFIED.json", manifest)
        print("Run and input copies verified. No scratch files were deleted.")


if __name__ == "__main__":
    main()
