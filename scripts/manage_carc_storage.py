#!/usr/bin/env python3
"""Safely place active CARC artifacts in scratch and archive completed runs."""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs/storage/carc_scratch_plan.json"
ARCHIVE_MANIFEST = "ARCHIVE_STORAGE_MANIFEST.json"
TEMPORARY_PATTERNS = ("*.tmp", ".state.pt.*", "core", "core.*")


def username() -> str:
    return os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name


def load_plan(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"storage plan must contain a JSON object: {path}")
    return value


def scratch_root(args, plan: dict) -> Path:
    raw = args.scratch_root or os.environ.get("PP_SCRATCH_ROOT")
    if raw is None:
        raw = str(plan["scratch_root_template"]).format(user=username())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise SystemExit(f"scratch root must be absolute: {path}")
    if path == Path("/") or path == ROOT or ROOT in path.parents:
        raise SystemExit(f"unsafe scratch root: {path}")
    return path


def relative_repo_path(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SystemExit(f"expected a safe repository-relative path: {raw}")
    return relative


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def rsync_copy(source: Path, destination: Path, *, exclude_checkpoints: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    excludes: list[str] = []
    if exclude_checkpoints:
        excludes = ["--exclude=checkpoints/", f"--exclude={ARCHIVE_MANIFEST}"]
    run([
        "rsync", "-rlt", "--stats", *excludes,
        f"{source}/", f"{destination}/",
    ])
    check = run([
        "rsync", "-rltcni", "--delete", *excludes,
        f"{source}/", f"{destination}/",
    ], capture=True)
    if check.stdout.strip():
        preview = "\n".join(check.stdout.splitlines()[:50])
        raise RuntimeError(
            f"copy verification failed for {source} -> {destination}:\n{preview}"
        )


def active_project_jobs() -> list[str]:
    if shutil.which("squeue") is None:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-u", username(), "-o", "%A|%j|%T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    jobs = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 2)
        if len(fields) == 3 and fields[1].startswith("pp_"):
            jobs.append(line)
    return jobs


def ensure_no_jobs(allow_active_jobs: bool) -> None:
    jobs = active_project_jobs()
    if jobs and not allow_active_jobs:
        detail = "\n".join(f"  {line}" for line in jobs)
        raise SystemExit(
            "refusing to change storage while Poisoned Paperwork jobs are active:\n"
            f"{detail}\nCancel them or pass --allow-active-jobs only after a manual audit."
        )


def bytes_under(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def completed_run(run_root: Path, expected_workers: int) -> tuple[bool, str]:
    merged = run_root / "merged"
    if not merged.is_dir() or not any(item.is_file() for item in merged.rglob("*")):
        return False, "strict merged output is missing"
    markers = sorted(run_root.glob("shards/worker_*/WORKER_COMPLETE.json"))
    if len(markers) != expected_workers:
        return False, f"found {len(markers)}/{expected_workers} worker markers"
    return True, "complete"


def command_plan(args, plan: dict) -> None:
    scratch = scratch_root(args, plan)
    print(f"Repository: {ROOT}")
    print(f"Scratch:    {scratch}")
    print("\nActive run links:")
    for raw in plan["active_run_links"]:
        relative = relative_repo_path(raw)
        print(f"  {relative} -> {scratch / relative}")
    print(f"  .cache -> {scratch / 'cache'}")
    print("\nCompleted checkpoint cleanup:")
    for item in plan["completed_checkpoint_runs"]:
        relative = relative_repo_path(item["path"])
        run_root = ROOT / relative
        complete, reason = completed_run(run_root, int(item["expected_workers"]))
        checkpoint_bytes = sum(
            bytes_under(path) for path in run_root.glob("shards/worker_*/checkpoints")
        )
        print(f"  {relative}: {reason}; checkpoints={human_bytes(checkpoint_bytes)}")
        if not complete:
            print("    cleanup will skip this run")


def command_prune(args, plan: dict) -> None:
    ensure_no_jobs(args.allow_active_jobs)
    candidates: list[tuple[Path, list[Path], int]] = []
    skipped: list[tuple[Path, str]] = []
    for item in plan["completed_checkpoint_runs"]:
        relative = relative_repo_path(item["path"])
        run_root = ROOT / relative
        complete, reason = completed_run(run_root, int(item["expected_workers"]))
        if not complete:
            skipped.append((relative, reason))
            continue
        checkpoints = sorted(run_root.glob("shards/worker_*/checkpoints"))
        candidates.append((
            relative,
            checkpoints,
            sum(bytes_under(path) for path in checkpoints),
        ))
    for relative, reason in skipped:
        print(f"SKIP {relative}: {reason}")
    total = sum(item[2] for item in candidates)
    for relative, checkpoints, size in candidates:
        print(f"READY {relative}: {len(checkpoints)} directories, {human_bytes(size)}")
    print(f"Total recoverable project storage: {human_bytes(total)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to remove verified completed checkpoints.")
        return
    for _, checkpoints, _ in candidates:
        for path in checkpoints:
            shutil.rmtree(path)
    print(f"Removed {human_bytes(total)} of completed-run checkpoints.")


def temporary_files() -> list[Path]:
    found: set[Path] = set()
    output = ROOT / "outputs"
    if not output.is_dir():
        return []
    for pattern in TEMPORARY_PATTERNS:
        found.update(path for path in output.rglob(pattern) if path.is_file())
    return sorted(found)


def command_prune_temporary(args, plan: dict) -> None:
    del plan
    ensure_no_jobs(args.allow_active_jobs)
    files = temporary_files()
    total = sum(path.stat().st_size for path in files)
    print(f"Temporary files: {len(files)} totaling {human_bytes(total)}")
    for path in files[:50]:
        print(f"  {path.relative_to(ROOT)}")
    if len(files) > 50:
        print(f"  ... and {len(files) - 50} more")
    if not args.apply:
        print("Dry run only. Re-run with --apply to delete only the listed temporary files.")
        return
    for path in files:
        path.unlink(missing_ok=True)
    print(f"Removed {len(files)} abandoned temporary files totaling {human_bytes(total)}.")


def migrate_directory(source: Path, destination: Path, *, apply: bool) -> None:
    if source.is_symlink():
        observed = source.resolve(strict=False)
        expected = destination.resolve(strict=False)
        if observed != expected:
            raise SystemExit(f"link points to the wrong target: {source} -> {observed}")
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Already linked: {source} -> {destination}")
        return
    print(f"Migrate: {source} -> {destination}")
    if not apply:
        return
    destination.mkdir(parents=True, exist_ok=True)
    if source.exists():
        if not source.is_dir():
            raise SystemExit(f"expected a directory: {source}")
        rsync_copy(source, destination)
        shutil.rmtree(source)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(destination, target_is_directory=True)
    if source.resolve() != destination.resolve():
        raise RuntimeError(f"link verification failed: {source}")
    print(f"Linked: {source} -> {destination}")


def command_setup(args, plan: dict) -> None:
    ensure_no_jobs(args.allow_active_jobs)
    scratch = scratch_root(args, plan)
    print(f"Scratch root: {scratch}")
    for raw in plan["active_run_links"]:
        relative = relative_repo_path(raw)
        migrate_directory(ROOT / relative, scratch / relative, apply=args.apply)
    migrate_directory(ROOT / ".cache", scratch / "cache", apply=args.apply)
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing every path.")


def verify_link(source: Path, destination: Path) -> None:
    if not source.is_symlink():
        raise SystemExit(f"required scratch link is missing: {source}")
    if source.resolve(strict=False) != destination.resolve(strict=False):
        raise SystemExit(f"scratch link has wrong target: {source} -> {source.resolve(strict=False)}")
    if not destination.is_dir() or not os.access(destination, os.W_OK | os.X_OK):
        raise SystemExit(f"scratch target is not writable/searchable: {destination}")


def command_verify(args, plan: dict) -> None:
    scratch = scratch_root(args, plan)
    for raw in plan["active_run_links"]:
        relative = relative_repo_path(raw)
        verify_link(ROOT / relative, scratch / relative)
        print(f"OK {relative} -> {scratch / relative}")
    verify_link(ROOT / ".cache", scratch / "cache")
    print(f"OK .cache -> {scratch / 'cache'}")
    for required in [ROOT / "outputs/runtime_configs", ROOT / "outputs/baseline"]:
        if required.is_symlink():
            raise SystemExit(f"permanent project path must not be a symlink: {required}")
        if not required.is_dir():
            raise SystemExit(f"required permanent project path is missing: {required}")
        print(f"OK permanent {required.relative_to(ROOT)}")
    print("SCRATCH STORAGE VERIFIED")


def command_verify_run(args, plan: dict) -> None:
    scratch = scratch_root(args, plan)
    relative = relative_repo_path(f"outputs/{args.stage}/{args.run_id}")
    verify_link(ROOT / relative, scratch / relative)
    print(f"SCRATCH RUN VERIFIED: {relative} -> {scratch / relative}")


def command_link_run(args, plan: dict) -> None:
    ensure_no_jobs(args.allow_active_jobs)
    scratch = scratch_root(args, plan)
    relative = relative_repo_path(f"outputs/{args.stage}/{args.run_id}")
    migrate_directory(ROOT / relative, scratch / relative, apply=args.apply)
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the path.")


def command_archive(args, plan: dict) -> None:
    del plan
    relative = relative_repo_path(f"outputs/{args.stage}/{args.run_id}")
    source = ROOT / relative
    complete, reason = completed_run(source, args.expected_workers)
    if not complete:
        raise SystemExit(f"refusing to archive incomplete run {relative}: {reason}")
    destination = ROOT / "outputs/final" / args.stage / args.run_id
    if destination.is_symlink():
        raise SystemExit(f"permanent archive must not be a symlink: {destination}")
    print(f"Archive: {source.resolve()} -> {destination}")
    if not args.apply:
        print("Dry run only. Re-run with --apply after reviewing the path.")
        return
    rsync_copy(source.resolve(), destination, exclude_checkpoints=True)
    files = [item for item in destination.rglob("*") if item.is_file()]
    manifest = {
        "stage": args.stage,
        "run_id": args.run_id,
        "source": str(source.resolve()),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "excluded": ["checkpoints/", "temporary files"],
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }
    (destination / ARCHIVE_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ARCHIVE VERIFIED: {destination}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    result.add_argument("--scratch-root", type=Path)
    subparsers = result.add_subparsers(dest="command", required=True)

    subparsers.add_parser("plan")

    prune = subparsers.add_parser("prune-completed-checkpoints")
    prune.add_argument("--apply", action="store_true")
    prune.add_argument("--allow-active-jobs", action="store_true")

    prune_temporary = subparsers.add_parser("prune-temporary-files")
    prune_temporary.add_argument("--apply", action="store_true")
    prune_temporary.add_argument("--allow-active-jobs", action="store_true")

    setup = subparsers.add_parser("setup")
    setup.add_argument("--apply", action="store_true")
    setup.add_argument("--allow-active-jobs", action="store_true")

    subparsers.add_parser("verify")

    verify_run = subparsers.add_parser("verify-run")
    verify_run.add_argument("--stage", required=True)
    verify_run.add_argument("--run-id", required=True)

    link_run = subparsers.add_parser("link-run")
    link_run.add_argument("--stage", required=True)
    link_run.add_argument("--run-id", required=True)
    link_run.add_argument("--apply", action="store_true")
    link_run.add_argument("--allow-active-jobs", action="store_true")

    archive = subparsers.add_parser("archive-run")
    archive.add_argument("--stage", required=True)
    archive.add_argument("--run-id", required=True)
    archive.add_argument("--expected-workers", type=int, default=5)
    archive.add_argument("--apply", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    plan = load_plan(args.plan)
    commands = {
        "plan": command_plan,
        "prune-completed-checkpoints": command_prune,
        "prune-temporary-files": command_prune_temporary,
        "setup": command_setup,
        "verify": command_verify,
        "verify-run": command_verify_run,
        "link-run": command_link_run,
        "archive-run": command_archive,
    }
    commands[args.command](args, plan)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"STORAGE ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
