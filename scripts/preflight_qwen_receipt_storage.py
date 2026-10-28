#!/usr/bin/env python3
"""Verify cross-account receipt storage, locking, and atomic publication on CARC.

Every teammate runs this command with the same session ID and Git commit.  The
last caller adds ``--require-all`` to prove that five distinct Unix accounts
published the fixed worker identities into one shared writable namespace.
This is a storage-only CPU check: it does not import Torch, load Qwen, or run an
attack.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SESSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shared_directory(path: Path, *, parent: Path | None = None) -> None:
    """Create or validate a group-writable setgid directory."""
    try:
        path.mkdir(mode=0o2770)
    except FileExistsError:
        pass
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"shared storage path is not a real directory: {path}")
    resolved = path.resolve()
    if parent is not None and not resolved.is_relative_to(parent.resolve()):
        raise RuntimeError(f"shared storage path escapes its parent: {path}")
    details = path.stat()
    mode = stat.S_IMODE(details.st_mode)
    if not mode & stat.S_IWGRP or not mode & stat.S_IXGRP:
        raise PermissionError(
            f"shared storage directory must be group-writable/searchable: {path} "
            f"(mode={mode:o})"
        )
    if not mode & stat.S_ISGID:
        raise PermissionError(
            f"shared storage directory must have the setgid bit: {path} "
            f"(mode={mode:o})"
        )


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _verify_marker(path: Path, expected: dict) -> dict:
    marker = _read_object(path)
    for field, value in expected.items():
        if marker.get(field) != value:
            raise RuntimeError(
                f"storage marker differs in {field}: {path} "
                f"({marker.get(field)!r} != {value!r})"
            )
    mode = stat.S_IMODE(path.stat().st_mode)
    if not mode & stat.S_IRGRP:
        raise PermissionError(f"storage marker is not group-readable: {path}")
    return marker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_attack/receipt_smoke_saaketh.json",
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require valid markers from all five distinct Unix accounts.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if SESSION_PATTERN.fullmatch(args.session_id) is None:
        raise SystemExit(
            "--session-id must be a lowercase filesystem-safe slug of at most 128 characters"
        )
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_git_commit) is None:
        raise SystemExit("--expected-git-commit must be an exact lowercase 40-character SHA")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    if not config_path.is_file() or not config_path.is_relative_to(ROOT):
        raise SystemExit("receipt storage preflight config must be a file inside this checkout")
    config = _read_object(config_path)

    from checkpoints import sha256_file
    from stages.qwen_attack_receipt import (
        _require_clean_git,
        _validate_attack_config,
        resolve_output_root,
    )
    from team import TEAM, get_team_member

    _validate_attack_config(config)
    if config.get("mode") != "smoke":
        raise SystemExit("storage preflight uses the unlocked receipt smoke config")
    _require_clean_git(config, ROOT)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if actual_commit != args.expected_git_commit:
        raise SystemExit(
            f"Git commit mismatch: expected {args.expected_git_commit}, found {actual_commit}"
        )

    member = get_team_member(args.user_id)
    run_root = resolve_output_root(config, ROOT)
    output_base = run_root.parents[1]
    output_mode = stat.S_IMODE(output_base.stat().st_mode)
    if not output_mode & stat.S_IWGRP or not output_mode & stat.S_IXGRP:
        raise PermissionError(
            "PP_RECEIPT_OUTPUT_ROOT must be group-writable/searchable before the "
            f"cross-account preflight (mode={output_mode:o})"
        )
    if not output_mode & stat.S_ISGID:
        raise PermissionError(
            "PP_RECEIPT_OUTPUT_ROOT must have setgid enabled before the "
            f"cross-account preflight (mode={output_mode:o})"
        )

    previous_umask = os.umask(0o002)
    try:
        preflight_root = output_base / "_receipt_storage_preflight"
        _shared_directory(preflight_root, parent=output_base)
        session = preflight_root / args.session_id
        _shared_directory(session, parent=preflight_root)
        markers = session / "markers"
        probes = session / "probes"
        _shared_directory(markers, parent=session)
        _shared_directory(probes, parent=session)

        lock_path = session / ".storage.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            worker_probe = probes / f"worker_{member.worker_id:02d}"
            _shared_directory(worker_probe, parent=probes)
            completed_probe = worker_probe / "atomic-directory-rename-complete"
            if not completed_probe.exists():
                temporary_probe = worker_probe / f".rename.tmp.{os.getpid()}"
                temporary_probe.mkdir(mode=0o770)
                (temporary_probe / "payload.txt").write_text(
                    f"{args.user_id}\n", encoding="utf-8"
                )
                os.replace(temporary_probe, completed_probe)
            if not completed_probe.is_dir():
                raise RuntimeError("atomic directory publication did not produce a directory")

            marker_path = markers / f"worker_{member.worker_id:02d}.json"
            stable = {
                "schema_version": 1,
                "verification": "qwen_receipt_shared_storage_v1",
                "session_id": args.session_id,
                "user_id": member.user_id,
                "name": member.name,
                "worker_id": member.worker_id,
                "git_commit": actual_commit,
                "config_path": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256_file(config_path),
                "output_base": str(output_base),
                "unix_uid": os.getuid(),
                "unix_gid": os.getgid(),
            }
            if marker_path.exists():
                marker = _verify_marker(marker_path, stable)
            else:
                marker = {
                    **stable,
                    "hostname": socket.gethostname(),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "python_prefix": str(Path(sys.prefix).resolve()),
                    "verified_at": _now(),
                }
                _atomic_json(marker_path, marker)
                marker = _verify_marker(marker_path, stable)

            all_markers: list[dict] = []
            missing_workers: list[int] = []
            for expected_member in TEAM.values():
                candidate = markers / f"worker_{expected_member.worker_id:02d}.json"
                if not candidate.is_file():
                    missing_workers.append(expected_member.worker_id)
                    continue
                all_markers.append(_verify_marker(candidate, {
                    "schema_version": 1,
                    "verification": "qwen_receipt_shared_storage_v1",
                    "session_id": args.session_id,
                    "user_id": expected_member.user_id,
                    "name": expected_member.name,
                    "worker_id": expected_member.worker_id,
                    "git_commit": actual_commit,
                    "config_path": str(config_path.relative_to(ROOT)),
                    "config_sha256": sha256_file(config_path),
                    "output_base": str(output_base),
                }))

            if args.require_all:
                if missing_workers:
                    raise RuntimeError(
                        f"shared storage preflight is missing workers: {missing_workers}"
                    )
                unix_uids = {value.get("unix_uid") for value in all_markers}
                if len(unix_uids) != len(TEAM) or None in unix_uids:
                    raise RuntimeError(
                        "shared storage preflight requires five distinct Unix accounts"
                    )
                completion = {
                    "schema_version": 1,
                    "status": "pass",
                    "verification": "qwen_receipt_shared_storage_v1",
                    "session_id": args.session_id,
                    "git_commit": actual_commit,
                    "config_sha256": sha256_file(config_path),
                    "output_base": str(output_base),
                    "markers": len(all_markers),
                    "unique_unix_uids": len(unix_uids),
                    "worker_ids": sorted(value["worker_id"] for value in all_markers),
                    "completed_at": _now(),
                }
                _atomic_json(session / "COMPLETE.json", completion)

        result = {
            "status": "pass",
            "verification": "qwen_receipt_shared_storage_v1",
            "session_id": args.session_id,
            "worker_id": member.worker_id,
            "user_id": member.user_id,
            "git_commit": actual_commit,
            "output_base": str(output_base),
            "marker": str(marker_path),
            "markers_present": len(all_markers),
            "missing_workers": missing_workers,
            "all_workers_verified": not missing_workers,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
