#!/usr/bin/env python3
"""Require a matching successful L40S smoke before final-protocol submission."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import sha256_file

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
s = json.loads((ROOT / "outputs/smoke" / f"{c['run_id']}.json").read_text())
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()
if (
    s["config_sha256"] != sha256_file(a.config)
    or s["git_commit"] != commit
    or "L40S" not in s["gpu"]
):
    raise SystemExit("Smoke config, code commit or hardware does not match")
if s.get("mode") != c["mode"] or not s["rows"]:
    raise SystemExit("Smoke mode/results missing")
print("Matching L40S smoke verified")
