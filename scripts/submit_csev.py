#!/usr/bin/env python3
"""Submit CSEV dependency chain, or explicitly resume one inactive worker."""
import argparse
import fcntl
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import atomic_json
from csev.study import read_config, preflight
from csev.runner import output, calibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument(
        "--resume-phase", choices=["development", "evaluation", "adaptive", "replay"]
    )
    p.add_argument("--worker", choices=["all", "0", "1", "2", "3", "4"])
    a = p.parse_args()
    config_path = a.config.resolve()
    c, plan = read_config(config_path, ROOT)
    preflight(c, plan, ROOT)
    if bool(a.resume_phase) != (a.worker is not None):
        p.error("resume requires both --resume-phase and --worker")
    ledger_path = ROOT / "outputs/runtime_configs" / (c["run_id"] + "_submissions.json")
    lock = ledger_path.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if ledger and not a.resume_phase:
        print(json.dumps(ledger, indent=2))
        raise SystemExit(
            "Submission ledger exists. Inspect status; resume only an inactive incomplete worker. No duplicate jobs submitted."
        )
    phases = (
        [a.resume_phase]
        if a.resume_phase
        else ["development", "evaluation", "adaptive", "replay"]
    )
    queue = subprocess.check_output(
        ["squeue", "-h", "-o", "%j"], text=True
    ).splitlines()
    if a.resume_phase:
        if a.resume_phase != "development":
            calibration(c)
        if (
            a.resume_phase == "replay"
            and not (output(c) / "adaptive/merged.json").exists()
        ):
            raise ValueError("adaptive merge required")
    (ROOT / "logs").mkdir(exist_ok=True)
    ids = {}
    for phase in phases:
        name = f'pp_{c["run_id"]}_{phase}'
        if name in queue:
            raise ValueError(f"{phase} has active workers; do not duplicate them")
        worker = (
            a.worker
            if a.resume_phase and a.worker != "all"
            else '"$SLURM_ARRAY_TASK_ID"'
        )
        script = (
            f"set -e\ncd {shlex.quote(str(ROOT))}\nsource scripts/activate_paperwork.sh\n"
            f"python -u scripts/csev_study.py run --config {shlex.quote(str(config_path))} --phase {phase} --worker {worker}\n"
        )
        cmd = [
            "sbatch",
            "--parsable",
            "--account=yzhao010_1531",
            "--partition=gpu",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=12:00:00",
            "--gpus-per-task=l40s:1",
            f"--job-name={name}",
            f"--output={ROOT}/logs/{name}_%A_%a.out",
            f"--error={ROOT}/logs/{name}_%A_%a.err",
        ]
        if not a.resume_phase or a.worker == "all":
            cmd += ["--array=0-4"]
        if not a.resume_phase:
            parent = {
                "evaluation": "development",
                "adaptive": "development",
                "replay": "adaptive",
            }.get(phase)
            if parent:
                cmd += [f'--dependency=afterok:{ids.get(parent,"PARENT_JOB_ID")}']
        cmd += ["--wrap", "bash -lc " + shlex.quote(script)]
        print(shlex.join(cmd), flush=True)
        if not a.submit:
            ids[phase] = phase.upper()
            continue
        key = f"{phase}_worker_{a.worker}" if a.resume_phase else phase
        if key in ledger and ledger[key].get("status") == "submitting":
            raise ValueError(
                "Ambiguous previous submission: inspect squeue before reconciling ledger"
            )
        ledger[key] = dict(status="submitting", command=cmd)
        atomic_json(ledger_path, ledger)
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode:
            ledger[key].update(status="submission_failed", stderr=r.stderr)
            atomic_json(ledger_path, ledger)
            raise RuntimeError(r.stderr)
        job = r.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Unexpected sbatch response; inspect ledger/queue")
        ids[phase] = job
        ledger[key].update(status="submitted", job_id=job)
        atomic_json(ledger_path, ledger)
        print(f"SUBMITTED {phase}: {job}", flush=True)
    if not a.submit:
        print("Dry run only. Add --submit to queue these jobs.")


if __name__ == "__main__":
    main()
