#!/usr/bin/env python3
"""Queue missing Donut studies behind a real L40S smoke without a Git update.

Copy this directory to outputs/runtime_tools/ready_donut_v1 on CARC. Run from
repository root, after activating paperwork. Default is preparation/check only.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

JOBS = [
    dict(
        template="donut_collected",
        stage="donut_attack",
        run_id="donut_collected_external_v1",
        config="donut_collected.json",
        name="pp_donut_attack_collected_receipts",
        records=360,
    ),
    dict(
        template="jpeg_eot_donut",
        stage="jpeg_eot_attack",
        run_id="donut_jpeg_eot_public_full_v1",
        config="donut_jpeg_eot_public_full_v1.json",
        name="pp_jpeg_eot_attack_donut_uniform_eot",
        records=363,
    ),
    dict(
        template="jpeg_eot_donut_hardview",
        stage="jpeg_eot_attack",
        run_id="donut_frequency_hardview_public_full_v1",
        config="donut_hardview_public_full_v1.json",
        name="pp_jpeg_eot_attack_donut_hardview_eot",
        records=363,
    ),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output(command, **kwargs):
    return subprocess.check_output(
        [str(x) for x in command], text=True, **kwargs
    ).strip()


def run(command, **kwargs):
    print(shlex.join([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def active_jobs():
    raw = output(
        ["squeue", "-h", "-u", f"{os.environ['USER']},kbali", "-o", "%A|%j|%T"]
    )
    return [
        dict(zip(["id", "name", "state"], line.split("|", 2)))
        for line in raw.splitlines()
        if line
    ]


def matching_active(job, active):
    # Individual worker submissions append _wN; arrays use the exact name.
    return [
        row
        for row in active
        if row["name"] == job["name"] or row["name"].startswith(job["name"] + "_w")
    ]


def has_progress(path):
    return (
        any(path.glob("shards/worker_*/records/*.json"))
        or any(path.glob("shards/worker_*/WORKER_COMPLETE.json"))
        or any(path.rglob("*.pt"))
    )


def sbatch_from_launcher(text, smoke_id, commit):
    lines = [line for line in text.splitlines() if line.startswith("sbatch ")]
    if len(lines) != 1:
        raise RuntimeError("launcher did not return exactly one sbatch command")
    args = shlex.split(lines[0])
    if "--gpus-per-task=l40s:1" not in args or "--array=0-4" not in args:
        raise RuntimeError("launcher changed the L40S/five-worker contract")
    for i, value in enumerate(args):
        if value.startswith("--export="):
            args[i] = value + f",PP_EXPECTED_GIT_SHA={commit}"
            break
    else:
        raise RuntimeError("missing worker export contract")
    return args[:-1] + [f"--dependency=afterok:{smoke_id}"] + args[-1:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    root = Path.cwd().resolve()
    if not (root / "scripts/submit_stage.sh").is_file():
        raise SystemExit("Run from the repository root")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Activate the CARC paperwork Python 3.11 environment")
    commit = output(["git", "rev-parse", "HEAD"])
    if not commit.startswith("35dd6c1"):
        raise SystemExit(
            "This bundle was reviewed for 35dd6c1; inspect before using another commit"
        )
    if output(["git", "status", "--porcelain"]):
        raise SystemExit(
            "Tracked checkout must remain clean; store these helpers under ignored outputs/"
        )
    root.joinpath("outputs/runtime_configs").mkdir(parents=True, exist_ok=True)
    lock = root / "outputs/runtime_configs/ready_donut_v1.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "outputs/runtime_configs/ready_donut_v1_submissions.json"
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {"git_commit": commit, "submitted": {}}
        )
        active = active_jobs()
        scratch = Path(
            os.environ.get(
                "PP_SCRATCH_ROOT", f"/scratch1/{os.environ['USER']}/poisoned-paperwork"
            )
        )
        if not scratch.is_absolute() or not str(scratch).startswith("/scratch1/"):
            raise SystemExit("Expected an absolute /scratch1 working-storage root")
        os.environ["PP_SCRATCH_ROOT"] = str(scratch)
        pending = []
        for original in JOBS:
            job = dict(original)
            run_path = root / "outputs" / job["stage"] / job["run_id"]
            live = matching_active(job, active)
            if live:
                print(
                    "SKIP active:", job["run_id"], sorted({row["id"] for row in live})
                )
                continue
            summary_path = run_path / "merged/summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if (
                    summary.get("records") == job["records"]
                    and summary.get("failures") == 0
                    and len(list(run_path.glob("shards/worker_*/WORKER_COMPLETE.json")))
                    == 5
                ):
                    print("SKIP completed:", job["run_id"])
                    continue
                raise SystemExit(
                    f"Existing incomplete/failed merged run needs inspection: {run_path}"
                )
            if job["run_id"] in state["submitted"]:
                raise SystemExit(
                    f"Previously submitted {job['run_id']} as {state['submitted'][job['run_id']]}; inspect/resume instead of resubmitting all"
                )
            scratch_run = scratch / "outputs" / job["stage"] / job["run_id"]
            if has_progress(run_path) or has_progress(scratch_run):
                raise SystemExit(
                    f'Existing progress needs worker-specific resume: {job["run_id"]}'
                )
            if run_path.exists() and not run_path.is_symlink():
                raise SystemExit(
                    f"Unexpected existing directory; will not migrate it automatically: {run_path}"
                )
            config = root / "outputs/runtime_configs" / job["config"]
            if not config.exists():
                command = [
                    sys.executable,
                    "scripts/prepare_stage_config.py",
                    job["template"],
                    "--output",
                    str(config),
                ]
                if job["stage"] == "jpeg_eot_attack":
                    command += [
                        "--source",
                        f"donut_public={root}/outputs/imported/donut_public_validation/merged/attacks_for_defenses.parquet",
                    ]
                run(command)
            settings = json.loads(config.read_text())
            if settings["run_id"] != job["run_id"] or settings["model_key"] != "donut":
                raise SystemExit(f"Unexpected runtime config: {config}")
            if (
                settings["stage"] != job["stage"]
                or settings.get("require_clean_git") is not True
                or settings.get("document_limit_per_dataset") is not None
            ):
                raise SystemExit(
                    "Runtime stage, Git requirement or full-cohort setting differs"
                )
            if settings["attack"]["max_steps"] != 100:
                raise SystemExit("Expected full 100-step production attack")
            if (
                settings["model"]["model_id"]
                != "naver-clova-ix/donut-base-finetuned-docvqa"
                or settings["model"]["revision"]
                != "b19d2e332684b0e2d35d9144ce34047767335cf8"
            ):
                raise SystemExit("Donut model pin differs")
            run(
                [
                    sys.executable,
                    "scripts/preflight_stage.py",
                    "--stage",
                    job["stage"],
                    "--config",
                    config,
                ]
            )
            job.update(config_path=str(config), config_sha256=digest(config))
            # These frozen studies contain 90 or 121 source-clean-correct documents.
            sys.path.insert(0, str(root / "src"))
            if job["stage"] == "donut_attack":
                from stages.collected_attacks import _load_candidates

                if len(_load_candidates(root, settings, "donut")[1]) != 90:
                    raise SystemExit("Expected 90 collected Donut candidates")
            pending.append(job)
        if not pending:
            print(
                "No missing, unsubmitted Donut runs. Existing jobs were left unchanged."
            )
            return
        for job in pending:
            print("READY:", job["run_id"], job["records"], "attacks")
        if not a.submit:
            print(
                "Checks passed. Run again with --submit to queue smoke and dependent arrays."
            )
            return
        if state.get("smoke_job_id"):
            raise SystemExit(
                f"Earlier bundle smoke/submission exists: {state_path}. Inspect it before any retry"
            )
        helper = Path(__file__).resolve().with_name("smoke_ready_donut.py")
        if not helper.is_file():
            raise SystemExit("Missing companion smoke_ready_donut.py")
        for job in pending:
            # Manual audit is limited to these unqueued runs with no saved progress.
            run(
                [
                    sys.executable,
                    "scripts/manage_carc_storage.py",
                    "link-run",
                    "--stage",
                    job["stage"],
                    "--run-id",
                    job["run_id"],
                    "--allow-active-jobs",
                    "--apply",
                ]
            )
            run(
                [
                    sys.executable,
                    "scripts/manage_carc_storage.py",
                    "verify-run",
                    "--stage",
                    job["stage"],
                    "--run-id",
                    job["run_id"],
                ]
            )
        manifest = root / "outputs/runtime_configs/ready_donut_v1_manifest.json"
        if manifest.exists():
            raise SystemExit("Existing submission manifest; inspect it before retrying")
        manifest_data = dict(
            root=str(root),
            git_commit=commit,
            jobs=pending,
            helper=str(helper),
            helper_sha256=digest(helper),
            scratch_root=str(scratch),
        )
        atomic(manifest, manifest_data)
        smoke_dir = scratch / "outputs/ready_donut_smoke_v1"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        batch = root / "outputs/runtime_configs/ready_donut_v1_smoke.sbatch"
        batch.write_text(
            "#!/usr/bin/env bash\nset -e\ncd "
            + shlex.quote(str(root))
            + "\nsource scripts/activate_paperwork.sh\n"
            + "export TMPDIR="
            + shlex.quote(str(smoke_dir / "tmp"))
            + '\nmkdir -p "$TMPDIR"\n'
            + "python "
            + shlex.quote(str(helper))
            + " --manifest "
            + shlex.quote(str(manifest))
            + "\n"
        )
        root.joinpath("logs").mkdir(exist_ok=True)
        # Persist an intent before external submission; a crash never silently duplicates work.
        state["smoke_job_id"] = "SUBMISSION_IN_PROGRESS"
        atomic(state_path, state)
        smoke_id = output(
            [
                "sbatch",
                "--parsable",
                "--account=yzhao010_1531",
                "--partition=gpu",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=02:00:00",
                "--gpus-per-task=l40s:1",
                "--job-name=pp_ready_donut_smoke_v1",
                f"--chdir={root}",
                f"--output={root}/logs/pp_ready_donut_smoke_%j.out",
                f"--error={root}/logs/pp_ready_donut_smoke_%j.err",
                batch,
            ]
        ).split(";")[0]
        if not smoke_id.isdigit():
            raise RuntimeError("Unexpected Slurm smoke ID")
        state["smoke_job_id"] = smoke_id
        atomic(state_path, state)
        print("SMOKE JOB:", smoke_id, flush=True)
        production_wrapper = (
            root / "outputs/runtime_configs/ready_donut_v1_worker.sbatch"
        )
        production_wrapper.write_text(
            "#!/usr/bin/env bash\nset -e\ncd "
            + shlex.quote(str(root))
            + "\nsource scripts/activate_paperwork.sh\n"
            + "python "
            + shlex.quote(str(helper))
            + " --verify --manifest "
            + shlex.quote(str(manifest))
            + " --expected-manifest-sha256 "
            + digest(manifest)
            + "\nexec bash scripts/slurm/gpu_stage.sbatch\n"
        )
        for job in pending:
            rendered = output(
                [
                    "bash",
                    "scripts/submit_stage.sh",
                    job["stage"],
                    "all",
                    job["config_path"],
                    "--dry-run",
                ]
            )
            command = sbatch_from_launcher(rendered, smoke_id, commit)
            command[-1] = str(production_wrapper)
            state["submitted"][job["run_id"]] = "SUBMISSION_IN_PROGRESS"
            atomic(state_path, state)
            job_id = output(command).split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError("Unexpected Slurm array ID")
            state["submitted"][job["run_id"]] = job_id
            atomic(state_path, state)
            print(
                "QUEUED:",
                job["run_id"],
                "array",
                job_id,
                "after successful smoke",
                smoke_id,
                flush=True,
            )
        print("Submission ledger:", state_path)
        print(
            "Do not pull/change this checkout. Failed smoke leaves dependent arrays blocked; inspect logs before retrying."
        )


if __name__ == "__main__":
    main()
