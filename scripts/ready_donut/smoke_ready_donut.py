#!/usr/bin/env python3
"""Runtime-only GPU smoke using the unchanged 35dd6c1 production implementations.

Subset selection and two-step budgets exist only in isolated smoke configs.
No production template, source table, queued Qwen run or tracked file is edited.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(path, expected=None):
    if expected and digest(path) != expected:
        raise RuntimeError("Submission manifest changed")
    manifest = json.loads(path.read_text())
    root = Path(manifest["root"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != manifest["git_commit"]:
        raise RuntimeError("Code commit changed after submission")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip():
        raise RuntimeError("Tracked source changed after submission")
    if digest(manifest["helper"]) != manifest["helper_sha256"]:
        raise RuntimeError("Smoke helper changed")
    for job in manifest["jobs"]:
        if digest(job["config_path"]) != job["config_sha256"]:
            raise RuntimeError("Production runtime config changed")
    return manifest


class SmokeInterruption(RuntimeError):
    pass


def run_smoke(manifest_path, manifest):
    import torch
    import numpy as np
    from PIL import Image

    root = Path(manifest["root"])
    sys.path.insert(0, str(root / "src"))
    from checkpoints import atomic_json
    from stages import collected_attacks, jpeg_eot_attack
    from stages.defenses import _load_clean_pages
    from stages.defenses import _load_attacked_pages
    from experiment_utils import load_collected_image

    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name():
        raise RuntimeError("This smoke requires L40S")
    results = []
    for job in manifest["jobs"]:
        production = json.loads(Path(job["config_path"]).read_text())
        smoke = copy.deepcopy(production)
        smoke["run_id"] = production["run_id"] + "_runtime_smoke_v1"
        smoke["attack"].update(max_steps=2, checkpoint_interval=1)
        smoke["smoke_only"] = True
        config_path = root / "outputs/runtime_configs" / f"{smoke['run_id']}.json"
        run = root / "outputs" / job["stage"] / smoke["run_id"]
        scratch = (
            Path(manifest["scratch_root"]) / "outputs" / job["stage"] / smoke["run_id"]
        )
        if run.is_symlink():
            if run.resolve() != scratch.resolve():
                raise RuntimeError("Smoke scratch link mismatch")
        elif run.exists():
            raise RuntimeError("Refusing to replace an existing smoke directory")
        else:
            scratch.mkdir(parents=True, exist_ok=True)
            run.parent.mkdir(parents=True, exist_ok=True)
            run.symlink_to(scratch, target_is_directory=True)
        if job["stage"] == "donut_attack":
            collected_attacks.preflight(production, root, "donut")
            manifest_frame, selected = collected_attacks._load_candidates(
                root, production, "donut"
            )
            selected = selected.iloc[:1].copy()
            smoke["smoke_only_document_ids"] = list(selected.index)
            worker_ids = [0]
            expected_records = 4
            clean_pages = {
                rid: [load_collected_image(root, manifest_frame.loc[rid])]
                for rid in selected.index
            }
        else:
            jpeg_eot_attack.preflight(production, root)
            selected = (
                jpeg_eot_attack._load_plan(root, production)
                .groupby("dataset_name", group_keys=False)
                .head(1)
                .sort_index()
            )
            if set(selected.dataset_name) != {"cord_v2", "sroie"}:
                raise RuntimeError("Smoke must cover both receipt datasets")
            smoke["attack"]["evaluation_interval"] = 1
            smoke["smoke_only_document_ids"] = list(selected.index)
            worker_ids = [0, 1]
            expected_records = 6
            clean_pages = _load_clean_pages(
                root, production, selected.experiment_split.to_dict()
            )
        if config_path.exists():
            if json.loads(config_path.read_text()) != smoke:
                raise RuntimeError("Existing smoke config differs")
        else:
            atomic_json(config_path, smoke)
        if job["stage"] == "donut_attack":
            original = collected_attacks._load_candidates
            try:
                collected_attacks._load_candidates = lambda *args, **kwargs: (
                    manifest_frame,
                    selected,
                )
                collected_attacks.run_donut(smoke, config_path, "user1", 0, 5)
                # Real stage rerun must preserve completed record bytes.
                before = {
                    str(p): digest(p) for p in run.glob("shards/*/records/*.json")
                }
                collected_attacks.run_donut(smoke, config_path, "user1", 0, 5)
                if before != {
                    str(p): digest(p) for p in run.glob("shards/*/records/*.json")
                }:
                    raise RuntimeError("Collected record resume changed outputs")
            finally:
                collected_attacks._load_candidates = original
        else:
            import attacks.jpeg_eot as engine

            original_plan = jpeg_eot_attack._load_plan
            original_checkpoint = engine.checkpoint_state
            try:
                jpeg_eot_attack._load_plan = lambda *args, **kwargs: selected
                if not list(run.glob("shards/worker_00/records/*.json")):

                    def interrupt_after_checkpoint(*args, **kwargs):
                        original_checkpoint(*args, **kwargs)
                        raise SmokeInterruption(
                            "Intentional smoke interruption after durable checkpoint"
                        )

                    engine.checkpoint_state = interrupt_after_checkpoint
                    try:
                        jpeg_eot_attack.run(smoke, config_path, "user1", 0, 5)
                    except SmokeInterruption:
                        print("Checkpoint interruption exercised", flush=True)
                    finally:
                        engine.checkpoint_state = original_checkpoint
                for worker in worker_ids:
                    jpeg_eot_attack.run(
                        smoke, config_path, f"user{worker+1}", worker, 5
                    )
            finally:
                jpeg_eot_attack._load_plan = original_plan
                engine.checkpoint_state = original_checkpoint
        records = [
            json.loads(p.read_text())
            for p in sorted(run.glob("shards/*/records/*.json"))
        ]
        if len(records) != expected_records or any(
            r["status"] != "complete" for r in records
        ):
            raise RuntimeError(f"Smoke records incomplete or failed: {run}")
        if set(r["record_id"] for r in records) != set(selected.index):
            raise RuntimeError("Smoke selected documents differ")
        if job["stage"] == "jpeg_eot_attack" and not any(
            r["checkpoint_resumed"] for r in records
        ):
            raise RuntimeError("No adaptive checkpoint resume was exercised")
        for record in records:
            if record["git_commit"] != manifest["git_commit"] or record[
                "config_sha256"
            ] != digest(config_path):
                raise RuntimeError("Smoke record provenance mismatch")
            if not all(
                math.isfinite(float(record[name]))
                for name in ["linf_255", "lpips", "ssim"]
            ):
                raise RuntimeError("Nonfinite smoke metrics")
            if job["stage"] == "donut_attack":
                path = root / record["image_path"]
                if digest(path) != record["image_sha256"]:
                    raise RuntimeError("Collected PNG hash mismatch")
                with Image.open(path) as image:
                    adversarial = [image.convert("RGB").copy()]
            else:
                for item in record["images"]:
                    if digest(item["path"]) != item["sha256"]:
                        raise RuntimeError("Adaptive PNG hash mismatch")
                record["source_root"] = str(root)
                # This helper accepts row-style attribute access for legacy image_path.
                import pandas as pd

                adversarial = _load_attacked_pages(pd.Series(record), root)
            clean = clean_pages[record["record_id"]]
            if len(clean) != len(adversarial) or any(
                a.size != c.size for a, c in zip(adversarial, clean)
            ):
                raise RuntimeError("Smoke PNG dimensions changed")
            for a, c in zip(adversarial, clean):
                if (
                    np.abs(
                        np.asarray(a, dtype=np.int16) - np.asarray(c, dtype=np.int16)
                    ).max()
                    > record["epsilon_255"]
                ):
                    raise RuntimeError("Reloaded smoke PNG exceeds epsilon")
        results.append(
            dict(
                run_id=job["run_id"],
                smoke_run_id=smoke["run_id"],
                records=len(records),
                status="passed",
            )
        )
    # No success receipt is written until every requested smoke passed.
    validate_manifest(manifest_path)
    atomic_json(
        manifest_path.with_name("ready_donut_v1_smoke_pass.json"),
        dict(
            manifest_sha256=digest(manifest_path),
            git_commit=manifest["git_commit"],
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            results=results,
        ),
    )
    print(
        "ALL REQUESTED DONUT SMOKES PASSED. This is compatibility validation, not an efficacy result.",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--expected-manifest-sha256")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    manifest = validate_manifest(a.manifest, a.expected_manifest_sha256)
    if a.verify:
        receipt = json.loads(
            a.manifest.with_name("ready_donut_v1_smoke_pass.json").read_text()
        )
        if (
            receipt["manifest_sha256"] != digest(a.manifest)
            or receipt["git_commit"] != manifest["git_commit"]
        ):
            raise RuntimeError("Smoke success receipt differs from production inputs")
        if {r["run_id"] for r in receipt["results"] if r["status"] == "passed"} != {
            j["run_id"] for j in manifest["jobs"]
        }:
            raise RuntimeError("Smoke success receipt is incomplete")
        print("Matching GPU smoke verified; production may start")
    else:
        run_smoke(a.manifest, manifest)


if __name__ == "__main__":
    main()
