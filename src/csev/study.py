"""Frozen study input preparation, calibration and strict completion contracts."""

import copy
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from checkpoints import atomic_json, sha256_file, sha256_text
from stages.final_protocol import load_pages, runtime_versions as base_versions


def runtime_versions():
    import importlib.metadata

    return dict(
        base_versions(),
        **{
            k: importlib.metadata.version(k)
            for k in ("lpips", "scikit-image", "matplotlib")
        },
    )


def digest(value):
    return sha256_text(json.dumps(value, sort_keys=True))


def git_commit(root):
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise RuntimeError(
            "tracked checkout must be clean; commit before freezing/running"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def select(documents, count, seed, salt):
    if len(documents) < count:
        raise ValueError(f"need {count} documents for {salt}, found {len(documents)}")
    return sorted(documents, key=lambda d: digest([seed, salt, d["record_id"]]))[:count]


def freeze_pages(pages, directory):
    if len(pages) != 1:
        raise ValueError("CSEV v1 is a single-page receipt study")
    path = directory / "page.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].convert("RGB").save(path)
    return [dict(path=str(path), sha256=sha256_file(path))]


def bounds(clean, attacked, epsilon):
    import numpy as np

    if len(clean) != len(attacked) or any(
        a.size != c.size for c, a in zip(clean, attacked)
    ):
        raise ValueError("page count or dimensions differ")
    value = max(
        int(abs(np.asarray(c, dtype="int16") - np.asarray(a, dtype="int16")).max())
        for c, a in zip(clean, attacked)
    )
    if value > epsilon:
        raise ValueError(f"saved PNG exceeds epsilon: {value}>{epsilon}")
    return value


def prepare(root, run_id, public_config, baseline_path, source_paths, scratch):
    import pandas as pd
    from stages.defenses import (
        _load_clean_pages,
        _load_attack_sources,
        _load_attacked_pages,
    )

    root = Path(root).resolve()
    if not run_id.replace("_", "").isalnum():
        raise ValueError("run ID must be alphanumeric/underscore")
    dest = root / "outputs/runtime_configs" / (run_id + ".json")
    bundle = root / "outputs/csev_inputs" / run_id
    output = root / "outputs/csev" / run_id
    scratch = Path(scratch).resolve() / run_id
    if any(p.exists() or p.is_symlink() for p in (dest, bundle, output, scratch)):
        raise ValueError("run already exists; use preflight/resume, never overwrite")
    settings = json.loads((root / "configs/csev/protocol_v1.json").read_text())
    legacy = json.loads(Path(public_config).read_text())
    pinned = json.loads((root / "configs/defenses/qwen_public_final.json").read_text())
    if legacy["model_key"] != "qwen" or legacy["model"] != pinned["model"]:
        raise ValueError("public evidence config is not the pinned Qwen model")
    pp = root / legacy["plan"]["path"]
    if sha256_file(pp) != legacy["plan"]["sha256"]:
        raise ValueError("public plan hash changed")
    public = json.loads(pp.read_text())
    dev = []
    for dataset, n in settings["development_counts"].items():
        dev += select(
            [d for d in public if d["dataset_name"] == dataset],
            n,
            settings["seed"],
            dataset,
        )
    contract = json.loads(
        (root / "data/processed/collected_receipts/manifest_contract.json").read_text()
    )
    manifest_path = root / contract["manifest"]
    if sha256_file(manifest_path) != contract["manifest_sha256"]:
        raise ValueError("collected manifest hash changed")
    frame = pd.read_csv(
        manifest_path, dtype={"clean_target": str, "attack_target": str}
    )
    if len(frame) != 114 or frame.record_id.duplicated().any():
        raise ValueError("expected 114 distinct collected receipts")
    base = pd.read_parquet(baseline_path)
    base = base.loc[base.model_key.eq("qwen")].copy()
    if base.record_id.duplicated().any() or set(base.record_id) != set(frame.record_id):
        raise ValueError("need exactly the full 114 Qwen baseline rows")
    if (
        not base.model_id.eq(pinned["model"]["model_id"]).all()
        or not base.model_revision.eq(pinned["model"]["revision"]).all()
    ):
        raise ValueError("baseline model revision mismatch")
    if (
        not base.clean_correct.isin([True, False]).all()
        or int(base.clean_correct.sum()) != 110
    ):
        raise ValueError("baseline outcomes differ from verified 110/114")
    eligible = dict(zip(base.record_id, base.clean_correct))
    external = [
        dict(
            record_id=r.record_id,
            dataset_name="collected_receipts",
            experiment_split=r.experiment_split,
            clean_target=str(r.clean_target),
            frozen_attack_target=str(r.attack_target),
            eligible=bool(eligible[r.record_id]),
            attacks=[],
        )
        for r in frame.itertuples()
    ]
    test = select(external, settings["evaluation_count"], settings["seed"], "external")
    pilot = select(
        [d for d in test if d["eligible"]],
        settings["pilot_count"],
        settings["seed"],
        "pilot",
    )
    selected = {d["record_id"]: d for d in test}
    sources = []
    pending = []
    for family in settings["families"]:
        path = Path(source_paths[family]).resolve()
        source = dict(
            name=family,
            records=str(path),
            sha256=sha256_file(path),
            artifact_root=str(root),
        )
        raw = pd.read_parquet(path)
        for column, expected_value in (
            ("model_id", pinned["model"]["model_id"]),
            ("model_revision", pinned["model"]["revision"]),
        ):
            if column not in raw or not raw[column].eq(expected_value).all():
                raise ValueError(f"{family}: missing or mismatched {column}")
        if not raw.status.eq("complete").all():
            raise ValueError(f"incomplete source: {family}")
        attacks = _load_attack_sources(
            root,
            dict(attack_sources=[source], attack_epsilons_255=settings["epsilons"]),
        )
        required = {
            (rid, e) for rid, ok in eligible.items() if ok for e in settings["epsilons"]
        }
        seen = list(zip(attacks.record_id, attacks.epsilon_255.astype(int)))
        if len(seen) != len(required) or set(seen) != required:
            raise ValueError(f"{family} must contain the complete 330-row source grid")
        for _, row in attacks.iterrows():
            if row.record_id not in selected:
                continue
            d = selected[row.record_id]
            if (
                str(row.clean_target) != d["clean_target"]
                or str(row.attack_target) != d["frozen_attack_target"]
            ):
                raise ValueError("source targets differ from frozen manifest")
            a = dict(family=family, epsilon=int(row.epsilon_255))
            d["attacks"].append(a)
            pending.append((d, a, row))
        sources.append(source)
    clean = _load_clean_pages(
        root,
        dict(collected_manifest_contract=contract),
        {d["record_id"]: d["experiment_split"] for d in test},
    )
    # Validate sources before creating a potentially large immutable bundle.
    commit = git_commit(root)
    versions = runtime_versions()
    for d in dev:
        d["attacks"] = []
        d["images"] = freeze_pages(
            load_pages(d["images"], root),
            bundle / "development" / digest(d["record_id"]),
        )
    for d in test:
        d["images"] = freeze_pages(
            clean[d["record_id"]], bundle / "evaluation" / digest(d["record_id"])
        )
    for d, a, row in pending:
        pages = _load_attacked_pages(row, root)
        bounds(clean[d["record_id"]], pages, a["epsilon"])
        a["images"] = freeze_pages(
            pages,
            bundle / "attacks" / digest([d["record_id"], a["family"], a["epsilon"]]),
        )
    plan = dict(
        development=dev, evaluation=test, pilot_ids=[d["record_id"] for d in pilot]
    )
    plan_path = bundle / "plan.json"
    atomic_json(plan_path, plan)
    config = dict(
        stage="csev",
        run_id=run_id,
        settings=settings,
        legacy=legacy,
        git_commit=commit,
        runtime_versions=versions,
        plan=str(plan_path),
        plan_sha256=sha256_file(plan_path),
        output=str(output),
        sources=sources,
        baseline=dict(
            path=str(Path(baseline_path).resolve()),
            sha256=sha256_file(Path(baseline_path)),
        ),
        collected_manifest=contract,
        public_config_sha256=sha256_file(Path(public_config)),
    )
    scratch.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(scratch, target_is_directory=True)
    atomic_json(dest, config)
    return dest


def read_config(path, root, check_runtime=True):
    c = json.loads(Path(path).read_text())
    if c["stage"] != "csev" or c["settings"]["version"] != 1:
        raise ValueError("unknown CSEV protocol")
    if git_commit(root) != c["git_commit"]:
        raise ValueError(
            "checkout differs from frozen study; restore its commit in a separate checkout"
        )
    if check_runtime and runtime_versions() != c["runtime_versions"]:
        raise ValueError("runtime differs from frozen study")
    if sha256_file(Path(c["plan"])) != c["plan_sha256"]:
        raise ValueError("plan changed")
    return c, json.loads(Path(c["plan"]).read_text())


def preflight(config, plan, root):
    s = config["settings"]
    if (
        Counter(d["dataset_name"] for d in plan["development"])
        != s["development_counts"]
    ):
        raise ValueError("development count changed")
    test = plan["evaluation"]
    if len(test) != s["evaluation_count"] or len(
        set(d["record_id"] for d in test)
    ) != len(test):
        raise ValueError("evaluation count changed")
    if (
        len(plan["pilot_ids"]) != s["pilot_count"]
        or len(set(plan["pilot_ids"])) != s["pilot_count"]
    ):
        raise ValueError("pilot count changed")
    if not set(plan["pilot_ids"]) <= {d["record_id"] for d in test if d["eligible"]}:
        raise ValueError("pilot eligibility differs")
    for d in plan["development"] + test:
        clean = load_pages(d["images"], root)
        for a in d["attacks"]:
            bounds(clean, load_pages(a["images"], root), a["epsilon"])
        if d in test:
            expected = (
                {(f, e) for f in s["families"] for e in s["epsilons"]}
                if d["eligible"]
                else set()
            )
            observed = [(a["family"], a["epsilon"]) for a in d["attacks"]]
            if set(observed) != expected or len(observed) != len(expected):
                raise ValueError("missing/duplicate selected attack")
    o = config["legacy"]["ocr"]
    if sha256_file(Path(o["executable"])) != o["executable_sha256"]:
        raise ValueError("Tesseract binary changed")
    for lang, sha in o["traineddata_sha256"].items():
        if sha256_file(Path(o["tessdata_dir"]) / (lang + ".traineddata")) != sha:
            raise ValueError("Tesseract language data changed")
    print(
        "CPU preflight passed: 60 development, 60 evaluation, 20 pilot documents; all PNGs checked.",
        flush=True,
    )


def calibrate(observations, settings):
    """No attack/test outcomes enter selection. Explicitly report infeasible goals."""
    from csev.defense import prediction

    n = len(observations)
    if not n:
        raise ValueError("no development observations")
    baseline = sum(r["full_prediction"] == r["clean_target"] for r in observations) / n
    candidates = sorted(
        {settings["threshold_minimum"]}
        | {
            r["gap"]
            for r in observations
            if r["gap"] is not None and r["gap"] >= settings["threshold_minimum"]
        }
    )
    curve = []
    for tau in candidates:
        preds = [prediction(r, tau) for r in observations]
        accepted = sum(p is not None for p in preds)
        correct = sum(
            p is not None and p == r["clean_target"]
            for p, r in zip(preds, observations)
        )
        curve.append(
            dict(
                threshold=tau,
                coverage=accepted / n,
                correct_answer_rate=correct / n,
                selective_accuracy=correct / accepted if accepted else None,
            )
        )
    feasible = [
        r
        for r in curve
        if r["coverage"] >= settings["minimum_coverage"]
        and r["selective_accuracy"] >= baseline - settings["accuracy_tolerance"]
    ]
    # If goals are impossible, retain the least restrictive nonnegative threshold.
    # Never tune on external outcomes or silently weaken the target.
    chosen = max(feasible, key=lambda r: r["threshold"]) if feasible else curve[0]
    return dict(
        **chosen,
        development_goal_met=bool(feasible),
        baseline_accuracy=baseline,
        rule="largest feasible threshold; otherwise minimum nonnegative threshold",
        curve=curve,
    )
