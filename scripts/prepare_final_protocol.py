#!/usr/bin/env python3
"""Freeze a complete final cohort, source eligibility, and actual image bytes.

Source syntax MODEL:ATTACK_TYPE:PARQUET. Baselines must contain model_key,
record_id and clean_correct; never infer clean eligibility from attack success.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import atomic_json, sha256_file, sha256_text
from stages.defenses import (
    _load_clean_pages,
    _load_attack_sources,
    _load_attacked_pages,
)
from stages.final_protocol import VARIANTS, runtime_versions


def save_pages(pages, directory):
    items = []
    for index, page in enumerate(pages):
        path = directory / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.convert("RGB").save(path)
        items.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256_file(path)))
    return items


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["defense", "transfer", "adaptive"], required=True
    )
    parser.add_argument("--model", choices=["qwen", "donut", "internvl"], required=True)
    parser.add_argument("--cohort", choices=["public", "collected"], required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tessdata-dir", type=Path)
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument(
        "--objective", choices=["targeted", "availability"], default="targeted"
    )
    parser.add_argument("--queries", type=int, default=2000)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        parser.error("run ID must contain only letters, digits and underscores")
    base = json.loads((ROOT / "configs/final_protocol/protocol_v1.json").read_text())
    legacy = json.loads((ROOT / "configs/defenses/qwen_public_final.json").read_text())
    config = {k: legacy[k] for k in ["seed", "prompts", "semantic_prompts", "defense"]}
    config.update(base)
    config.update(
        mode=args.mode,
        model_key=args.model,
        run_id=args.run_id,
        ready_for_production=True,
    )
    config["model"] = config.pop("models")[args.model]
    config["runtime_versions"] = runtime_versions()
    config["adaptive"].update(objective=args.objective, queries=args.queries)
    config["adaptive"]["report_queries"] = sorted(
        set(
            [v for v in config["adaptive"]["report_queries"] if v <= args.queries]
            + [args.queries]
        )
    )
    from stages.transfer_robustness import TRANSFORMS

    config["conditions"] = (
        list(TRANSFORMS)
        if args.mode == "transfer"
        else (VARIANTS if args.mode == "defense" else ["evidence"])
    )
    bundle = ROOT / "outputs/final_inputs" / args.run_id
    destination = ROOT / "outputs/runtime_configs" / f"{args.run_id}.json"
    if bundle.exists() or destination.exists():
        parser.error(
            "immutable input bundle/config already exists; reuse it or choose a new run ID"
        )
    contract = (
        json.loads(
            (
                ROOT / "data/processed/collected_receipts/manifest_contract.json"
            ).read_text()
        )
        if args.cohort == "collected"
        else None
    )
    if contract:
        frame = pd.read_csv(
            ROOT / contract["manifest"],
            dtype={"clean_target": str, "attack_target": str},
        )
        if sha256_file(ROOT / contract["manifest"]) != contract["manifest_sha256"]:
            raise ValueError("collected manifest drift")
        expected = {"collected_receipts": 114}
    else:
        frame = pd.read_csv(
            ROOT / "data/processed/master_manifest.csv",
            dtype={"clean_target": str, "attack_target": str},
        )
        frame = frame.loc[
            frame.experiment_split.eq("validation") & frame.usable.eq(True)
        ]
        expected = {"cord_v2": 100, "sroie": 126, "resume_parsing_vision": 75}
        if args.model == "donut":
            frame = frame.loc[frame.dataset_name.ne("resume_parsing_vision")]
            expected.pop("resume_parsing_vision")
    if frame.groupby("dataset_name").size().to_dict() != expected:
        raise ValueError("full clean cohort counts differ")
    config["expected_clean_counts"] = expected
    baseline = {}
    provenance = []
    for path in args.baseline:
        path = path.resolve()
        table = pd.read_parquet(path)
        for _, row in table.iterrows():
            model_key = str(row["model_key"])
            pinned = base["models"].get(model_key)
            if (
                pinned is None
                or row.get("model_id") != pinned["model_id"]
                or row.get("model_revision") != pinned["revision"]
            ):
                raise ValueError(
                    "baseline model identity/revision missing or differs from pinned source"
                )
            key = (model_key, str(row["record_id"]))
            value = row["clean_correct"]
            if pd.isna(value) or value not in [True, False]:
                raise ValueError("invalid baseline outcome")
            if key in baseline and baseline[key] != bool(value):
                raise ValueError("contradictory baseline")
            baseline[key] = bool(value)
        provenance.append(
            dict(kind="baseline", path=str(path), sha256=sha256_file(path))
        )
    documents = {
        row.record_id: dict(
            record_id=row.record_id,
            dataset_name=row.dataset_name,
            experiment_split=row.experiment_split,
            clean_target=str(row.clean_target),
            frozen_attack_target=str(row.attack_target),
            attacks=[],
        )
        for row in frame.itertuples()
    }
    pending = []
    for source_spec in args.source:
        source_model, attack_type, raw_path = source_spec.split(":", 2)
        if source_model not in {"qwen", "donut"}:
            raise ValueError("source model must be Qwen or Donut")
        path = Path(raw_path).resolve()
        source = dict(
            name=f"{source_model}:{attack_type}:{sha256_file(path)}",
            records=str(path),
            sha256=sha256_file(path),
            artifact_root=str(ROOT),
        )
        raw = pd.read_parquet(path)
        if not raw.status.eq("complete").all():
            raise ValueError("source has incomplete evaluations")
        epsilons = (
            [2, 4, 8, 16]
            if args.mode == "transfer" and attack_type == "standard"
            else [4, 8, 16]
        )
        attacks = _load_attack_sources(
            ROOT, dict(attack_sources=[source], attack_epsilons_255=epsilons)
        )
        # Full clean-correct set from independently frozen baseline tables.
        datasets = set(attacks.dataset_name)
        eligible = {
            rid
            for rid, d in documents.items()
            if d["dataset_name"] in datasets
            and baseline.get((source_model, rid)) is True
        }
        applicable = {
            rid
            for rid, d in documents.items()
            if not (
                source_model == "donut" and d["dataset_name"] == "resume_parsing_vision"
            )
        }
        if any((source_model, rid) not in baseline for rid in applicable):
            raise ValueError(
                f"missing full {source_model} baseline; supply --baseline tables"
            )
        observed = set(zip(attacks.record_id, attacks.epsilon_255.astype(int)))
        required = {(rid, epsilon) for rid in eligible for epsilon in epsilons}
        if observed != required or len(attacks) != len(required):
            raise ValueError(
                f"source must contain each eligible document at {epsilons} exactly once"
            )
        for _, row in attacks.iterrows():
            if str(row.clean_target) != documents[row.record_id]["clean_target"]:
                raise ValueError("source ground truth differs from manifest")
            if (
                str(row.attack_target)
                != documents[row.record_id]["frozen_attack_target"]
            ):
                raise ValueError(
                    "source attack target differs from frozen manifest policy"
                )
            from final_protocol.reporting import source_outcome

            success = source_outcome(row)
            source_id = sha256_text(
                source["name"] + "|" + str(row.source_experiment_key)
            )
            attack = dict(
                source_id=source_id,
                source_model=source_model,
                attack_type=attack_type,
                epsilon_255=int(row.epsilon_255),
                attack_target=str(row.attack_target),
                source_attack_success=success,
            )
            documents[row.record_id]["attacks"].append(attack)
            pending.append((attack, row))
        provenance.append(dict(kind="attack", **source))
    # Across split source files each family must cover every applicable dataset.
    for source_model, attack_type in {
        (a["source_model"], a["attack_type"]) for a, _ in pending
    }:
        observed = {
            (d["record_id"], a["epsilon_255"])
            for d in documents.values()
            for a in d["attacks"]
            if a["source_model"] == source_model and a["attack_type"] == attack_type
        }
        required = {
            (rid, e)
            for rid in documents
            if baseline.get((source_model, rid)) is True
            for e in (
                [2, 4, 8, 16]
                if args.mode == "transfer" and attack_type == "standard"
                else [4, 8, 16]
            )
        }
        family_count = sum(
            1
            for d in documents.values()
            for a in d["attacks"]
            if a["source_model"] == source_model and a["attack_type"] == attack_type
        )
        if observed != required or family_count != len(required):
            raise ValueError(
                f"incomplete family {source_model}:{attack_type}; include every dataset source"
            )
    if args.mode == "adaptive" and not pending:
        raise ValueError("adaptive requires source eligible examples and seed images")
    if args.mode != "transfer":
        executable = shutil.which(args.tesseract)
        if executable is None or args.tessdata_dir is None:
            parser.error("provide Tesseract executable and --tessdata-dir")
        language = "eng+ind" if args.cohort == "public" else "eng"
        config["ocr"].update(
            executable=executable,
            executable_sha256=sha256_file(executable),
            version=subprocess.check_output(
                [executable, "--version"], text=True
            ).splitlines()[0],
            tessdata_dir=str(args.tessdata_dir.resolve()),
            language=language,
            traineddata_sha256={
                lang: sha256_file(args.tessdata_dir / (lang + ".traineddata"))
                for lang in language.split("+")
            },
        )
    # Export immutable copies only after validating the complete source plan.
    pages = _load_clean_pages(
        ROOT,
        {"collected_manifest_contract": contract},
        frame.set_index("record_id").experiment_split.to_dict(),
    )
    for rid, document in documents.items():
        document["images"] = save_pages(pages[rid], bundle / "clean" / sha256_text(rid))
    for attack, row in pending:
        attack["images"] = save_pages(
            _load_attacked_pages(row, ROOT), bundle / "attacks" / attack["source_id"]
        )
    plan = bundle / "plan.json"
    atomic_json(plan, list(documents.values()))
    config["plan"] = dict(path=str(plan.relative_to(ROOT)), sha256=sha256_file(plan))
    config["input_provenance"] = provenance
    atomic_json(destination, config)
    print(destination)
    print(
        "Frozen full clean cohort and source copies. Run CPU preflight and L40S smoke before production."
    )


if __name__ == "__main__":
    main()
