#!/usr/bin/env python3
"""Validate Khalid's Donut runs and build a defense/adaptive source table."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import sha256_file, sha256_text  # noqa: E402


PRIMARY_EPSILONS = (2, 4, 8, 16)
EXPECTED_DOCUMENTS = {"cord_v2": 75, "sroie": 46}
EXPECTED_MODEL_ID = "naver-clova-ix/donut-base-finetuned-docvqa"
EXPECTED_REVISION = "b19d2e332684b0e2d35d9144ce34047767335cf8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run", action="append", type=Path, required=True,
        help="Completed Donut output run; repeat for the epsilon-split runs.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/imported/donut_public_validation/merged/attacks_for_defenses.parquet",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    imported_root = output.parent.parent
    imported_images = imported_root / "images"
    frames = []
    images_by_hash: dict[str, list[Path]] = {}
    for raw_run in args.source_run:
        run = raw_run.expanduser().resolve()
        table = run / "merged" / "attacks.parquet"
        config_path = run / "config.json"
        if not table.is_file() or not config_path.is_file():
            raise SystemExit(f"incomplete Donut source run: {run}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_id") != EXPECTED_MODEL_ID:
            raise SystemExit(f"unexpected Donut model: {run}")
        if config.get("model_revision") != EXPECTED_REVISION:
            raise SystemExit(f"unexpected Donut revision: {run}")
        attack = config.get("attack", {})
        if (
            attack.get("max_steps") != 100
            or attack.get("loss_type") != "paper_margin"
            or attack.get("optimizer") != "adam"
            or attack.get("quantize_each_step") is not True
        ):
            raise SystemExit(f"Donut attack contract drift: {run}")
        frame = pd.read_parquet(table)
        frame["source_run"] = run.name
        frame["source_table_sha256"] = sha256_file(table)
        frames.append(frame)
        for image in sorted((run / "shards").glob("worker_*/images/*.png")):
            digest = sha256_file(image)
            images_by_hash.setdefault(digest, []).append(image)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "record_id", "dataset_name", "experiment_split", "epsilon_255",
        "clean_target", "target_text", "question", "status", "model_id",
        "model_revision", "attacked_image_sha256", "targeted_success_reloaded",
        "seed",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise SystemExit(f"Donut sources are missing columns: {missing}")
    combined = combined.loc[
        combined.status.eq("complete")
        & combined.epsilon_255.astype(int).isin(PRIMARY_EPSILONS)
    ].copy()
    combined["epsilon_255"] = combined.epsilon_255.astype(int)
    if len(combined) != 121 * len(PRIMARY_EPSILONS):
        raise SystemExit(f"expected 484 primary Donut records; found {len(combined)}")
    counts = combined.groupby("dataset_name").record_id.nunique().to_dict()
    if counts != EXPECTED_DOCUMENTS:
        raise SystemExit(f"Donut document counts drifted: {counts}")
    per_epsilon = combined.groupby("epsilon_255").record_id.nunique().to_dict()
    if per_epsilon != {epsilon: 121 for epsilon in PRIMARY_EPSILONS}:
        raise SystemExit(f"Donut epsilon coverage drifted: {per_epsilon}")
    for record_id, group in combined.groupby("record_id"):
        for field in (
            "dataset_name", "experiment_split", "clean_target", "target_text",
            "question", "model_id", "model_revision",
        ):
            if group[field].nunique(dropna=False) != 1:
                raise SystemExit(f"Donut source disagreement for {record_id}: {field}")

    imported_images.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in combined.itertuples(index=False):
        candidates = images_by_hash.get(str(row.attacked_image_sha256), [])
        record_prefix = str(row.record_id).replace("/", "_") + "__"
        epsilon_marker = f"__eps-{int(row.epsilon_255)}__"
        matching = [
            path for path in candidates
            if path.name.startswith(record_prefix) and epsilon_marker in path.name
        ]
        if len(matching) != 1 or not matching[0].is_file():
            raise SystemExit(f"missing attacked PNG for {row.record_id} eps={row.epsilon_255}")
        source = matching[0]
        destination = imported_images / source.name
        if destination.exists():
            if sha256_file(destination) != str(row.attacked_image_sha256):
                raise SystemExit(f"conflicting imported PNG: {destination}")
        else:
            shutil.copy2(source, destination)
        if sha256_file(destination) != str(row.attacked_image_sha256):
            raise SystemExit(f"copied PNG hash mismatch: {destination}")
        try:
            stored = destination.relative_to(ROOT)
        except ValueError:
            stored = destination
        paths.append(str(stored))
    combined["image_path"] = paths
    combined["attack_target"] = combined.target_text.astype(str)
    combined["clean_target"] = combined.clean_target.astype(str)
    combined["prompt_text"] = combined.question.astype(str)
    combined["model_key"] = "donut"
    combined["saved_image_success"] = combined.targeted_success_reloaded.astype(bool)
    combined["experiment_key"] = combined.apply(
        lambda row: sha256_text(
            f"donut-import|{row.record_id}|{int(row.epsilon_255)}|{int(row.seed)}"
        ),
        axis=1,
    )
    combined = combined.sort_values(
        ["dataset_name", "record_id", "epsilon_255"]
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    csv_path = output.with_suffix(".csv")
    combined.to_csv(csv_path, index=False)
    summary = {
        "records": len(combined),
        "unique_documents": int(combined.record_id.nunique()),
        "epsilons_255": list(PRIMARY_EPSILONS),
        "documents_by_dataset": counts,
        "output_sha256": sha256_file(output),
        "copied_images": len(paths),
        "image_directory": str(imported_images),
        "epsilon_32_policy": "archived_stress_test_excluded_from_matched_primary_protocol",
    }
    output.with_name("import_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
