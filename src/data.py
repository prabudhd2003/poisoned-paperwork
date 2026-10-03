"""Processed-data selection and loading for modeling stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DATASETS = ("sroie", "cord_v2", "resume_parsing_vision")
RECEIPT_DATASETS = {"sroie", "cord_v2"}
EXPECTED_COUNTS = {
    "validation": {"sroie": 126, "cord_v2": 100, "resume_parsing_vision": 75},
    "test": {"sroie": 361, "cord_v2": 100, "resume_parsing_vision": 75},
}
EXPECTED_TASK = {
    "sroie": "receipt_total",
    "cord_v2": "receipt_total",
    "resume_parsing_vision": "resume_highest_degree",
}


def select_baseline_manifest(manifest, config: dict[str, Any]):
    """Return the exact usable document set for one baseline run."""
    required = {
        "record_id", "dataset_name", "document_id", "source_split", "source_index",
        "experiment_split", "task", "clean_target", "attack_target", "usable",
        "attack_eligible",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"processed manifest is missing: {', '.join(missing)}")
    selected = manifest[
        manifest["dataset_name"].isin(config["datasets"])
        & manifest["experiment_split"].eq(config["split"])
        & manifest["usable"].eq(True)
    ].copy()
    if selected.empty:
        raise ValueError("the baseline selection is empty")
    if selected["record_id"].duplicated().any():
        raise ValueError("the selected manifest contains duplicate record IDs")
    for dataset_name, expected_task in EXPECTED_TASK.items():
        tasks = set(selected.loc[selected["dataset_name"].eq(dataset_name), "task"])
        if tasks != {expected_task}:
            raise ValueError(
                f"{dataset_name} must contain task {expected_task!r}; found {sorted(tasks)}"
            )
    smoke = config.get("smoke_per_dataset")
    if smoke is not None:
        smoke = int(smoke)
        if smoke < 1:
            raise ValueError("smoke_per_dataset must be positive")
        selected = (
            selected.sort_values("record_id")
            .groupby("dataset_name", group_keys=False)
            .head(smoke)
        )
        counts = selected.groupby("dataset_name").size().to_dict()
        expected = {dataset: smoke for dataset in DATASETS}
    else:
        counts = selected.groupby("dataset_name").size().to_dict()
        expected = EXPECTED_COUNTS[config["split"]]
    if counts != expected:
        raise ValueError(f"baseline dataset counts are {counts}; expected {expected}")
    return selected.sort_values("record_id").set_index("record_id", drop=False)


def load_processed_rows(
    repo_root: Path, split: str, record_ids: list[str] | set[str]
) -> dict[str, dict[str, Any]]:
    """Load only requested processed documents, preserving resume page order."""
    from datasets import load_from_disk

    wanted = set(record_ids)
    rows: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({value.split("/", 1)[0] for value in wanted}):
        dataset_path = repo_root / "data" / "processed" / dataset_name
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"processed dataset not found: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))[split]
        for row in dataset:
            record_id = f"{dataset_name}/{row['source_split']}/{row['document_id']}"
            if record_id in wanted:
                rows[record_id] = row
    missing = sorted(wanted - rows.keys())
    if missing:
        raise ValueError(f"processed datasets are missing {len(missing)} requested records")
    return rows


def row_images(row: dict[str, Any]) -> list[Any]:
    images = row.get("images")
    if images:
        return list(images)
    image = row.get("image")
    if image is None:
        raise ValueError("processed row contains neither image nor images")
    return [image]
