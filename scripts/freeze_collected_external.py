#!/usr/bin/env python3
"""Create the immutable 100-receipt external-test manifest and freeze config hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from checkpoints import sha256_file  # noqa: E402


CONFIGS = (
    "configs/baseline/collected_external_test.json",
    "configs/qwen_attack/collected_external_test.json",
    "configs/donut_attack/collected_external_test.json",
    "configs/eot_patch/active.json",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=ROOT / "collected receipts")
    parser.add_argument("--labels", type=Path, default=ROOT / "annotations/collected_receipt_labels.csv")
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--update-configs", action="store_true")
    return parser.parse_args()


def decimal_text(value: str) -> str:
    try:
        amount = Decimal(value.replace("$", "").replace(",", "").strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid receipt total: {value!r}") from exc
    if amount <= 0:
        raise ValueError("receipt total must be positive")
    return f"{amount.quantize(Decimal('0.01')):.2f}"


def main():
    import pandas as pd

    args = parse_args()
    labels = pd.read_csv(args.labels, dtype=str).fillna("")
    required = {"image_filename", "visible_total", "status"}
    if not required <= set(labels.columns):
        raise SystemExit(f"labels missing columns: {sorted(required - set(labels.columns))}")
    labels = labels.loc[labels.status.str.lower().eq("verified")].sort_values("image_filename")
    if len(labels) != args.expected or labels.image_filename.duplicated().any():
        raise SystemExit(f"need exactly {args.expected} unique verified labels; found {len(labels)}")
    output = ROOT / "data/processed/collected_receipts/images"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(labels.itertuples(index=False)):
        source = args.images / row.image_filename
        if source.suffix.lower() != ".png":
            raise SystemExit(f"only PNG inputs are allowed: {source}")
        if not source.is_file():
            raise SystemExit(f"missing input image: {source}")
        record_id = f"collected_receipts/external_test/c{index:05d}"
        destination = output / f"c{index:05d}.png"
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.save(destination, format="PNG", optimize=False)
        clean = decimal_text(row.visible_total)
        target = f"{(Decimal(clean) * Decimal(10)).quantize(Decimal('0.01')):.2f}"
        rows.append({
            "record_id": record_id, "dataset_name": "collected_receipts",
            "experiment_split": "external_test", "task": "receipt_total",
            "source_filename": row.image_filename,
            "source_sha256": sha256_file(source),
            "processed_path": str(destination.relative_to(ROOT)),
            "processed_sha256": sha256_file(destination),
            "clean_target": clean, "attack_target": target,
            "status": "verified", "usable": True, "attack_eligible": True,
            "num_pages": 1, "width": image.width, "height": image.height,
            "reviewer": getattr(row, "reviewer", ""), "notes": getattr(row, "notes", ""),
        })
    manifest = ROOT / "data/processed/collected_receipts/manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    digest = sha256_file(manifest)
    contract = {
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": digest, "expected_documents": args.expected,
    }
    (manifest.parent / "manifest_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))
    if args.update_configs:
        for relative in CONFIGS:
            path = ROOT / relative
            config = json.loads(path.read_text(encoding="utf-8"))
            target = config.get("collected_manifest_contract", config)
            target.update(contract)
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(f"updated {relative}")
        print("Review, commit, and push the processed images, manifest, labels, and configs before CARC submission.")


if __name__ == "__main__":
    main()
