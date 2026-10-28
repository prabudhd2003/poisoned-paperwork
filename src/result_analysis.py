"""CPU-only helpers used by result-analysis notebooks (never loads a VLM)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_table(path: str | Path):
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Run/merge the terminal stage first: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def parse_paths(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
    if isinstance(value, dict):
        value = [value]
    return [str(item.get("path")) if isinstance(item, dict) else str(item) for item in value]


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def show_comparison(clean_path, attacked_path, *, title="", amplification=16, dpi=180):
    """Full-resolution clean/attack/difference and localized heatmap viewer."""
    import matplotlib.pyplot as plt

    clean = np.asarray(load_rgb(clean_path), dtype=np.int16)
    attacked = np.asarray(load_rgb(attacked_path), dtype=np.int16)
    if clean.shape != attacked.shape:
        raise ValueError(f"shape mismatch: {clean.shape} vs {attacked.shape}")
    delta = attacked - clean
    magnitude = np.abs(delta).max(axis=2)
    figure, axes = plt.subplots(1, 4, figsize=(24, 8), dpi=dpi)
    axes[0].imshow(clean.astype(np.uint8)); axes[0].set_title("Original")
    axes[1].imshow(attacked.astype(np.uint8)); axes[1].set_title("Attacked")
    axes[2].imshow(np.clip(np.abs(delta) * amplification, 0, 255).astype(np.uint8))
    axes[2].set_title(f"Absolute RGB difference ×{amplification}")
    heat = axes[3].imshow(magnitude, cmap="magma", vmin=0, vmax=max(1, magnitude.max()))
    axes[3].set_title(f"Per-pixel max change (max={magnitude.max():.0f}/255)")
    figure.colorbar(heat, ax=axes[3], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def rate_table(frame, group_columns, outcome, *, count_name="trials"):
    grouped = frame.groupby(group_columns, dropna=False)[outcome]
    return grouped.agg(**{count_name: "size", "successes": "sum", "rate": "mean"}).reset_index()
