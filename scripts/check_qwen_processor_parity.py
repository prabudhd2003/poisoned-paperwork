#!/usr/bin/env python3
"""CPU-only parity check for Qwen's official-forward BPDA preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tensor_pixels(image, processor, grid):
    import torch
    import torch.nn.functional as F
    ip = processor.image_processor
    p, temporal, merge = int(ip.patch_size), int(ip.temporal_patch_size), int(ip.merge_size)
    gt, gh, gw = map(int, grid)
    raw = torch.from_numpy(__import__("numpy").array(image.convert("RGB")).copy())
    x = raw.permute(2, 0, 1).float().div(255).unsqueeze(0)
    if x.shape[-2:] != (gh * p, gw * p):
        x = F.interpolate(x, size=(gh * p, gw * p), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
    mean = x.new_tensor(ip.image_mean).reshape(1, 3, 1, 1)
    std = x.new_tensor(ip.image_std).reshape(1, 3, 1, 1)
    x = ((x - mean) / std).repeat(temporal, 1, 1, 1)
    x = x.reshape(gt, temporal, 3, gh // merge, merge, p, gw // merge, merge, p)
    return x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gt * gh * gw, 3 * temporal * p * p)


def _sample_pages(root: Path, split: str):
    from datasets import load_from_disk
    from data import row_images
    pages = []
    targets = (("sroie", 5), ("cord_v2", 5), ("resume_parsing_vision", 10))
    for dataset_name, limit in targets:
        data = load_from_disk(str(root / "data" / "processed" / dataset_name))[split]
        accepted = 0
        for row in data:
            for page in row_images(row):
                page = page.convert("RGB")
                pages.append((dataset_name, page))
                accepted += 1
                if accepted >= limit:
                    break
            if accepted >= limit:
                break
    observed = {
        dataset_name: sum(name == dataset_name for name, _ in pages)
        for dataset_name, _ in targets
    }
    expected = dict(targets)
    if observed != expected:
        raise RuntimeError(
            f"processor parity requires page counts {expected}; found {observed}"
        )
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor
    from attacks.qwen import exact_forward_bpda
    from models.qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, build_messages

    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tolerance = config.get("processor_tolerance") or {}
    max_tolerance = float(tolerance.get("max", -1))
    mean_tolerance = float(tolerance.get("mean", -1))
    surrogate_mean_tolerance = float(tolerance.get("surrogate_mean", -1))
    if min(max_tolerance, mean_tolerance, surrogate_mean_tolerance) < 0:
        raise SystemExit(
            "Config must define nonnegative processor_tolerance.max, .mean, and .surrogate_mean"
        )
    processor = AutoProcessor.from_pretrained(config["model_id"], revision=config["model_revision"],
        min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS)
    pages = _sample_pages(ROOT, args.split)
    shapes = set()
    maximum = surrogate_maximum = 0.0
    absolute_sum = surrogate_absolute_sum = 0.0
    element_count = 0
    for index, (dataset_name, page) in enumerate(pages):
        prompt = "Read the document."
        text = processor.apply_chat_template(build_messages(1, prompt), tokenize=False,
                                             add_generation_prompt=True)
        official = processor(text=[text], images=[page], padding=True, return_tensors="pt")
        grid = official["image_grid_thw"][0]
        differentiable = _tensor_pixels(page, processor, grid)
        reference = official["pixel_values"].float()
        if tuple(differentiable.shape) != tuple(reference.shape):
            raise SystemExit(f"shape mismatch page={index} dataset={dataset_name}: "
                             f"{tuple(differentiable.shape)} vs {tuple(reference.shape)}")
        shapes.add(tuple(differentiable.shape))
        exact = exact_forward_bpda(differentiable, reference)
        difference = (exact - reference).abs()
        surrogate_difference = (differentiable - reference).abs()
        maximum = max(maximum, float(difference.max()))
        surrogate_maximum = max(surrogate_maximum, float(surrogate_difference.max()))
        absolute_sum += float(difference.double().sum())
        surrogate_absolute_sum += float(surrogate_difference.double().sum())
        element_count += difference.numel()
        ip = processor.image_processor
        p, temporal = int(ip.patch_size), int(ip.temporal_patch_size)
        scale = difference.new_tensor(ip.image_std).reshape(1, 3, 1, 1, 1) * 255
        error_255 = surrogate_difference.reshape(-1, 3, temporal, p, p) * scale
        print(f"page={index:02d} dataset={dataset_name} input={page.width}x{page.height} "
              f"processor_shape={list(reference.shape)} "
              f"max_error_255={float(error_255.max()):.6f} "
              f"mean_error_255={float(error_255.mean()):.6f}")
    observed_mean = absolute_sum / element_count
    surrogate_observed_mean = surrogate_absolute_sum / element_count
    print(
        f"Checked {len(pages)} pages across {len(shapes)} processor tensor shapes on CPU; "
        f"exact-forward normalized max={maximum:.8f}, mean={observed_mean:.8f}; "
        f"surrogate normalized max={surrogate_maximum:.8f}, mean={surrogate_observed_mean:.8f}."
    )
    if (maximum > max_tolerance or observed_mean > mean_tolerance or
            surrogate_observed_mean > surrogate_mean_tolerance):
        raise SystemExit(
            "PROCESSOR PARITY FAILED: "
            f"exact-forward max/mean={maximum:.8f}/{observed_mean:.8f}, "
            f"allowed={max_tolerance:.8f}/{mean_tolerance:.8f}; "
            f"surrogate mean={surrogate_observed_mean:.8f}, "
            f"allowed={surrogate_mean_tolerance:.8f}"
        )
    print("PROCESSOR PARITY PASSED (exact official forward, differentiable BPDA backward)")


if __name__ == "__main__":
    main()
