#!/usr/bin/env python3
"""CPU-only parity check for Qwen's official and differentiable page preprocessing."""
from __future__ import annotations

import argparse
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
    x = F.interpolate(x, size=(gh * p, gw * p), mode="bicubic", align_corners=False,
                      antialias=True).clamp(0, 1)
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
        seen_sizes = {page.size for _, page in pages}
        for row in data:
            for page in row_images(row):
                page = page.convert("RGB")
                if page.size in seen_sizes:
                    continue
                pages.append((dataset_name, page))
                seen_sizes.add(page.size)
                accepted += 1
                if accepted >= limit:
                    break
            if accepted >= limit:
                break
    if sum(name in {"sroie", "cord_v2"} for name, _ in pages) < 10:
        raise RuntimeError("need at least ten receipt pages across SROIE and CORD")
    if sum(name == "resume_parsing_vision" for name, _ in pages) < 5 or len(pages) < 20:
        raise RuntimeError("need at least five resume pages and 20 varied total pages")
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="66285546d2b821cf421d4f5eb2576359d3770cd3")
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args()
    import torch
    from transformers import AutoProcessor
    from models.qwen import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, build_messages

    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision,
        min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS)
    pages = _sample_pages(ROOT, args.split)
    shapes = set()
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
        difference = (differentiable - reference).abs()
        ip = processor.image_processor
        p, temporal = int(ip.patch_size), int(ip.temporal_patch_size)
        scale = difference.new_tensor(ip.image_std).reshape(1, 3, 1, 1, 1) * 255
        error_255 = difference.reshape(-1, 3, temporal, p, p) * scale
        print(f"page={index:02d} dataset={dataset_name} input={page.width}x{page.height} "
              f"processor_shape={list(reference.shape)} "
              f"max_error_255={float(error_255.max()):.6f} "
              f"mean_error_255={float(error_255.mean()):.6f}")
    print(f"Checked {len(pages)} pages across {len(shapes)} processor tensor shapes on CPU.")


if __name__ == "__main__":
    main()
