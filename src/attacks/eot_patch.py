"""Localized Qwen patch attack with deterministic expectation over transforms."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from attacks.qwen import Objective
from checkpoints import sha256_file
from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil


CORNER_ORDER = ("bottom_right", "top_right", "bottom_left", "top_left")


def choose_patch_box(image: torch.Tensor, fraction: float = 0.125, inset: float = 0.02):
    """Choose the brightest corner, using the frozen tie-break order."""
    _, height, width = image.shape
    side = max(1, round(min(height, width) * fraction))
    x_inset, y_inset = round(width * inset), round(height * inset)
    boxes = {
        "bottom_right": (width - x_inset - side, height - y_inset - side),
        "top_right": (width - x_inset - side, y_inset),
        "bottom_left": (x_inset, height - y_inset - side),
        "top_left": (x_inset, y_inset),
    }
    scores = {}
    for name, (left, top) in boxes.items():
        crop = image[:, top:top + side, left:left + side]
        scores[name] = float(crop.mean())
    name = max(CORNER_ORDER, key=lambda value: (scores[value], -CORNER_ORDER.index(value)))
    left, top = boxes[name]
    return {"name": name, "left": left, "top": top, "side": side,
            "area_fraction": side * side / (width * height)}


def apply_patch(clean: torch.Tensor, patch: torch.Tensor, box: dict) -> torch.Tensor:
    mask = torch.zeros_like(clean)
    left, top, side = box["left"], box["top"], box["side"]
    mask[:, top:top + side, left:left + side] = 1
    canvas = torch.zeros_like(clean)
    canvas[:, top:top + side, left:left + side] = patch
    return clean * (1 - mask) + canvas * mask


def _eot(x: torch.Tensor, rng: random.Random, settings: dict) -> torch.Tensor:
    _, height, width = x.shape
    scale = rng.uniform(*settings["resize_scale"])
    small = F.interpolate(
        x.unsqueeze(0), size=(max(1, round(height * scale)), max(1, round(width * scale))),
        mode="bicubic", align_corners=False, antialias=True,
    )
    y = F.interpolate(small, size=(height, width), mode="bicubic",
                      align_corners=False, antialias=True).squeeze(0)
    # JPEG BPDA surrogate: exact quantized values forward, identity backward.
    quantized = torch.round(y * 255) / 255
    y = y + (quantized - y).detach()
    sigma = rng.uniform(*settings["blur_sigma"])
    if sigma > 0.05:
        kernel = 3 if sigma < 0.75 else 5
        y = F.avg_pool2d(y.unsqueeze(0), kernel, stride=1, padding=kernel // 2).squeeze(0)
    y = (y * rng.uniform(*settings["brightness"])).clamp(0, 1)
    shift = float(settings.get("translation_fraction", 0.02))
    dx = round(rng.uniform(-shift, shift) * width)
    dy = round(rng.uniform(-shift, shift) * height)
    padded = F.pad(
        y,
        (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)),
        mode="constant", value=1.0,
    )
    start_x, start_y = max(-dx, 0), max(-dy, 0)
    return padded[:, start_y:start_y + height, start_x:start_x + width].clamp(0, 1)


def optimize_patch(wrapper, page, prompt: str, target: str, config: dict,
                   output: Path, seed: int):
    """Optimize only patch pixels; every other source pixel remains exact."""
    clean = pil_to_rgb_tensor(page.convert("RGB"), wrapper.device)
    box = choose_patch_box(clean, float(config["patch_fraction"]), float(config["inset_fraction"]))
    side = box["side"]
    patch = clean[:, box["top"]:box["top"] + side, box["left"]:box["left"] + side].clone()
    patch = patch.detach().requires_grad_(True)
    objective = Objective(wrapper, [page.convert("RGB")], prompt, target, config["processor_tolerance"])
    rng = random.Random(seed)
    use_eot = bool(config["use_eot"])
    samples = int(config["eot_samples"] if use_eot else 1)
    trace, started = [], time.perf_counter()
    start_step = 0
    checkpoint = output.with_suffix(".pt")
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=wrapper.device, weights_only=False)
        if state["seed"] != seed or state["box"] != box:
            raise RuntimeError("patch checkpoint contract drift")
        patch = state["patch"].to(wrapper.device).requires_grad_(True)
        start_step = int(state["step"])
        trace = state["trace"]
        rng.setstate(state["rng_state"])
    for step in range(start_step + 1, int(config["max_steps"]) + 1):
        composed = apply_patch(clean, patch, box)
        views = [_eot(composed, rng, config["eot_distribution"]) if use_eot else composed
                 for _ in range(samples)]
        loss = torch.stack([objective.loss([view])[0] for view in views]).mean()
        gradient, = torch.autograd.grad(loss, patch)
        if not torch.isfinite(gradient).all() or float(gradient.norm()) == 0:
            raise RuntimeError("patch attack produced an invalid gradient")
        with torch.no_grad():
            patch = (patch - float(config["step_size_255"]) / 255 * gradient.sign()).clamp(0, 1)
            patch = (patch * 255).round().div(255).detach().requires_grad_(True)
        if step % int(config["checkpoint_interval"]) == 0 or step == int(config["max_steps"]):
            trace.append({"step": step, "eot_loss": float(loss.detach()),
                          "gradient_norm": float(gradient.norm())})
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_name(checkpoint.name + f".tmp.{os.getpid()}")
            torch.save({"patch": patch.detach().cpu(), "step": step, "trace": trace,
                        "rng_state": rng.getstate(), "seed": seed, "box": box}, temporary)
            os.replace(temporary, checkpoint)
    adversarial = apply_patch(clean, patch, box).detach()
    image_path = output.with_suffix(".png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_image = image_path.with_name(image_path.name + f".tmp.{os.getpid()}")
    rgb_tensor_to_pil(adversarial).save(temporary_image, format="PNG")
    os.replace(temporary_image, image_path)
    with Image.open(image_path) as source:
        saved = source.convert("RGB").copy()
    reloaded = pil_to_rgb_tensor(saved, wrapper.device)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, box["top"]:box["top"] + side, box["left"]:box["left"] + side] = False
    if not torch.equal((reloaded * 255).round()[mask], (clean * 255).round()[mask]):
        raise RuntimeError("pixels outside the patch changed")
    return {
        "page": saved, "image_path": str(image_path), "image_sha256": sha256_file(image_path),
        "box": box, "trace": trace, "runtime_seconds": time.perf_counter() - started,
        "outside_patch_exact": True,
        "bpda_contract": "official Qwen forward; differentiable EOT/JPEG BPDA backward",
    }
