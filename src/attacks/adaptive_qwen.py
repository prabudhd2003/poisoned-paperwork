"""Adaptive EOT-PGD against the full selective multi-view Qwen defense."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from attacks.donut import project_linf
from attacks.qwen import Objective, save_reload
from models.donut import pil_to_rgb_tensor


def _blur(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    size = radius * 2 + 1
    return F.avg_pool2d(x.unsqueeze(0), size, stride=1, padding=radius).squeeze(0)


def differentiable_views(x: torch.Tensor, config: dict) -> list[torch.Tensor]:
    """Backward approximations matching the real defense's semantic views."""
    views = []
    for transform_id in config.get("multiview_transforms", [
        "identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95",
    ]):
        if transform_id == "identity":
            y = x
        elif transform_id.startswith("jpeg_q"):
            # Exact real JPEG is used for final success. The backward pass uses
            # straight-through 8-bit quantization, a documented BPDA surrogate.
            quantized = torch.round(x * 255) / 255
            y = x + (quantized - x).detach()
        elif transform_id.startswith("resize_"):
            scale = float(transform_id.removeprefix("resize_"))
            h, w = x.shape[-2:]
            small = F.interpolate(
                x.unsqueeze(0), size=(max(1, round(h * scale)), max(1, round(w * scale))),
                mode="bicubic", align_corners=False, antialias=True,
            )
            y = F.interpolate(small, size=(h, w), mode="bicubic",
                              align_corners=False, antialias=True).squeeze(0)
        elif transform_id.startswith("blur_"):
            y = _blur(x, 1)
        elif transform_id.startswith("brightness_"):
            y = (x * float(transform_id.removeprefix("brightness_"))).clamp(0, 1)
        else:
            raise ValueError(f"no adaptive surrogate for {transform_id}")
        views.append(y.clamp(0, 1))
    return views


def adaptive_multiview_attack(
    wrapper, page, prompt: str, target: str, attack_config: dict,
    defense_config: dict, epsilon_255: int, output: Path,
):
    """Optimize expected target loss across every frozen defense view."""
    device = wrapper.device
    clean = pil_to_rgb_tensor(page.convert("RGB"), device)
    objective = Objective(
        wrapper, [page.convert("RGB")], prompt, target,
        attack_config["processor_tolerance"],
    )
    x = clean.clone().requires_grad_(True)
    alpha = float(attack_config["attack"]["step_size_255"][str(epsilon_255)]) / 255
    trace, started = [], time.perf_counter()
    for step in range(1, int(attack_config["attack"]["max_steps"]) + 1):
        losses = [objective.loss([view])[0] for view in differentiable_views(x, defense_config)]
        loss = torch.stack(losses).mean()
        gradient, = torch.autograd.grad(loss, x)
        if not torch.isfinite(gradient).all() or float(gradient.norm()) == 0:
            raise RuntimeError("adaptive defense attack produced an invalid gradient")
        with torch.no_grad():
            x = project_linf(
                x - alpha * gradient.sign(), clean, epsilon_255, quantize=True
            ).detach().requires_grad_(True)
        if step % int(attack_config["attack"].get("checkpoint_interval", 10)) == 0:
            trace.append({"step": step, "eot_loss": float(loss.detach()),
                          "gradient_norm": float(gradient.norm())})
    pages, image_info = save_reload([x], [clean], output, epsilon_255)
    return {
        "pages": pages, "images": image_info, "trace": trace,
        "runtime_seconds": time.perf_counter() - started,
        "linf_255": image_info[0]["linf_255"],
        "bpda": "real Pillow forward for final evaluation; straight-through/differentiable backward",
    }

