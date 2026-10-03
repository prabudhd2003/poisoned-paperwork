"""Targeted image optimization used by the Donut stage.

All image tensors in this module are RGB ``float32`` tensors in ``[0, 1]``.
Attack budgets and step sizes are configured in ordinary 0--255 pixel units
and converted exactly once at the optimization boundary.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    """Final state returned by :func:`targeted_image_attack`."""

    adversarial: torch.Tensor
    completed_steps: int
    best_loss: float
    final_loss: float
    exact_target: bool
    early_stop_reason: str | None


def project_linf(
    candidate: torch.Tensor,
    clean: torch.Tensor,
    epsilon_255: float,
    *,
    quantize: bool = False,
) -> torch.Tensor:
    """Project an RGB tensor into the valid L-infinity pixel-space ball."""
    if epsilon_255 < 0 or epsilon_255 > 255:
        raise ValueError("epsilon_255 must be between 0 and 255")
    if candidate.shape != clean.shape:
        raise ValueError("candidate and clean tensors must have identical shapes")

    epsilon = float(epsilon_255) / 255.0
    projected = torch.maximum(torch.minimum(candidate, clean + epsilon), clean - epsilon)
    projected = projected.clamp(0.0, 1.0)
    if quantize:
        projected = torch.round(projected * 255.0) / 255.0
        # Rounding is safe for integer-valued epsilon budgets, but project once
        # more so this helper remains correct for arbitrary configured budgets.
        projected = torch.maximum(torch.minimum(projected, clean + epsilon), clean - epsilon)
        projected = projected.clamp(0.0, 1.0)
    return projected


def targeted_token_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a teacher-forced targeted loss and mean target log probability.

    ``target_positions`` identifies the decoder-logit position that predicts
    each corresponding token in ``target_ids``. This makes prompt and padding
    exclusion explicit instead of relying on a model-specific labels shortcut.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1, sequence, vocabulary]")
    target_ids = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
    target_positions = target_positions.reshape(-1).to(device=logits.device, dtype=torch.long)
    if target_ids.numel() == 0 or target_ids.numel() != target_positions.numel():
        raise ValueError("target ids and positions must have the same nonzero length")
    if target_positions.min() < 0 or target_positions.max() >= logits.shape[1]:
        raise ValueError("target position falls outside the decoder sequence")

    selected = logits[0, target_positions, :]
    target_logits = selected.gather(1, target_ids[:, None]).squeeze(1)
    mean_log_probability = F.log_softmax(selected, dim=-1).gather(
        1, target_ids[:, None]
    ).mean()

    if loss_type == "cross_entropy":
        loss = F.cross_entropy(selected, target_ids)
    elif loss_type == "paper_margin":
        top_logits, top_ids = selected.max(dim=-1)
        loss = torch.where(
            top_ids.eq(target_ids),
            torch.zeros_like(target_logits),
            top_logits - target_logits,
        ).mean()
    else:
        raise ValueError("loss_type must be 'cross_entropy' or 'paper_margin'")
    return loss, mean_log_probability


def checkpoint_state(
    path: Path,
    *,
    adversarial: torch.Tensor,
    clean_sha256: str,
    completed_steps: int,
    best_loss: float,
    optimizer: torch.optim.Optimizer | None,
    config_sha256: str,
    model_revision: str,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save a resumable, self-validating attack checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "adversarial": adversarial.detach().cpu(),
        "clean_sha256": clean_sha256,
        "completed_steps": int(completed_steps),
        "best_loss": float(best_loss),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config_sha256": config_sha256,
        "model_revision": model_revision,
        "extra_state": extra_state or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint_state(
    path: Path,
    *,
    clean_sha256: str,
    config_sha256: str,
    model_revision: str,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint only when its immutable provenance still matches."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only`` was added.
        state = torch.load(path, map_location="cpu")
    expected = {
        "clean_sha256": clean_sha256,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match the current run")
    state["adversarial"] = state["adversarial"].to(device=device, dtype=torch.float32)
    return state


def targeted_image_attack(
    clean: torch.Tensor,
    objective: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, Any]]],
    *,
    epsilon_255: float,
    step_size_255: float,
    max_steps: int,
    optimizer_name: str = "adam",
    quantize_each_step: bool = True,
    start_adversarial: torch.Tensor | None = None,
    start_step: int = 0,
    start_best_loss: float = float("inf"),
    optimizer_state: dict[str, Any] | None = None,
    checkpoint_interval: int = 10,
    checkpoint_callback: Callable[
        [int, torch.Tensor, float, float, dict[str, Any], torch.optim.Optimizer | None], None
    ]
    | None = None,
    early_stop: bool = False,
) -> AttackResult:
    """Minimize a targeted objective under an RGB-space L-infinity bound.

    The objective returns ``(loss, metadata)``. If metadata contains an
    ``exact_target`` boolean, it is used for logging and optional early stop.
    """
    if clean.ndim != 3 or clean.shape[0] != 3:
        raise ValueError("clean must be an RGB tensor with shape [3, H, W]")
    if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
        raise ValueError("clean image must contain finite values in [0, 1]")
    if step_size_255 <= 0:
        raise ValueError("step_size_255 must be positive")
    if max_steps < 0 or not 0 <= start_step <= max_steps:
        raise ValueError("invalid optimization step count")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")

    clean = clean.detach()
    initial = clean if start_adversarial is None else start_adversarial
    adversarial = project_linf(initial.detach(), clean, epsilon_255, quantize=quantize_each_step)
    adversarial = adversarial.clone().requires_grad_(True)

    optimizer: torch.optim.Optimizer | None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([adversarial], lr=float(step_size_255) / 255.0)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
    elif optimizer_name == "sign":
        optimizer = None
        if optimizer_state:
            raise ValueError("sign PGD has no optimizer state to restore")
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sign'")

    best_loss = float(start_best_loss)
    final_loss = float("nan")
    exact_target = False
    completed_steps = start_step
    early_stop_reason: str | None = None

    for step in range(start_step + 1, max_steps + 1):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        elif adversarial.grad is not None:
            adversarial.grad = None

        loss, metadata = objective(adversarial)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite scalar attack loss at step {step}")
        loss.backward()
        if adversarial.grad is None or not torch.isfinite(adversarial.grad).all():
            raise RuntimeError(f"missing or non-finite image gradient at step {step}")

        with torch.no_grad():
            if optimizer is None:
                adversarial.add_(
                    adversarial.grad.sign(), alpha=-float(step_size_255) / 255.0
                )
            else:
                optimizer.step()
            adversarial.copy_(
                project_linf(
                    adversarial,
                    clean,
                    epsilon_255,
                    quantize=quantize_each_step,
                )
            )

        final_loss = float(loss.detach().item())
        best_loss = min(best_loss, final_loss)
        exact_target = bool(metadata.get("exact_target", False))
        completed_steps = step

        if checkpoint_callback and (
            step % checkpoint_interval == 0 or step == max_steps or (early_stop and exact_target)
        ):
            checkpoint_callback(step, adversarial, best_loss, final_loss, metadata, optimizer)
        if early_stop and exact_target:
            early_stop_reason = "exact_target_in_memory"
            break

    return AttackResult(
        adversarial=adversarial.detach(),
        completed_steps=completed_steps,
        best_loss=best_loss,
        final_loss=final_loss,
        exact_target=exact_target,
        early_stop_reason=early_stop_reason,
    )
