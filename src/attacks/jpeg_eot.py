"""Defense-aware full-image EOT/BPDA attack for Qwen document inputs.

The optimized artifact is always a bounded RGB PNG. Real Pillow JPEG, resize,
blur, and brightness transforms run in the forward pass; a straight-through
BPDA approximation supplies gradients. The target loss is averaged over a
frozen set of visual transformations and semantically equivalent prompts.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from attacks.donut import checkpoint_state, load_checkpoint_state, project_linf
from attacks.qwen import Objective, _perceptibility, exact_forward_bpda, save_reload
from checkpoints import sha256_bytes, sha256_text
from defenses import evaluate_defense, evaluate_semantic_consistency, transform_pages
from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil


def transform_bpda_tensor(image: torch.Tensor, transform_id: str) -> torch.Tensor:
    """Use exact defense pixels forward and identity gradients backward."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("defense BPDA expects an RGB tensor with shape [3,H,W]")
    official_page = transform_pages(
        [rgb_tensor_to_pil(image.detach())], str(transform_id)
    )[0]
    official = pil_to_rgb_tensor(official_page, image.device).to(image.dtype)
    return exact_forward_bpda(image, official)


def jpeg_bpda_tensor(image: torch.Tensor, quality: int) -> torch.Tensor:
    """Convenience wrapper for the exact-JPEG-forward contract test."""
    return transform_bpda_tensor(image, f"jpeg_q{int(quality)}")


def quality_schedule(
    qualities: Sequence[int], samples: int, *, seed: int, step: int
) -> tuple[int, ...]:
    """Deterministically sample JPEG qualities for an auditable test."""
    values = tuple(int(value) for value in qualities)
    if not values or any(not 1 <= value <= 100 for value in values):
        raise ValueError("JPEG qualities must be integers in [1,100]")
    if len(set(values)) != len(values):
        raise ValueError("JPEG qualities must be unique")
    if not 1 <= int(samples) <= len(values):
        raise ValueError("EOT samples must be between one and the quality count")
    digest = hashlib.sha256(f"{seed}|{step}|jpeg-eot".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(rng.sample(values, int(samples)))


def view_schedule(
    count: int,
    samples: int,
    *,
    seed: int,
    step: int,
    weights: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Deterministically sample unique adaptive views, optionally by hardness."""
    if count < 1 or not 1 <= int(samples) <= count:
        raise ValueError("adaptive view sample count is invalid")
    digest = hashlib.sha256(f"{seed}|{step}|defense-eot".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if weights is None:
        return tuple(rng.sample(range(count), int(samples)))
    probabilities = [float(value) for value in weights]
    if len(probabilities) != count or any(
        not np.isfinite(value) or value <= 0 for value in probabilities
    ):
        raise ValueError("hard-view weights must be finite and positive")
    available = list(range(count))
    selected = []
    for _ in range(int(samples)):
        total = sum(probabilities[index] for index in available)
        threshold = rng.random() * total
        cumulative = 0.0
        choice = available[-1]
        for index in available:
            cumulative += probabilities[index]
            if cumulative >= threshold:
                choice = index
                break
        selected.append(choice)
        available.remove(choice)
    return tuple(selected)


def hard_view_weights(
    losses: Sequence[float], *, temperature: float, floor: float
) -> list[float]:
    """Convert per-view target losses into stable worst-view sampling weights."""
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("hard-view losses must be a nonempty finite vector")
    if temperature <= 0 or floor < 0 or floor * len(values) >= 1:
        raise ValueError("invalid hard-view temperature or probability floor")
    logits = (values - values.max()) / float(temperature)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    probabilities = probabilities * (1 - floor * len(values)) + floor
    probabilities /= probabilities.sum()
    return probabilities.tolist()


def low_frequency_component(tensor: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return a smooth FFT low-pass component with the original tensor shape."""
    if tensor.ndim != 3 or not 0 < float(cutoff) <= 1:
        raise ValueError("low-frequency filtering expects [C,H,W] and cutoff in (0,1]")
    work = tensor.float()
    height, width = work.shape[-2:]
    fy = torch.fft.fftfreq(height, device=work.device).reshape(-1, 1)
    fx = torch.fft.rfftfreq(width, device=work.device).reshape(1, -1)
    radius = torch.sqrt(fy.square() + fx.square()) / (2**-0.5)
    mask = torch.exp(-((radius / float(cutoff)).clamp_min(0) ** 4))
    spectrum = torch.fft.rfft2(work, dim=(-2, -1))
    filtered = torch.fft.irfft2(
        spectrum * mask, s=(height, width), dim=(-2, -1)
    )
    return filtered.to(tensor.dtype)


def frequency_aware_update(
    image: torch.Tensor,
    original: torch.Tensor,
    gradient: torch.Tensor,
    *,
    alpha: float,
    epsilon_255: int,
    cutoff: float,
    gradient_mix: float,
    delta_mix: float,
) -> torch.Tensor:
    """Take one bounded update biased toward compression-stable frequencies."""
    if not 0 <= gradient_mix <= 1 or not 0 <= delta_mix <= 1:
        raise ValueError("frequency mixes must be in [0,1]")
    low_gradient = low_frequency_component(gradient, cutoff)
    direction = (1 - gradient_mix) * gradient + gradient_mix * low_gradient
    candidate = image - float(alpha) * direction.sign()
    delta = candidate - original
    low_delta = low_frequency_component(delta, cutoff)
    candidate = original + (1 - delta_mix) * delta + delta_mix * low_delta
    return project_linf(candidate, original, epsilon_255, quantize=True)


def _defense_seed(*parts: object) -> int:
    """Match the deterministic resize-vote seed used by ``defenses.py``."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def resolved_training_views(
    config: dict, record_id: str, page_count: int = 1
) -> list[dict]:
    """Add the exact per-document random resize views used by the defense."""
    views = [dict(view) for view in config["attack"]["training_views"]]
    if config["attack"].get("include_exact_resize_vote_views"):
        count = int(config["defense"]["resize_votes"])
        defense_seed = int(config["defense"].get("defense_seed", 566))
        for index in range(count):
            rng = random.Random(_defense_seed(record_id, defense_seed, index))
            transforms = [
                f"resize_{rng.uniform(0.70, 1.00):.17g}"
                for _ in range(page_count)
            ]
            views.append({
                "transform": transforms[0],
                "page_transforms": transforms,
                "prompt_index": 0,
                "source": f"exact_resize_vote_{index}",
            })
    return views


def _transformed_pages_bpda(
    xs: Sequence[torch.Tensor], view: dict
) -> list[torch.Tensor]:
    transforms = view.get("page_transforms")
    if transforms is None:
        transforms = [str(view["transform"])] * len(xs)
    if len(transforms) != len(xs):
        raise ValueError("adaptive view page transforms do not match page count")
    return [
        transform_bpda_tensor(image, str(transform_id))
        for image, transform_id in zip(xs, transforms)
    ]


def _eot_loss_and_gradients(
    objectives: Sequence[Objective],
    xs: Sequence[torch.Tensor],
    *,
    views: Sequence[dict],
    samples: int,
    seed: int,
    step: int,
    view_weights: Sequence[float] | None = None,
) -> tuple[float, float, list[torch.Tensor], tuple[int, ...]]:
    """Average defense-aware loss and page gradients without a joint graph."""
    selected = view_schedule(
        len(views), samples, seed=seed, step=step, weights=view_weights
    )
    gradients = [torch.zeros_like(image) for image in xs]
    losses: list[float] = []
    log_probabilities: list[float] = []

    for view_index in selected:
        view = views[view_index]
        objective = objectives[int(view["prompt_index"])]
        if len(xs) == 1:
            transformed = _transformed_pages_bpda(xs, view)
            loss, log_probability = objective.loss(transformed)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite defense-aware target loss")
            gradient, = torch.autograd.grad(loss, xs)
            gradients[0].add_(gradient.detach())
            losses.append(float(loss.detach()))
            log_probabilities.append(float(log_probability.detach()))
        else:
            # Rebuild each graph so multi-page documents do not retain the
            # model activations for all pages at once.
            for page_index in range(len(xs)):
                transformed = _transformed_pages_bpda(xs, view)
                loss, log_probability = objective.loss(
                    transformed, grad_page=page_index
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite defense-aware target loss")
                gradient, = torch.autograd.grad(loss, xs[page_index])
                gradients[page_index].add_(gradient.detach())
                if page_index == 0:
                    losses.append(float(loss.detach()))
                    log_probabilities.append(float(log_probability.detach()))

    gradients = [gradient / len(selected) for gradient in gradients]
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("nonfinite defense-aware image gradient")
    return (
        float(np.mean(losses)),
        float(np.mean(log_probabilities)),
        gradients,
        selected,
    )


@torch.no_grad()
def _all_view_statistics(
    objectives: Sequence[Objective], xs: Sequence[torch.Tensor], views: Sequence[dict]
) -> tuple[float, float, list[float]]:
    losses, log_probabilities = [], []
    for view in views:
        objective = objectives[int(view["prompt_index"])]
        transformed = _transformed_pages_bpda(xs, view)
        loss, log_probability = objective.loss(transformed)
        losses.append(float(loss))
        log_probabilities.append(float(log_probability))
    return float(np.mean(losses)), float(np.mean(log_probabilities)), losses


def _eot_loss_value(
    objectives: Sequence[Objective], xs: Sequence[torch.Tensor], views: Sequence[dict]
) -> tuple[float, float]:
    loss, log_probability, _ = _all_view_statistics(objectives, xs, views)
    return loss, log_probability


def _result_dict(result, target: str) -> dict[str, object]:
    return {
        "prediction_normalized": result.final_prediction,
        "target_success": result.final_prediction == target,
        "abstained": result.abstained,
        "agreement": result.agreement,
        "entropy": result.entropy,
        "view_ids": list(result.view_ids),
        "view_predictions": list(result.normalized_predictions),
    }


def evaluate_saved_png(
    wrapper,
    pages: Sequence[Image.Image],
    prompt: str,
    target: str,
    dataset_name: str,
    record_id: str,
    config: dict,
    normalize_fn: Callable[[str | None], str | None],
) -> tuple[str, dict[str, dict[str, object]]]:
    """Evaluate the saved PNG against every completed project defense."""
    raw_png = wrapper.predict(list(pages), prompt)
    results: dict[str, dict[str, object]] = {}
    for defense_id in ("jpeg", "resize_vote", "selective_multiview"):
        result, _ = evaluate_defense(
            pages=pages,
            defense_id=defense_id,
            record_id=record_id,
            predict=lambda value: wrapper.predict(value, prompt),
            normalize=normalize_fn,
            config=config["defense"],
        )
        results[defense_id] = _result_dict(result, target)

    task = "resume" if dataset_name == "resume_parsing_vision" else "receipt"
    semantic, _ = evaluate_semantic_consistency(
        pages=pages,
        record_id=record_id,
        predict=lambda value, semantic_prompt: wrapper.predict(value, semantic_prompt),
        normalize=normalize_fn,
        transforms=config["defense"]["semantic_transforms"],
        prompts=config["semantic_prompts"][task],
        minimum_votes=int(config["defense"]["semantic_minimum_votes"]),
    )
    results["semantic_consistency"] = _result_dict(semantic, target)
    return raw_png, results


def attack(
    wrapper,
    pages: Sequence[Image.Image],
    row: dict,
    config: dict,
    epsilon: int,
    shard: Path,
    key: str,
    provenance: dict,
    lpips_model,
    normalize_fn: Callable[[str | None], str | None],
) -> dict:
    """Optimize a bounded PNG against all four completed defenses."""
    started = time.perf_counter()
    device = wrapper.device
    clean = [pil_to_rgb_tensor(page.convert("RGB"), device) for page in pages]
    prompt = str(row["prompt_text"])
    target = str(row["attack_target"])
    clean_target = str(row["clean_target"])
    dataset_name = str(row["dataset_name"])
    record_id = str(row["record_id"])
    if normalize_fn(wrapper.predict(list(pages), prompt)) != clean_target:
        raise RuntimeError("clean prediction drifted before adaptive optimization")

    task = "resume" if dataset_name == "resume_parsing_vision" else "receipt"
    prompts = list(config["semantic_prompts"][task])
    settings = config["attack"]
    if hasattr(wrapper, "make_adaptive_objective"):
        objectives = [
            wrapper.make_adaptive_objective(
                list(pages),
                value,
                target,
                config["processor_tolerance"],
                loss_type=str(settings["loss_type"]),
            )
            for value in prompts
        ]
    else:
        objectives = [
            Objective(wrapper, pages, value, target, config["processor_tolerance"])
            for value in prompts
        ]
    reference_grid = objectives[0].grids.tolist()
    if any(objective.grids.tolist() != reference_grid for objective in objectives):
        raise RuntimeError("semantic prompt changed the frozen image grid")
    parity = objectives[0].parity(clean)
    views = resolved_training_views(config, record_id, len(pages))
    samples = int(settings["eot_samples_per_step"])
    attack_method = str(settings.get("method", "uniform_multiview_eot"))
    sampling_method = str(settings.get("view_sampling", "uniform"))
    frequency = settings.get("frequency_aware", {})
    frequency_enabled = bool(frequency.get("enabled", False))
    seed = int(config["seed"]) + int(key[:8], 16)

    contract = {
        **provenance,
        "record_id": record_id,
        "epsilon_255": int(epsilon),
        "page_sizes": [list(page.size) for page in pages],
        "source_image_hashes": [sha256_bytes(page.tobytes()) for page in pages],
        "target_token_ids": [objective.target_ids for objective in objectives],
        "chat_hashes": [objective.template_hash for objective in objectives],
        "attack": settings,
        "defense": config["defense"],
        "semantic_prompts": prompts,
    }
    contract_hash = sha256_text(json.dumps(contract, sort_keys=True))
    checkpoint = shard / "checkpoints" / key / "state.pt"
    xs = [image.clone().requires_grad_(True) for image in clean]
    initial_loss, _ = _eot_loss_value(objectives, clean, views)
    best_loss = initial_loss
    best_pages = [image.detach().cpu() for image in clean]
    best_step = 0
    best_defense_count = -1
    best_defense_loss = float("inf")
    best_defense_pages = None
    best_defense_step = 0
    view_weights = [1.0 / len(views)] * len(views)
    start_step = 0
    logs: list[dict] = []
    prior_seconds = 0.0
    checkpoint_resumed = checkpoint.is_file()

    if checkpoint.is_file():
        state = load_checkpoint_state(
            checkpoint,
            clean_sha256=contract_hash,
            config_sha256=provenance["config_sha256"],
            model_revision=wrapper.revision,
            device=device,
        )
        extra = state["extra_state"]
        if extra.get("contract") != contract:
            raise RuntimeError("adaptive checkpoint contract drift")
        xs = [value.to(device).requires_grad_(True) for value in extra["pages"]]
        best_loss = float(state["best_loss"])
        best_pages = extra["best_pages"]
        best_step = int(extra["best_step"])
        best_defense_count = int(extra.get("best_defense_count", -1))
        best_defense_loss = float(extra.get("best_defense_loss", float("inf")))
        best_defense_pages = extra.get("best_defense_pages")
        best_defense_step = int(extra.get("best_defense_step", 0))
        view_weights = [float(value) for value in extra.get("view_weights", view_weights)]
        start_step = int(state["completed_steps"])
        logs = list(extra["logs"])
        prior_seconds = float(extra.get("runtime_seconds", 0.0))

    torch.cuda.reset_peak_memory_stats()
    alpha = float(settings["step_size_255"][str(int(epsilon))]) / 255.0
    max_steps = int(settings["max_steps"])
    checkpoint_interval = int(settings["checkpoint_interval"])
    evaluation_interval = int(settings["evaluation_interval"])
    final_step = start_step
    already_complete = bool(logs and logs[-1].get("all_defenses_success"))
    step_range = () if already_complete else range(start_step + 1, max_steps + 1)
    for step in step_range:
        loss_value, mean_logp, gradients, sampled = _eot_loss_and_gradients(
            objectives,
            xs,
            views=views,
            samples=samples,
            seed=seed,
            step=step,
            view_weights=view_weights if sampling_method == "hard_view" else None,
        )
        norm = float(
            sum(gradient.float().square().sum() for gradient in gradients).sqrt()
        )
        if norm <= 0:
            raise RuntimeError("zero defense-aware image gradient")
        if loss_value < best_loss:
            best_loss = loss_value
            best_pages = [image.detach().cpu() for image in xs]
            best_step = step - 1
        with torch.no_grad():
            if frequency_enabled:
                xs = [
                    frequency_aware_update(
                        image,
                        original,
                        gradient,
                        alpha=alpha,
                        epsilon_255=epsilon,
                        cutoff=float(frequency["cutoff"]),
                        gradient_mix=float(frequency["gradient_mix"]),
                        delta_mix=float(frequency["delta_mix"]),
                    ).detach().requires_grad_(True)
                    for image, original, gradient in zip(xs, clean, gradients)
                ]
            else:
                xs = [
                    project_linf(
                        image - alpha * gradient.sign(),
                        original,
                        epsilon,
                        quantize=True,
                    ).detach().requires_grad_(True)
                    for image, original, gradient in zip(xs, clean, gradients)
                ]
        final_step = step

        if step % checkpoint_interval == 0 or step == max_steps:
            loss_now, logp_now, view_losses = _all_view_statistics(
                objectives, xs, views
            )
            if sampling_method == "hard_view":
                view_weights = hard_view_weights(
                    view_losses,
                    temperature=float(settings["hard_view_temperature"]),
                    floor=float(settings["hard_view_probability_floor"]),
                )
            if loss_now < best_loss:
                best_loss = loss_now
                best_pages = [image.detach().cpu() for image in xs]
                best_step = step
            entry = {
                "step": step,
                "target_loss": loss_now,
                "mean_target_log_probability": logp_now,
                "sampled_view_indices": list(sampled),
                "sampled_views": [views[index] for index in sampled],
                "view_losses": view_losses,
                "view_weights_next": view_weights,
                "gradient_norm": norm,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            }
            if step % evaluation_interval == 0 or step == max_steps:
                evaluated_pages, _ = save_reload(
                    xs, clean, shard / "images" / key, epsilon
                )
                _, evaluated = evaluate_saved_png(
                    wrapper,
                    evaluated_pages,
                    prompt,
                    target,
                    dataset_name,
                    record_id,
                    config,
                    normalize_fn,
                )
                success_count = sum(
                    bool(value["target_success"]) for value in evaluated.values()
                )
                entry["defense_results"] = evaluated
                entry["defenses_defeated"] = success_count
                entry["all_defenses_success"] = success_count == len(evaluated)
                if (
                    success_count > best_defense_count
                    or (
                        success_count == best_defense_count
                        and loss_now < best_defense_loss
                    )
                ):
                    best_defense_count = success_count
                    best_defense_loss = loss_now
                    best_defense_pages = [image.detach().cpu() for image in xs]
                    best_defense_step = step
            logs.append(entry)
            checkpoint_state(
                checkpoint,
                adversarial=xs[0],
                clean_sha256=contract_hash,
                completed_steps=step,
                best_loss=best_loss,
                optimizer=None,
                config_sha256=provenance["config_sha256"],
                model_revision=wrapper.revision,
                extra_state={
                    "contract": contract,
                    "pages": [image.detach().cpu() for image in xs],
                    "best_pages": best_pages,
                    "best_step": best_step,
                    "best_defense_count": best_defense_count,
                    "best_defense_loss": best_defense_loss,
                    "best_defense_pages": best_defense_pages,
                    "best_defense_step": best_defense_step,
                    "view_weights": view_weights,
                    "logs": logs,
                    "runtime_seconds": prior_seconds + time.perf_counter() - started,
                },
            )
            print(
                json.dumps({"record_id": record_id, "epsilon": epsilon, **entry}),
                flush=True,
            )
            if entry.get("all_defenses_success"):
                break

    if not logs:
        raise RuntimeError("adaptive attack wrote no checkpoint")
    if best_defense_count > 0 and best_defense_pages is not None:
        selected = [value.to(device) for value in best_defense_pages]
        selected_step = best_defense_step
        selected_reason = "most_exact_defenses_defeated"
    else:
        selected = [value.to(device) for value in best_pages]
        selected_step = best_step
        selected_reason = "lowest_frozen_multiview_eot_loss"
    reloaded, image_info = save_reload(
        selected, clean, shard / "images" / key, epsilon
    )
    raw_png, defense_results = evaluate_saved_png(
        wrapper,
        reloaded,
        prompt,
        target,
        dataset_name,
        record_id,
        config,
        normalize_fn,
    )
    for item, original, page in zip(image_info, clean, reloaded):
        lpips_value, ssim = _perceptibility(
            original, pil_to_rgb_tensor(page, device), lpips_model
        )
        item.update(lpips=lpips_value, ssim=ssim)
    final_loss, final_logp = _eot_loss_value(objectives, selected, views)
    png_prediction = normalize_fn(raw_png)
    defense_successes = {
        name: bool(value["target_success"])
        for name, value in defense_results.items()
    }

    return {
        **parity,
        "raw_response": raw_png,
        "prediction_normalized": png_prediction,
        "png_target_success": png_prediction == target,
        "saved_image_success": defense_successes["jpeg"],
        "primary_jpeg_success": defense_successes["jpeg"],
        "resize_vote_success": defense_successes["resize_vote"],
        "selective_multiview_success": defense_successes["selective_multiview"],
        "semantic_consistency_success": defense_successes["semantic_consistency"],
        "all_defenses_success": all(defense_successes.values()),
        "defense_success_fraction": float(np.mean(list(defense_successes.values()))),
        "defense_results": defense_results,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "mean_target_log_probability": final_logp,
        "selected_step": selected_step,
        "selected_reason": selected_reason,
        "completed_steps": final_step,
        "gradient_norm_final": logs[-1]["gradient_norm"],
        "optimization_trace": logs,
        "images": image_info,
        "linf_255": max(value["linf_255"] for value in image_info),
        "lpips": float(np.mean([value["lpips"] for value in image_info])),
        "ssim": float(np.mean([value["ssim"] for value in image_info])),
        "runtime_seconds": prior_seconds + time.perf_counter() - started,
        "peak_gpu_memory_mb": max(value["peak_gpu_memory_mb"] for value in logs),
        "checkpoint_resumed": checkpoint_resumed,
        "target_token_ids_hashes": [
            objective.target_token_ids_hash for objective in objectives
        ],
        "contract_hash": contract_hash,
        "training_views": views,
        "attack_method": attack_method,
        "view_sampling": sampling_method,
        "frequency_aware": frequency,
        "bpda_method": "real_pillow_defense_forward_identity_backward",
    }
