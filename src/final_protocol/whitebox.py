"""Full-page plus localized-region PGD seed for actual-pipeline query search.

OCR localization is refreshed on current pixels. The crop locations are held
constant within a gradient step. OCR recognition has no surrogate gradient;
actual-pipeline query search follows this seed to attack that remaining gate.
"""

import os
import pickle
import time
from pathlib import Path


def region_boxes(pages, regions, padding):
    result = []
    for region in regions:
        page = pages[region.page]
        l, t, r, b = region.box
        pad = max(padding, b - t)
        result.append(
            (
                region.page,
                max(0, l - pad),
                max(0, t - pad),
                min(page.width, r + pad),
                min(page.height, b + pad),
            )
        )
    return result


def optimize(
    wrapper,
    pages,
    prompt,
    target,
    dataset,
    ocr,
    evidence,
    epsilon,
    settings,
    checkpoint,
    contract,
):
    import torch
    from attacks.qwen import Objective
    from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil
    from final_protocol.evidence import localize

    clean = [pil_to_rgb_tensor(p, wrapper.device) for p in pages]
    checkpoint = Path(checkpoint)
    state = {
        "step": 0,
        "arrays": None,
        "contract": contract,
        "losses": [],
        "runtime_seconds": 0.0,
        "gradient_model_passes": 0,
        "ocr_pages": 0,
    }
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        if state["contract"] != contract:
            raise RuntimeError("whitebox checkpoint drift")
    xs = [x.clone().detach().requires_grad_(True) for x in clean]
    if state["arrays"] is not None:
        xs = [
            torch.tensor(a, device=wrapper.device).requires_grad_(True)
            for a in state["arrays"]
        ]

    def objective(view):
        if hasattr(wrapper, "make_adaptive_objective"):
            return wrapper.make_adaptive_objective(
                view,
                prompt,
                target,
                settings["processor_tolerance"],
                loss_type="cross_entropy",
            )
        return Objective(wrapper, view, prompt, target, settings["processor_tolerance"])

    full = objective(pages)
    full.parity(clean)
    for step in range(state["step"], settings["steps"]):
        started = time.perf_counter()
        current = [rgb_tensor_to_pil(x) for x in xs]
        state["ocr_pages"] += len(current)
        _, regions, _ = localize(ocr(current), dataset, evidence["minimum_confidence"])
        boxes = region_boxes(current, regions, evidence["crop_padding_pixels"])
        # Donut can read one crop at a time; use each separately for both models.
        views = [(full, None)] + [
            (objective([current[i].crop((l, t, r, b))]), box)
            for box in boxes
            for i, l, t, r, b in [box]
        ]
        gradients = [torch.zeros_like(x) for x in xs]
        losses = []
        for obj, box in views:
            indices = range(len(xs)) if box is None else [box[0]]
            for index in indices:
                view = (
                    xs
                    if box is None
                    else [xs[box[0]][:, box[2] : box[4], box[1] : box[3]]]
                )
                loss, _ = obj.loss(view, grad_page=index if box is None else 0)
                (gradient,) = torch.autograd.grad(loss, xs[index])
                if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise RuntimeError("nonfinite evidence surrogate gradient")
                gradients[index] += gradient.detach() / len(views)
                losses.append(float(loss.detach()))
                state["gradient_model_passes"] += 1
        if not any(g.abs().max() > 0 for g in gradients):
            raise RuntimeError("zero surrogate gradients; do not report robustness")
        with torch.no_grad():
            for x, c, g in zip(xs, clean, gradients):
                value = x - (epsilon / 255) * settings["step_fraction"] * g.sign()
                value = torch.minimum(
                    torch.maximum(value, c - epsilon / 255), c + epsilon / 255
                ).clamp(0, 1)
                x.copy_(value.mul(255).round().div(255))
        state.update(step=step + 1, arrays=[x.detach().cpu().numpy() for x in xs])
        state["losses"].append(sum(losses) / len(losses))
        state["runtime_seconds"] += time.perf_counter() - started
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint)
    return [rgb_tensor_to_pil(x) for x in xs], state
