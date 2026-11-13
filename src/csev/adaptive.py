"""Checkpointed adaptive PGD + actual-verifier query search.

Localization is recomputed on current pixels at each actual query. Its discrete
coordinates are held fixed within gradient steps; actual-pipeline search attacks
that nonsmooth part. This is a bounded empirical attack, not a certificate.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
from checkpoints import sha256_file
from csev.defense import answer_objective, prediction
from csev.study import bounds


def checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def project(array, clean, epsilon):
    return np.clip(
        np.rint(array),
        np.maximum(0, clean.astype(float) - epsilon),
        np.minimum(255, clean.astype(float) + epsilon),
    ).astype("uint8")


def gradient_step(
    wrapper, image, clean, target, prompt, observation, threshold, epsilon, settings
):
    import torch
    from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil
    from csev.defense import mask

    x = pil_to_rgb_tensor(image, wrapper.device).detach().requires_grad_(True)
    box, control = observation.get("box"), observation.get("control")
    terms = [("full", None, 1.0)]
    if box:
        terms.append(("crop", box, 1.0))
    # Compute hinge activation without retaining model graphs.
    if box and control:
        with torch.no_grad():
            losses = []
            for b in (box, control):
                masked = mask(image, b, settings["mask_value"])
                obj = answer_objective(
                    wrapper, masked, prompt, target, settings["processor_tolerance"]
                )
                losses.append(
                    float(obj.loss([pil_to_rgb_tensor(masked, wrapper.device)])[0])
                )
        if losses[0] - losses[1] < threshold:
            weight = settings["adaptive"]["score_weight"]
            terms.extend([("mask", box, -weight), ("mask", control, weight)])
    grad = torch.zeros_like(x)
    total = 0.0
    for kind, b, weight in terms:
        if kind == "crop":
            l, t, r, bt = b
            view = x[:, t:bt, l:r]
        elif kind == "mask":
            l, t, r, bt = b
            keep = torch.ones_like(x)
            keep[:, t:bt, l:r] = 0
            view = x * keep + (1 - keep) * (settings["mask_value"] / 255)
        else:
            view = x
        obj = answer_objective(
            wrapper,
            rgb_tensor_to_pil(view),
            prompt,
            target,
            settings["processor_tolerance"],
        )
        obj.parity([view])
        loss, _ = obj.loss([view])
        (g,) = torch.autograd.grad(loss, x)
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise RuntimeError("nonfinite adaptive loss/gradient")
        grad += weight * g.detach()
        total += weight * float(loss.detach())
        del loss, g, obj, view
    if not bool(grad.abs().max() > 0):
        raise RuntimeError("zero adaptive gradient; cannot interpret as robustness")
    with torch.no_grad():
        updated = (
            x - (epsilon / 255) * settings["adaptive"]["step_fraction"] * grad.sign()
        )
    return project(np.asarray(rgb_tensor_to_pil(updated)), clean, epsilon), total


def search(
    clean_image,
    dataset,
    target,
    epsilon,
    restart,
    settings,
    threshold,
    verifier,
    step_fn,
    state_path,
    contract,
):
    """step_fn(image, observation)->(uint8 pixels, loss); injectable CPU testing."""
    clean = np.asarray(clean_image.convert("RGB"))
    rng = np.random.default_rng(int(contract[:8], 16) + restart)
    a = settings["adaptive"]
    refreshes = (a["steps"] + a["refresh_interval"] - 1) // a["refresh_interval"]
    if a["queries"] < refreshes + 2:
        raise ValueError(
            "query budget cannot cover gradient refreshes and final verification"
        )
    state = dict(
        contract=contract,
        step=0,
        queries=0,
        trace=[],
        runtime_seconds=0.0,
        current=project(
            clean.astype(float) + rng.integers(-epsilon, epsilon + 1, clean.shape),
            clean,
            epsilon,
        ),
        best=None,
        best_rank=None,
        observation=None,
        seen_rejection=False,
        done=False,
    )
    state_path = Path(state_path)
    if state_path.exists():
        with state_path.open("rb") as f:
            state = pickle.load(f)
        if state["contract"] != contract:
            raise ValueError("adaptive checkpoint contract changed")
        rng.bit_generator.state = state["rng"]

    def save():
        state["rng"] = rng.bit_generator.state
        checkpoint(state_path, state)

    def query(array, phase):
        im = Image.fromarray(array)
        obs = verifier(im, dataset)
        pred = prediction(obs, threshold)
        target_score = verifier.score(im, verifier.prompt, target)
        rank = (
            int(pred == target),
            int(obs["full_prediction"] == target),
            int(obs["crop_agrees"]),
            obs["gap"] if obs["gap"] is not None else -1e6,
            target_score,
        )
        state["queries"] += 1
        state["seen_rejection"] |= pred is None
        state["trace"].append(
            dict(
                phase=phase,
                step=state["step"],
                query=state["queries"],
                prediction=pred,
                full_prediction=obs["full_prediction"],
                gap=obs["gap"],
                target_score=target_score,
            )
        )
        if state["best_rank"] is None or rank > state["best_rank"]:
            state.update(best=array.copy(), best_rank=rank)
        return obs

    if state["done"]:
        return state
    start = time.perf_counter()
    if state["observation"] is None:
        state["observation"] = query(state["current"], "initial")
        save()
    while state["step"] < a["steps"]:
        array, loss = step_fn(Image.fromarray(state["current"]), state["observation"])
        state["current"] = project(array, clean, epsilon)
        state["step"] += 1
        if state["step"] % a["refresh_interval"] == 0 or state["step"] == a["steps"]:
            state["observation"] = query(state["current"], "gradient")
        state["last_surrogate_loss"] = loss
        state["runtime_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        save()
    # Reserve the last actual-verifier query for reloaded selected PNG evaluation.
    while state["queries"] < a["queries"] - 1:
        candidate = state["best"].copy()
        h, w = clean.shape[:2]
        size = max(
            1,
            round(min(h, w) * (0.15 if state["queries"] < a["queries"] // 2 else 0.05)),
        )
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        signs = rng.choice([-1, 1], size=(1, 1, 3))
        candidate[y : y + size, x : x + size] = project(
            clean[y : y + size, x : x + size].astype(float) + epsilon * signs,
            clean[y : y + size, x : x + size],
            epsilon,
        )
        query(candidate, "query_search")
        state["runtime_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        save()
    final_path = state_path.parent / "selected.png"
    Image.fromarray(state["best"]).save(final_path)
    with Image.open(final_path) as im:
        reloaded = im.convert("RGB").copy()
    linf = bounds([clean_image], [reloaded], epsilon)
    obs = verifier(reloaded, dataset)
    state["queries"] += 1
    if int(prediction(obs, threshold) == target) != state["best_rank"][0]:
        raise RuntimeError(
            "saved PNG target acceptance differs from candidate verification"
        )
    state.update(
        done=True,
        final_observation=obs,
        final_prediction=prediction(obs, threshold),
        images=[dict(path=str(final_path), sha256=sha256_file(final_path))],
        linf_255=linf,
    )
    state["runtime_seconds"] += time.perf_counter() - start
    save()
    return state
