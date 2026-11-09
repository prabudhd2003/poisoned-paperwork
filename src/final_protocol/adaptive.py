"""Actual-pipeline, query-based bounded search with atomic resumable state.

This is a square-style random search with discrete component feedback, not an
implementation of the published Square Attack loss or a robustness certificate.
"""

import io
import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image


def reload_pages(arrays):
    pages = []
    for array in arrays:
        stream = io.BytesIO()
        Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
        stream.seek(0)
        with Image.open(stream) as image:
            pages.append(image.convert("RGB").copy())
    return pages


def score(result, target, objective):
    if objective == "availability":
        return (int(result["abstained"]),)
    components = [
        result[k] for k in ("full_prediction", "crop_prediction", "ocr_prediction")
    ]
    return (
        int(result["prediction"] == target and not result["abstained"]),
        sum(value == target for value in components),
    )


def search(
    clean_pages, seeds, evaluate, target, epsilon, settings, checkpoint, contract
):
    """Every query redoes OCR/localization; labels are used only by this attacker."""
    clean = [np.asarray(page.convert("RGB"), dtype=np.int16) for page in clean_pages]
    checkpoint = Path(checkpoint)
    rng = np.random.default_rng(settings["seed"])

    def project(arrays):
        if len(arrays) != len(clean) or any(
            a.shape != c.shape for a, c in zip(arrays, clean)
        ):
            raise ValueError("attack page count or dimensions differ")
        return [
            np.minimum(np.maximum(a, c - epsilon), c + epsilon)
            .clip(0, 255)
            .astype(np.uint8)
            for a, c in zip(arrays, clean)
        ]

    if checkpoint.exists():
        # Checkpoints are local files created by this process, never imported sources.
        with checkpoint.open("rb") as stream:
            state = pickle.load(stream)
        if state["contract"] != contract:
            raise RuntimeError("adaptive checkpoint contract drift")
        rng.bit_generator.state = state["rng"]
    else:
        state = dict(
            contract=contract,
            queries=0,
            best=project(clean),
            best_score=(-1,),
            best_result=None,
            curve=[],
            seed_index=0,
            runtime_seconds=0.0,
            vlm_calls=0,
            ocr_pages=0,
        )
    proposals = [clean] + [
        [np.asarray(p, dtype=np.int16) for p in pages] for pages in seeds
    ]
    while state["queries"] < settings["queries"]:
        if state["seed_index"] < len(proposals):
            candidate = project(proposals[state["seed_index"]])
            state["seed_index"] += 1
        else:
            candidate = [a.copy() for a in state["best"]]
            # Restarts cover flat decision regions without relying on OCR gradients.
            if state["queries"] % settings["restart_interval"] == 0:
                candidate = project(
                    [c + rng.integers(-epsilon, epsilon + 1, c.shape) for c in clean]
                )
            index = int(rng.integers(len(candidate)))
            height, width = candidate[index].shape[:2]
            fraction = (
                settings["square_fraction"]
                * (1 - state["queries"] / settings["queries"])
                + 0.005
            )
            size = max(1, min(height, width, int(min(height, width) * fraction)))
            top, left = int(rng.integers(height - size + 1)), int(
                rng.integers(width - size + 1)
            )
            sign = rng.choice([-1, 1], size=(1, 1, 3))
            candidate[index][top : top + size, left : left + size] = np.clip(
                clean[index][top : top + size, left : left + size] + sign * epsilon,
                0,
                255,
            )
        result = evaluate(reload_pages(candidate))
        for cost in ("runtime_seconds", "vlm_calls", "ocr_pages"):
            state[cost] += result.get(cost, 0)
        value = score(result, target, settings["objective"])
        state["queries"] += 1
        if value >= tuple(state["best_score"]):
            state.update(best=candidate, best_score=value, best_result=result)
        if state["queries"] in settings["report_queries"]:
            state["curve"].append(
                dict(queries=state["queries"], score=list(state["best_score"]))
            )
        state["rng"] = rng.bit_generator.state
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, checkpoint)
        if state["best_score"][0] == 1:
            break
    return reload_pages(state["best"]), state
