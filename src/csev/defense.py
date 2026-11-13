"""Label-free spatial verifier and Qwen answer-token scoring.

Coordinates are model-generated, never read from annotations. Missing evidence
is a measured rejection; runtime/model exceptions are errors, not rejections.
"""

import json
import math
import time

from normalization import normalize_receipt_total


def parse_box(text, size, settings):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)["box"]
        if not isinstance(value, list) or len(value) != 4:
            return None
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1000
            for v in value
        ):
            return None
        w, h = size
        l, t, r, b = [round(v * d / 1000) for v, d in zip(value, (w, h, w, h))]
        if r <= l or b <= t:
            return None
        minimum = settings["minimum_box_pixels"]
        # Expand small valid boxes so the model processor can read the crop.
        if r - l < minimum:
            l = max(0, min(w - minimum, (l + r - minimum) // 2))
            r = min(w, l + minimum)
        if b - t < minimum:
            t = max(0, min(h - minimum, (t + b - minimum) // 2))
            b = min(h, t + minimum)
        if (r - l) * (b - t) / (w * h) > settings["maximum_box_area_fraction"]:
            return None
        return (l, t, r, b)
    except (ValueError, TypeError, KeyError):
        return None


def control_box(box, size):
    """Farthest nonoverlapping equally sized grid box; no label-dependent search."""
    w, h = size
    l, t, r, b = box
    bw, bh = r - l, b - t
    candidates = []
    for fy in (0, 0.25, 0.5, 0.75, 1):
        for fx in (0, 0.25, 0.5, 0.75, 1):
            x, y = round((w - bw) * fx), round((h - bh) * fy)
            if x + bw <= l or r <= x or y + bh <= t or b <= y:
                candidates.append((x, y, x + bw, y + bh))
    return max(
        candidates, key=lambda a: ((a[0] - l) ** 2 + (a[1] - t) ** 2, a), default=None
    )


def mask(image, box, value):
    result = image.copy()
    result.paste((value, value, value), box)
    return result


def infer(image, dataset, prompt, predict, score, settings):
    """Return reusable observations, without threshold or ground truth access."""
    start = time.perf_counter()
    norm = lambda value: normalize_receipt_total(value, dataset)
    full_raw = predict([image], prompt)
    full = norm(full_raw)
    result = dict(
        full_prediction=full,
        crop_prediction=None,
        box=None,
        control=None,
        gap=None,
        crop_agrees=False,
        reason="invalid_answer",
        vlm_calls=1,
    )
    if full is not None:
        location = predict([image], settings["localization_prompt"])
        result.update(localization_raw=location, vlm_calls=2)
        box = parse_box(location, image.size, settings)
        result.update(box=box, reason="invalid_region")
        if box:
            crop = norm(predict([image.crop(box)], prompt))
            result.update(
                crop_prediction=crop,
                crop_agrees=crop == full,
                vlm_calls=3,
                reason="crop_disagreement",
            )
            control = control_box(box, image.size)
            result["control"] = control
            if control and crop == full:
                s_b = score(mask(image, control, settings["mask_value"]), prompt, full)
                s_r = score(mask(image, box, settings["mask_value"]), prompt, full)
                if not math.isfinite(s_b) or not math.isfinite(s_r):
                    raise RuntimeError("nonfinite CSEV score")
                result.update(
                    gap=s_b - s_r,
                    score_control=s_b,
                    score_evidence=s_r,
                    vlm_calls=5,
                    reason="scored",
                )
            elif not control:
                result["reason"] = "missing_control"
    result["runtime_seconds"] = time.perf_counter() - start
    return result


def prediction(observation, threshold):
    gap = observation["gap"]
    return (
        observation["full_prediction"]
        if observation["crop_agrees"] and gap is not None and gap >= threshold
        else None
    )


def answer_objective(wrapper, image, prompt, answer, tolerance):
    """Use existing exact-forward BPDA but score answer tokens only (no EOS)."""
    import torch
    from attacks.qwen import Objective

    obj = Objective(wrapper, [image], prompt, answer, tolerance)
    ids = wrapper.processor.tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError("empty answer tokens")
    tail = torch.tensor([ids], device=wrapper.device, dtype=torch.long)
    prefix = obj.prefix["input_ids"]
    obj.ids = torch.cat([prefix, tail], dim=1)
    obj.positions = torch.arange(
        prefix.shape[1] - 1, obj.ids.shape[1] - 1, device=wrapper.device
    )
    obj.target_tensor = tail.flatten()
    return obj


class Verifier:
    def __init__(self, wrapper, settings, prompt):
        self.wrapper, self.settings, self.prompt = wrapper, settings, prompt

    def predict(self, pages, prompt):
        from attacks.qwen import clear_rope

        clear_rope(self.wrapper.model)
        original = self.wrapper.max_new_tokens
        try:
            if prompt == self.settings["localization_prompt"]:
                self.wrapper.max_new_tokens = self.settings["localization_tokens"]
            return self.wrapper.predict(pages, prompt)
        finally:
            self.wrapper.max_new_tokens = original

    def score(self, image, prompt, answer):
        import torch
        from models.donut import pil_to_rgb_tensor

        obj = answer_objective(
            self.wrapper, image, prompt, answer, self.settings["processor_tolerance"]
        )
        with torch.no_grad():
            return -float(obj.loss([pil_to_rgb_tensor(image, self.wrapper.device)])[0])

    def __call__(self, image, dataset):
        return infer(
            image, dataset, self.prompt, self.predict, self.score, self.settings
        )
