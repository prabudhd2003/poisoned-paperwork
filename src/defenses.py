"""Reproducible document defenses and the selective multi-view contribution."""

from __future__ import annotations

import hashlib
import io
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class DefenseResult:
    defense_id: str
    final_prediction: str | None
    raw_predictions: tuple[str, ...]
    normalized_predictions: tuple[str | None, ...]
    view_ids: tuple[str, ...]
    agreement: float
    entropy: float
    abstained: bool
    additional_vlm_passes: int


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("RGB").copy()


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    if not 0 < float(scale) <= 1:
        raise ValueError("resize scale must be in (0,1]")
    image = image.convert("RGB")
    reduced = (
        max(1, round(image.width * float(scale))),
        max(1, round(image.height * float(scale))),
    )
    return image.resize(reduced, Image.Resampling.BICUBIC).resize(
        image.size, Image.Resampling.BICUBIC
    )


def transform_pages(
    pages: Sequence[Image.Image], transform_id: str, *, seed: int = 566
) -> list[Image.Image]:
    """Apply an auditable real-Pillow defense transform to every page."""
    if not pages:
        raise ValueError("at least one page is required")
    result = []
    rng = random.Random(seed)
    for page in pages:
        page = page.convert("RGB")
        if transform_id == "identity":
            transformed = page.copy()
        elif transform_id.startswith("jpeg_q"):
            transformed = jpeg_roundtrip(page, int(transform_id.removeprefix("jpeg_q")))
        elif transform_id.startswith("resize_"):
            transformed = resize_roundtrip(page, float(transform_id.removeprefix("resize_")))
        elif transform_id.startswith("blur_"):
            transformed = page.filter(ImageFilter.GaussianBlur(
                radius=float(transform_id.removeprefix("blur_"))
            ))
        elif transform_id.startswith("brightness_"):
            transformed = ImageEnhance.Brightness(page).enhance(
                float(transform_id.removeprefix("brightness_"))
            )
        elif transform_id == "random_resize":
            transformed = resize_roundtrip(page, rng.uniform(0.70, 1.00))
        else:
            raise ValueError(f"unknown transform: {transform_id}")
        if transformed.mode != "RGB" or transformed.size != page.size:
            raise RuntimeError("defense transform changed mode or final dimensions")
        result.append(transformed)
    return result


def _consensus(values: Sequence[str | None], minimum_votes: int) -> tuple[str | None, float, float]:
    if not values:
        raise ValueError("consensus needs predictions")
    counts = Counter(values)
    winner, votes = sorted(
        counts.items(), key=lambda item: (-item[1], str(item[0]))
    )[0]
    probabilities = [count / len(values) for count in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
    normalized_entropy = entropy / math.log(max(2, len(values)))
    return (winner if winner is not None and votes >= minimum_votes else None,
            votes / len(values), normalized_entropy)


def evaluate_defense(
    *, pages: Sequence[Image.Image], defense_id: str, record_id: str,
    predict: Callable[[Sequence[Image.Image]], str],
    normalize: Callable[[str | None], str | None], config: dict,
) -> tuple[DefenseResult, dict[str, list[Image.Image]]]:
    """Evaluate one complete defense and return every view for image auditing."""
    if defense_id == "none":
        views = (("identity", transform_pages(pages, "identity")),)
        minimum_votes = 1
        abstain_on_disagreement = False
    elif defense_id == "jpeg":
        transform = f"jpeg_q{int(config['jpeg_quality'])}"
        views = ((transform, transform_pages(pages, transform)),)
        minimum_votes = 1
        abstain_on_disagreement = False
    elif defense_id == "resize_vote":
        votes = int(config.get("resize_votes", 5))
        views = tuple(
            (
                f"random_resize_{index}",
                transform_pages(
                    pages, "random_resize",
                    seed=_seed(record_id, config.get("defense_seed", 566), index),
                ),
            )
            for index in range(votes)
        )
        minimum_votes = votes // 2 + 1
        abstain_on_disagreement = False
    elif defense_id == "selective_multiview":
        transform_ids = tuple(config.get("multiview_transforms", [
            "identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95",
        ]))
        views = tuple(
            (transform_id, transform_pages(pages, transform_id, seed=_seed(record_id, transform_id)))
            for transform_id in transform_ids
        )
        minimum_votes = int(config.get("multiview_minimum_votes", 3))
        abstain_on_disagreement = True
    else:
        raise ValueError(f"unsupported defense: {defense_id}")

    raw = tuple(predict(view_pages) for _, view_pages in views)
    normalized = tuple(normalize(value) for value in raw)
    winner, agreement, entropy = _consensus(normalized, minimum_votes)
    abstained = winner is None and abstain_on_disagreement
    if winner is None and not abstain_on_disagreement:
        # Reproducible availability fallback: use the identity/base response.
        winner = normalized[0]
    result = DefenseResult(
        defense_id=defense_id, final_prediction=winner,
        raw_predictions=raw, normalized_predictions=normalized,
        view_ids=tuple(name for name, _ in views), agreement=agreement,
        entropy=entropy, abstained=abstained,
        additional_vlm_passes=max(0, len(views) - 1),
    )
    return result, {name: value for name, value in views}


def evaluate_semantic_consistency(
    *,
    pages: Sequence[Image.Image],
    record_id: str,
    predict: Callable[[Sequence[Image.Image], str], str],
    normalize: Callable[[str | None], str | None],
    transforms: Sequence[str],
    prompts: Sequence[str],
    minimum_votes: int,
) -> tuple[DefenseResult, dict[str, list[Image.Image]]]:
    """Joint visual/prompt ensemble with selective majority consensus.

    Each view pairs a deterministic image transformation with a semantically
    equivalent prompt.  The answer is released only when the configured
    majority agrees after task normalization; otherwise the defense abstains.
    """
    if len(transforms) != len(prompts) or len(transforms) < 3:
        raise ValueError("semantic consistency needs equally sized transform/prompt lists")
    if not 1 <= int(minimum_votes) <= len(transforms):
        raise ValueError("semantic minimum_votes is outside the ensemble size")
    views: dict[str, list[Image.Image]] = {}
    raw = []
    view_ids = []
    for index, (transform_id, prompt) in enumerate(zip(transforms, prompts)):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("semantic prompts must be nonempty strings")
        view_id = f"semantic_{index:02d}_{transform_id}"
        view_pages = transform_pages(
            pages, transform_id, seed=_seed(record_id, "semantic", index, transform_id)
        )
        views[view_id] = view_pages
        view_ids.append(view_id)
        raw.append(predict(view_pages, prompt))
    normalized = tuple(normalize(value) for value in raw)
    winner, agreement, entropy = _consensus(normalized, int(minimum_votes))
    return DefenseResult(
        defense_id="semantic_consistency",
        final_prediction=winner,
        raw_predictions=tuple(raw),
        normalized_predictions=normalized,
        view_ids=tuple(view_ids),
        agreement=agreement,
        entropy=entropy,
        abstained=winner is None,
        additional_vlm_passes=len(transforms) - 1,
    ), views

