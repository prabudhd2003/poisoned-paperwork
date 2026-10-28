"""Pinned InternVL2.5 adapter used only for black-box transfer evaluation."""

from __future__ import annotations

import gc
import math
from typing import Sequence


DEFAULT_MODEL_ID = "OpenGVLab/InternVL2_5-4B"
DEFAULT_REVISION = "2cf4a8158bbc40d35015e7c63b527890de4d27b3"


def _dynamic_preprocess(image, image_size: int = 448, max_tiles: int = 12):
    """Official-style InternVL aspect-ratio tiling, frozen locally for auditability."""
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {(i, j) for n in range(1, max_tiles + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if 1 <= i * j <= max_tiles},
        key=lambda value: value[0] * value[1],
    )
    target = min(
        ratios,
        key=lambda value: (
            abs(aspect - value[0] / value[1]),
            -value[0] * value[1] if width * height > 0.5 * image_size**2 * value[0] * value[1] else 0,
        ),
    )
    resized = image.resize((image_size * target[0], image_size * target[1]), Image.Resampling.BICUBIC)
    tiles = []
    for index in range(target[0] * target[1]):
        left = (index % target[0]) * image_size
        top = (index // target[0]) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


class InternVL:
    """Deterministic, inference-only InternVL wrapper; never exposes gradients."""

    def __init__(
        self, *, device, model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION, precision: str = "bfloat16",
        max_new_tokens: int = 32, image_size: int = 448, max_tiles: int = 12,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.image_size = int(image_size)
        self.max_tiles = int(max_tiles)
        self.dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, trust_remote_code=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True,
        ).eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _pixels(self, pages: Sequence):
        import torch
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        convert = transforms.Compose([
            transforms.Lambda(lambda value: value.convert("RGB")),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), normalize,
        ])
        tensors, counts = [], []
        for page in pages:
            tiles = _dynamic_preprocess(page, self.image_size, self.max_tiles)
            counts.append(len(tiles))
            tensors.extend(convert(tile) for tile in tiles)
        return torch.stack(tensors).to(self.device, dtype=self.dtype), counts

    def predict(self, images, prompt: str) -> str:
        from PIL import Image

        pages = [images] if isinstance(images, Image.Image) else list(images)
        pixels, counts = self._pixels(pages)
        prefix = "\n".join(f"Image-{index + 1}: <image>" for index in range(len(pages)))
        question = f"{prefix}\n{prompt}" if prefix else prompt
        generation = {
            "max_new_tokens": self.max_new_tokens, "do_sample": False,
            "num_beams": 1,
        }
        return self.model.chat(
            self.tokenizer, pixels, question, generation,
            num_patches_list=counts, history=None, return_history=False,
        )

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
