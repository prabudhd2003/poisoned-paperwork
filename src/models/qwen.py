"""Pinned Qwen2.5-VL-3B adapter for deterministic clean inference.

Heavy imports (torch, transformers) happen inside methods so CPU-only unit tests
can import the module-level constants without a GPU stack.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28
DEFAULT_MAX_NEW_TOKENS = 32


def select_dtype(device):
    """BF16 when the GPU supports it, otherwise FP16 (per the baseline spec)."""
    import torch

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_messages(num_images: int, prompt: str) -> list[dict[str, Any]]:
    """One user turn: every page image in original order, then the question."""
    if num_images < 1:
        raise ValueError("at least one image is required")
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class QwenVL:
    """Batch-size-1 greedy Qwen2.5-VL wrapper (frozen weights, eval mode)."""

    def __init__(
        self,
        *,
        device,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("transformers with Qwen2.5-VL support is required") from exc

        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_new_tokens = int(max_new_tokens)
        self.dtype = select_dtype(device)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.last_input_info: dict[str, Any] = {}

    @property
    def settings(self) -> dict[str, Any]:
        """Processor/generation settings recorded in every output record."""
        return {
            "precision": str(self.dtype).replace("torch.", ""),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "batch_size": 1,
        }

    def predict(self, images: Any, prompt: str) -> str:
        """Return the full decoded response (not normalized, not stripped)."""
        import torch
        from PIL import Image

        if isinstance(images, Image.Image):
            images = [images]
        pages: Sequence[Image.Image] = [image.convert("RGB") for image in images]
        messages = build_messages(len(pages), prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=list(pages), padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_length = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        grid = inputs.get("image_grid_thw")
        self.last_input_info = {
            "page_count": len(pages),
            "image_sizes": [list(page.size) for page in pages],
            "input_ids_shape": list(inputs["input_ids"].shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "image_grid_thw": grid.tolist() if grid is not None else None,
        }
        return response

    def unload(self) -> None:
        """Free GPU memory before another large model is loaded."""
        import torch

        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
