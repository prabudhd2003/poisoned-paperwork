"""Pinned Donut DocVQA adapter for clean inference and image attacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from attacks.donut import targeted_token_loss


DEFAULT_MODEL_ID = "naver-clova-ix/donut-base-finetuned-docvqa"
DEFAULT_REVISION = "b19d2e332684b0e2d35d9144ce34047767335cf8"
PROMPT_TEMPLATE = "<s_docvqa><s_question>{question}</s_question><s_answer>"


def pil_to_rgb_tensor(image: Image.Image, device: torch.device | None = None) -> torch.Tensor:
    """Convert a PIL image into a CHW RGB float tensor without torchvision."""
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device)


def rgb_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW RGB float tensor to an exactly quantized PIL image."""
    import numpy as np

    array = (
        image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(np.asarray(array), mode="RGB")


def differentiable_donut_preprocess(
    image: torch.Tensor,
    *,
    height: int,
    width: int,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
    align_long_axis: bool,
) -> torch.Tensor:
    """Differentiable Donut resize, center-pad, and normalization pipeline."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    x = image
    input_height, input_width = x.shape[-2:]
    if align_long_axis and (
        (width < height and input_width > input_height)
        or (width > height and input_width < input_height)
    ):
        x = torch.rot90(x, 3, dims=(-2, -1))
        input_height, input_width = x.shape[-2:]

    scale = min(float(height) / input_height, float(width) / input_width)
    resized_height = max(1, min(height, int(input_height * scale)))
    resized_width = max(1, min(width, int(input_width * scale)))
    x = F.interpolate(
        x.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    x = x.clamp(0, 1)

    delta_height = height - resized_height
    delta_width = width - resized_width
    left = delta_width // 2
    right = delta_width - left
    top = delta_height // 2
    bottom = delta_height - top
    x = F.pad(x, (left, right, top, bottom), value=0.0)

    mean = torch.tensor(image_mean, device=x.device, dtype=x.dtype)[:, None, None]
    std = torch.tensor(image_std, device=x.device, dtype=x.dtype)[:, None, None]
    return (x - mean) / std


@dataclass(frozen=True)
class DonutSettings:
    height: int
    width: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    align_long_axis: bool


class DonutDocVQA:
    """Small wrapper that keeps clean and adversarial Donut paths identical."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str = DEFAULT_REVISION,
        max_length: int = 64,
    ) -> None:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers is required to load Donut") from exc

        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.max_length = int(max_length)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        tokenizer = self.processor.tokenizer
        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        image_processor = self.processor.image_processor
        size = image_processor.size
        if isinstance(size, dict):
            height, width = int(size["height"]), int(size["width"])
        elif hasattr(size, "height") and hasattr(size, "width"):
            height, width = int(size.height), int(size.width)
        else:
            height, width = map(int, size)
        self.settings = DonutSettings(
            height=height,
            width=width,
            image_mean=tuple(float(v) for v in image_processor.image_mean),
            image_std=tuple(float(v) for v in image_processor.image_std),
            align_long_axis=bool(getattr(image_processor, "do_align_long_axis", False)),
        )

    def prompt(self, question: str) -> str:
        return PROMPT_TEMPLATE.format(question=question)

    def _token_ids(self, text: str) -> torch.Tensor:
        return self.processor.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

    def preprocess_for_attack(self, image: torch.Tensor) -> torch.Tensor:
        return differentiable_donut_preprocess(
            image,
            height=self.settings.height,
            width=self.settings.width,
            image_mean=self.settings.image_mean,
            image_std=self.settings.image_std,
            align_long_axis=self.settings.align_long_axis,
        ).unsqueeze(0)

    def attack_objective(
        self,
        image: torch.Tensor,
        *,
        question: str,
        target_text: str,
        loss_type: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        prompt_ids = self._token_ids(self.prompt(question))
        target_ids = self._token_ids(target_text + "</s_answer>")
        decoder_input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        outputs = self.model(
            pixel_values=self.preprocess_for_attack(image),
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )
        start = prompt_ids.shape[1] - 1
        positions = torch.arange(
            start, start + target_ids.shape[1], device=self.device, dtype=torch.long
        )
        loss, mean_log_probability = targeted_token_loss(
            outputs.logits,
            target_ids[0],
            positions,
            loss_type=loss_type,
        )
        return loss, {"mean_target_log_probability": float(mean_log_probability.detach())}

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image, question: str) -> str:
        prompt = self.prompt(question)
        inputs = self.processor(images=image.convert("RGB"), text=prompt, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        prompt_ids = inputs.get("input_ids")
        if prompt_ids is None:
            prompt_ids = self._token_ids(prompt)
        else:
            prompt_ids = prompt_ids.to(self.device)
        generated = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=self.max_length,
            do_sample=False,
        )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        if self.processor.tokenizer.pad_token:
            text = text.replace(self.processor.tokenizer.pad_token, "")
        if self.processor.tokenizer.eos_token:
            text = text.replace(self.processor.tokenizer.eos_token, "")
        if "<s_answer>" in text:
            text = text.split("<s_answer>", 1)[1]
        if "</s_answer>" in text:
            text = text.split("</s_answer>", 1)[0]
        return text.strip()
