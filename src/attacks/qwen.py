"""Shared differentiable targeted attack core for Qwen2.5-VL."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from attacks.donut import checkpoint_state, load_checkpoint_state, project_linf, targeted_token_loss
from checkpoints import sha256_bytes, sha256_file, sha256_text
from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil
from models.qwen import build_messages


def clear_rope(model) -> None:
    """Clear cached rotary offsets across the model before each multimodal forward."""
    for item in (model, *model.modules()):
        if hasattr(item, "rope_deltas"):
            item.rope_deltas = None


class Objective:
    """Frozen chat prefix, exact target token sequence, and differentiable page inputs."""
    def __init__(self, wrapper, pages: Sequence[Image.Image], prompt: str, target: str,
                 tolerances: dict[str, float]):
        self.wrapper, self.ip = wrapper, wrapper.processor.image_processor
        ip = self.ip
        if not (ip.do_resize and ip.do_rescale and ip.do_normalize):
            raise ValueError("Unsupported Qwen image processor flags")
        if abs(float(ip.rescale_factor) - 1 / 255) > 1e-10 or int(ip.resample) != int(Image.Resampling.BICUBIC):
            raise ValueError("Expected Qwen RGB rescale 1/255 and bicubic interpolation")
        self.chat = wrapper.processor.apply_chat_template(build_messages(len(pages), prompt),
            tokenize=False, add_generation_prompt=True)
        self.prefix = wrapper.processor(text=[self.chat], images=list(pages), padding=True,
                                        return_tensors="pt").to(wrapper.device)
        self.grids = self.prefix["image_grid_thw"]
        self.target_ids = wrapper.processor.tokenizer.encode(target + "<|im_end|>", add_special_tokens=False)
        if not self.target_ids:
            raise ValueError("target answer tokenized to an empty sequence")
        tail = torch.tensor([self.target_ids], device=wrapper.device, dtype=torch.long)
        self.ids = torch.cat([self.prefix["input_ids"], tail], dim=1)
        prefix_length = self.prefix["input_ids"].shape[1]
        self.positions = torch.arange(prefix_length - 1, self.ids.shape[1] - 1, device=wrapper.device)
        self.target_tensor = tail.flatten()
        self.tolerances = tolerances
        self.template_hash = sha256_text(self.chat)
        self.target_token_ids_hash = sha256_text(json.dumps(self.target_ids))
        self.page_grid_slices: list[tuple[int, int]] = []
        offset = 0
        for t, h, w in self.grids.tolist():
            count = int(t * h * w)
            self.page_grid_slices.append((offset, offset + count))
            offset += count

    def pixels(self, xs: Sequence[torch.Tensor], grad_page: int | None = None) -> torch.Tensor:
        ip = self.ip
        p, temporal, merge = int(ip.patch_size), int(ip.temporal_patch_size), int(ip.merge_size)
        if len(xs) != len(self.grids):
            raise ValueError("page/grid count mismatch")
        result = []
        for index, (x, grid) in enumerate(zip(xs, self.grids.tolist())):
            gt, gh, gw = map(int, grid)
            if gt != 1 or gh % merge or gw % merge:
                raise ValueError("unexpected still-image grid")
            y = x.unsqueeze(0)
            if y.shape[-2:] != (gh * p, gw * p):
                y = F.interpolate(y, size=(gh * p, gw * p), mode="bicubic",
                                  align_corners=False, antialias=True).clamp(0, 1)
            mean = x.new_tensor(ip.image_mean).reshape(1, 3, 1, 1)
            std = x.new_tensor(ip.image_std).reshape(1, 3, 1, 1)
            y = ((y - mean) / std).repeat(temporal, 1, 1, 1)
            y = y.reshape(gt, temporal, 3, gh // merge, merge, p, gw // merge, merge, p)
            y = y.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gt * gh * gw, 3 * temporal * p * p)
            if grad_page is not None and index != grad_page:
                y = y.detach()
            result.append(y)
        return torch.cat(result)

    def parity(self, xs: Sequence[torch.Tensor]) -> dict:
        diff, official = self.pixels(xs), self.prefix["pixel_values"].float()
        if diff.shape != official.shape:
            raise RuntimeError(f"processor tensor shape differs: {tuple(diff.shape)} vs {tuple(official.shape)}")
        error = (diff - official).abs()
        p, temporal = int(self.ip.patch_size), int(self.ip.temporal_patch_size)
        scale = error.new_tensor(self.ip.image_std).reshape(1, 3, 1, 1, 1) * 255
        error_255 = error.reshape(-1, 3, temporal, p, p) * scale
        stats = {"official_processor_shape": list(official.shape),
                 "differentiable_processor_shape": list(diff.shape),
                 "processor_max_abs_difference": float(error.max()),
                 "processor_mean_abs_difference": float(error.mean()),
                 "processor_max_abs_difference_255": float(error_255.max()),
                 "processor_mean_abs_difference_255": float(error_255.mean())}
        if (stats["processor_max_abs_difference"] > self.tolerances["max"] or
                stats["processor_mean_abs_difference"] > self.tolerances["mean"]):
            raise RuntimeError(f"Qwen preprocessing parity failed: {stats}")
        return stats

    def loss(self, xs: Sequence[torch.Tensor], grad_page: int | None = None):
        clear_rope(self.wrapper.model)
        output = self.wrapper.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids),
            pixel_values=self.pixels(xs, grad_page), image_grid_thw=self.grids, use_cache=False)
        return targeted_token_loss(output.logits, self.target_tensor, self.positions,
                                   loss_type="cross_entropy")

    @torch.inference_mode()
    def decode(self, xs: Sequence[torch.Tensor]) -> str:
        clear_rope(self.wrapper.model)
        batch = dict(self.prefix)
        batch["pixel_values"] = self.pixels(xs)
        generated = self.wrapper.model.generate(**batch, max_new_tokens=self.wrapper.max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None)
        return self.wrapper.processor.batch_decode(generated[:, batch["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def save_reload(xs, clean, folder: Path, epsilon: float):
    folder.mkdir(parents=True, exist_ok=True)
    pages, metadata = [], []
    for index, (x, original) in enumerate(zip(xs, clean)):
        path = folder / f"page_{index:03d}.png"
        bounded = project_linf(x.detach(), original, epsilon, quantize=True)
        rgb_tensor_to_pil(bounded).save(path, format="PNG")
        with Image.open(path) as source:
            page = source.convert("RGB").copy()
        if page.size != (original.shape[2], original.shape[1]):
            raise RuntimeError("saved image dimensions changed")
        loaded = pil_to_rgb_tensor(page, original.device)
        linf = float((loaded - original).abs().max() * 255)
        if linf > epsilon + 1e-4:
            raise RuntimeError("saved image exceeds its pixel-space epsilon bound")
        pages.append(page)
        metadata.append({"page": index, "path": str(path), "sha256": sha256_file(path),
                         "linf_255": linf, "width": page.width, "height": page.height})
    return pages, metadata


def attack(wrapper, pages, row, config, epsilon, shard: Path, key: str, provenance: dict,
           lpips_model, normalize_fn: Callable[[str | None], str | None]):
    """Joint sign-PGD with resumable full-contract checkpoints and official PNG verification."""
    started = time.perf_counter()
    device = wrapper.device
    clean = [pil_to_rgb_tensor(page.convert("RGB"), device) for page in pages]
    prompt, target = row["prompt_text"], str(row["attack_target"])
    clear_rope(wrapper.model)
    raw_clean = wrapper.predict(pages, prompt)
    if raw_clean != row["raw_response"] or normalize_fn(raw_clean) != row["prediction_normalized"]:
        raise RuntimeError("clean baseline prediction drift; stopping before attack")
    objective = Objective(wrapper, pages, prompt, target, config["processor_tolerance"])
    parity_stats = objective.parity(clean)
    if normalize_fn(objective.decode(clean)) != normalize_fn(raw_clean):
        raise RuntimeError("surrogate clean decoding differs from official baseline inference")
    contract = {**provenance, "record_id": row["record_id"], "epsilon": epsilon,
        "grid": objective.grids.tolist(), "chat_hash": objective.template_hash,
        "target_token_ids": objective.target_ids, "page_sizes": [list(p.size) for p in pages],
        "image_hashes": [sha256_bytes(p.tobytes()) for p in pages],
        "multi_page_fallback": config["attack"].get("multi_page_fallback", "gradient_accumulation_across_pages")}
    contract_hash = sha256_text(json.dumps(contract, sort_keys=True))
    checkpoint = shard / "checkpoints" / key / "state.pt"
    xs = [image.clone().requires_grad_(True) for image in clean]
    best_pages = [image.detach().cpu() for image in clean]
    with torch.no_grad():
        initial_loss = float(objective.loss(clean)[0])
    best_loss, start_step, logs, prior_seconds = initial_loss, 0, [], 0.0
    checkpoint_resumed = checkpoint.is_file()
    if checkpoint.is_file():
        state = load_checkpoint_state(checkpoint, clean_sha256=contract_hash,
            config_sha256=provenance["config_sha256"], model_revision=wrapper.revision, device=device)
        extra = state["extra_state"]
        if extra.get("contract") != contract:
            raise RuntimeError("checkpoint contract drift")
        xs = [image.to(device).requires_grad_(True) for image in extra["pages"]]
        best_pages, best_loss = extra["best_pages"], float(state["best_loss"])
        start_step, logs = int(state["completed_steps"]), extra["logs"]
        prior_seconds = float(extra.get("runtime_seconds", 0.0))
        import random
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(extra["cuda_rng_state"])

    torch.cuda.reset_peak_memory_stats()
    settings = config["attack"]
    alpha = float(settings["step_size_255"][str(int(epsilon))]) / 255
    final_step = start_step
    successful_checkpoint = bool(logs and logs[-1]["saved_image_success"])
    if not successful_checkpoint:
        for step in range(start_step + 1, int(settings["max_steps"]) + 1):
            # Accumulate exact per-page input gradients without retaining a joint
            # multi-page activation graph; selected once in run config for all docs.
            gradients, losses, log_prob = [], [], None
            use_accumulation = len(xs) > 1 and settings.get("multi_page_fallback") == "gradient_accumulation_across_pages"
            if use_accumulation:
                for page_index in range(len(xs)):
                    loss, log_prob = objective.loss(xs, grad_page=page_index)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite target loss")
                    gradient, = torch.autograd.grad(loss, xs[page_index])
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError("nonfinite image gradient")
                    gradients.append(gradient.detach())
                    losses.append(float(loss.detach()))
            else:
                loss, log_prob = objective.loss(xs)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite target loss")
                gradients = list(torch.autograd.grad(loss, xs))
                losses = [float(loss.detach())]
                if not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError("nonfinite image gradient")
            norm = float(sum(g.float().square().sum() for g in gradients).sqrt())
            if norm <= 0:
                raise RuntimeError("zero image gradient")
            current_loss = float(np.mean(losses))
            if current_loss < best_loss:
                best_loss, best_pages = current_loss, [x.detach().cpu() for x in xs]
            with torch.no_grad():
                xs = [project_linf(x - alpha * grad.sign(), original, epsilon, quantize=True).detach().requires_grad_(True)
                      for x, original, grad in zip(xs, clean, gradients)]
            del gradients
            final_step = step
            if step % int(settings["checkpoint_interval"]) == 0 or step == int(settings["max_steps"]):
                with torch.no_grad():
                    loss_now, mean_logp = objective.loss(xs)
                    loss_now, mean_logp = float(loss_now), float(mean_logp)
                if loss_now < best_loss:
                    best_loss, best_pages = loss_now, [x.detach().cpu() for x in xs]
                decoded_memory = objective.decode(xs)
                reloaded, image_info = save_reload(xs, clean, shard / "images" / key, epsilon)
                raw_saved = wrapper.predict(reloaded, prompt)
                entry = {"step": step, "target_loss": loss_now, "mean_target_log_probability": mean_logp,
                    "raw_in_memory": decoded_memory, "raw_reloaded": raw_saved,
                    "in_memory_success": normalize_fn(decoded_memory) == target,
                    "saved_image_success": normalize_fn(raw_saved) == target,
                    "linf_255": max(x["linf_255"] for x in image_info), "gradient_norm": norm,
                    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20}
                logs.append(entry)
                checkpoint_state(checkpoint, adversarial=xs[0], clean_sha256=contract_hash,
                    completed_steps=step, best_loss=best_loss, optimizer=None,
                    config_sha256=provenance["config_sha256"], model_revision=wrapper.revision,
                    extra_state={"contract": contract, "pages": [x.detach().cpu() for x in xs],
                        "best_pages": best_pages, "logs": logs, "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        "runtime_seconds": prior_seconds + time.perf_counter() - started})
                print(json.dumps({"record_id": row["record_id"], "epsilon": epsilon, **entry}), flush=True)
                if entry["saved_image_success"]:
                    break

    reloaded, image_info = save_reload(xs, clean, shard / "images" / key, epsilon)
    raw_saved = wrapper.predict(reloaded, prompt)
    for item, original, page in zip(image_info, clean, reloaded):
        lpips_value, ssim = _perceptibility(original, pil_to_rgb_tensor(page, device), lpips_model)
        item.update(lpips=lpips_value, ssim=ssim)
    with torch.no_grad():
        final_loss, final_logp = objective.loss(xs)
    if not logs:
        raise RuntimeError("no checkpoint was written; checkpoint_interval must be positive")
    return {**parity_stats, "raw_response": raw_saved, "prediction_normalized": normalize_fn(raw_saved),
        "saved_image_success": normalize_fn(raw_saved) == target,
        "in_memory_success": logs[-1]["in_memory_success"], "initial_loss": initial_loss,
        "final_loss": float(final_loss), "best_loss": best_loss,
        "mean_target_log_probability": float(final_logp), "completed_steps": final_step,
        "gradient_norm_final": logs[-1]["gradient_norm"], "optimization_trace": logs,
        "images": image_info, "linf_255": max(x["linf_255"] for x in image_info),
        "lpips": float(np.mean([x["lpips"] for x in image_info])),
        "ssim": float(np.mean([x["ssim"] for x in image_info])),
        "runtime_seconds": prior_seconds + time.perf_counter() - started,
        "peak_gpu_memory_mb": max(x["peak_gpu_memory_mb"] for x in logs),
        "checkpoint_resumed": checkpoint_resumed,
        "target_token_ids_hash": objective.target_token_ids_hash,
        "image_grid_thw": objective.grids.tolist(), "contract_hash": contract_hash}


def _perceptibility(clean, adversarial, lpips_model):
    from skimage.metrics import structural_similarity
    a = clean.detach().permute(1, 2, 0).cpu().numpy()
    b = adversarial.detach().permute(1, 2, 0).cpu().numpy()
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    with torch.inference_mode():
        lpips_value = float(lpips_model(clean.unsqueeze(0).mul(2).sub(1),
                                         adversarial.unsqueeze(0).mul(2).sub(1)).item())
    return lpips_value, ssim
