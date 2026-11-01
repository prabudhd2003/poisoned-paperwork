"""CPU contract tests for the public JPEG-EOT adaptive attack."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class JpegEotTests(unittest.TestCase):
    def test_real_jpeg_forward_and_identity_backward(self):
        from attacks.jpeg_eot import jpeg_bpda_tensor
        from defenses import jpeg_roundtrip
        from models.donut import pil_to_rgb_tensor, rgb_tensor_to_pil

        clean = torch.linspace(0, 1, 3 * 20 * 24).reshape(3, 20, 24)
        clean.requires_grad_(True)
        attacked = jpeg_bpda_tensor(clean, 85)
        official = pil_to_rgb_tensor(
            jpeg_roundtrip(rgb_tensor_to_pil(clean.detach()), 85),
            torch.device("cpu"),
        )
        torch.testing.assert_close(attacked.detach(), official)
        attacked.sum().backward()
        torch.testing.assert_close(clean.grad, torch.ones_like(clean))

    def test_every_visual_defense_has_exact_forward_and_bpda_gradient(self):
        from attacks.jpeg_eot import transform_bpda_tensor

        for transform_id in (
            "identity", "jpeg_q85", "resize_0.85", "blur_0.5",
            "brightness_0.95",
        ):
            clean = torch.linspace(0, 1, 3 * 20 * 24).reshape(3, 20, 24)
            clean.requires_grad_(True)
            transformed = transform_bpda_tensor(clean, transform_id)
            self.assertEqual(transformed.shape, clean.shape)
            transformed.sum().backward()
            torch.testing.assert_close(clean.grad, torch.ones_like(clean))

    def test_quality_schedule_is_deterministic(self):
        from attacks.jpeg_eot import quality_schedule

        first = quality_schedule([65, 75, 85, 95], 2, seed=566, step=11)
        second = quality_schedule([65, 75, 85, 95], 2, seed=566, step=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)

    def test_hard_view_weights_emphasize_the_largest_loss(self):
        from attacks.jpeg_eot import hard_view_weights, view_schedule

        weights = hard_view_weights(
            [0.2, 3.0, 0.5], temperature=0.75, floor=0.01
        )
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[1], weights[2])
        first = view_schedule(3, 2, seed=566, step=7, weights=weights)
        second = view_schedule(3, 2, seed=566, step=7, weights=weights)
        self.assertEqual(first, second)

    def test_frequency_update_is_bounded_and_quantized(self):
        from attacks.jpeg_eot import frequency_aware_update, low_frequency_component

        original = torch.full((3, 24, 20), 0.5)
        checkerboard = torch.arange(24).reshape(-1, 1) + torch.arange(20)
        gradient = ((checkerboard % 2) * 2 - 1).float().repeat(3, 1, 1)
        low = low_frequency_component(gradient, 0.25)
        self.assertLess(float(low.abs().mean()), float(gradient.abs().mean()))
        updated = frequency_aware_update(
            original,
            original,
            gradient,
            alpha=4 / 255,
            epsilon_255=8,
            cutoff=0.25,
            gradient_mix=0.75,
            delta_mix=0.5,
        )
        self.assertLessEqual(float((updated - original).abs().max() * 255), 8.0001)
        torch.testing.assert_close(updated * 255, torch.round(updated * 255))

    def test_exact_resize_vote_views_are_reproducible(self):
        from attacks.jpeg_eot import resolved_training_views

        config = json.loads(
            (ROOT / "configs/jpeg_eot_attack/public_full.json").read_text()
        )
        first = resolved_training_views(config, "sroie/validation/example", 2)
        second = resolved_training_views(config, "sroie/validation/example", 2)
        self.assertEqual(first, second)
        exact = [view for view in first if str(view.get("source", "")).startswith("exact_resize_vote")]
        self.assertEqual(len(exact), config["defense"]["resize_votes"])
        self.assertTrue(all(len(view["page_transforms"]) == 2 for view in exact))

    def test_full_config_is_frozen_to_three_nonzero_epsilons(self):
        from stages.jpeg_eot_attack import validate_config

        config = json.loads(
            (ROOT / "configs/jpeg_eot_attack/public_full.json").read_text()
        )
        config["ready_for_production"] = True
        validate_config(config)
        self.assertEqual(config["attack"]["epsilons_255"], [4, 8, 16])
        self.assertIsNone(config["document_limit_per_dataset"])
        transforms = {view["transform"] for view in config["attack"]["training_views"]}
        self.assertTrue({"jpeg_q85", "resize_0.85", "blur_0.5"}.issubset(transforms))
        self.assertEqual(config["attack"]["evaluation_interval"] % config["attack"]["checkpoint_interval"], 0)

    def test_frequency_hardview_full_config_is_valid(self):
        from stages.jpeg_eot_attack import validate_config

        config = json.loads(
            (ROOT / "configs/jpeg_eot_attack/public_frequency_hardview.json").read_text()
        )
        config["ready_for_production"] = True
        validate_config(config)
        self.assertEqual(
            config["attack"]["method"], "frequency_hardview_minimax_eot"
        )
        self.assertEqual(config["attack"]["view_sampling"], "hard_view")
        self.assertTrue(config["attack"]["frequency_aware"]["enabled"])

    def test_epsilon_two_cannot_enter_full_study_silently(self):
        from stages.jpeg_eot_attack import validate_config

        config = json.loads(
            (ROOT / "configs/jpeg_eot_attack/public_full.json").read_text()
        )
        config["ready_for_production"] = True
        config["attack"]["epsilons_255"] = [2, 4, 8, 16]
        with self.assertRaisesRegex(ValueError, "epsilons"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
