"""CPU contract tests for the public JPEG-EOT adaptive attack."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class JpegEotTests(unittest.TestCase):
    def test_reviewed_partial_run_compatibility_is_exact(self):
        from stages.jpeg_eot_attack import (
            _legacy_commit_is_approved,
            _scientific_resume_fingerprint,
        )

        registry = json.loads(
            (ROOT / "configs/provenance/jpeg_eot_resume_compatibility.json")
            .read_text()
        )
        observed_fingerprint = _scientific_resume_fingerprint(ROOT)
        approvals = registry["approvals"]
        self.assertTrue(approvals)
        self.assertTrue(all(
            item["approved_scientific_sha256"] == observed_fingerprint
            for item in approvals
        ))

        approval = approvals[0]
        expected = {
            "run_id": approval["run_id"],
            "config_sha256": approval["config_sha256"],
        }
        self.assertTrue(_legacy_commit_is_approved(
            ROOT, expected, approval["record_git_commit"]
        ))
        self.assertFalse(_legacy_commit_is_approved(
            ROOT, {**expected, "config_sha256": "0" * 64},
            approval["record_git_commit"],
        ))
        self.assertFalse(_legacy_commit_is_approved(
            ROOT, expected, "0" * 40
        ))

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

    def test_matched_donut_adaptive_configs_are_valid(self):
        from stages.jpeg_eot_attack import validate_config

        for name, method, track in (
            ("donut_public_full.json", "uniform_multiview_eot", "donut_uniform_eot"),
            (
                "donut_public_frequency_hardview.json",
                "frequency_hardview_minimax_eot",
                "donut_hardview_eot",
            ),
        ):
            config = json.loads((ROOT / "configs/jpeg_eot_attack" / name).read_text())
            config["ready_for_production"] = True
            validate_config(config)
            self.assertEqual(config["model_key"], "donut")
            self.assertEqual(config["track"], track)
            self.assertEqual(config["attack"]["method"], method)
            self.assertEqual(config["attack"]["epsilons_255"], [4, 8, 16])

    def test_collected_qwen_and_donut_adaptive_templates_are_valid(self):
        from stages.jpeg_eot_attack import validate_config
        from scripts.prepare_stage_config import load_template

        cases = (
            (
                "qwen_collected_full.json", "qwen", 110,
                "qwen_collected_uniform_eot", "uniform_multiview_eot",
            ),
            (
                "qwen_collected_frequency_hardview.json", "qwen", 110,
                "qwen_collected_hardview_eot",
                "frequency_hardview_minimax_eot",
            ),
            (
                "donut_collected_full.json", "donut", 90,
                "donut_collected_uniform_eot", "uniform_multiview_eot",
            ),
            (
                "donut_collected_frequency_hardview.json", "donut", 90,
                "donut_collected_hardview_eot",
                "frequency_hardview_minimax_eot",
            ),
        )
        for name, model_key, documents, track, method in cases:
            with self.subTest(name=name):
                config = load_template(ROOT / "configs/jpeg_eot_attack" / name)
                config["ready_for_production"] = True
                validate_config(config)
                self.assertEqual(config.get("model_key", "qwen"), model_key)
                self.assertEqual(config["dataset_scope"], "collected_receipts")
                self.assertEqual(
                    config["expected_documents_by_dataset"],
                    {"collected_receipts": documents},
                )
                self.assertEqual(config["track"], track)
                self.assertEqual(config["attack"]["method"], method)

    def test_collected_plan_restores_frozen_primary_prompt(self):
        import pandas as pd

        from checkpoints import sha256_file
        from stages.jpeg_eot_attack import _load_plan
        from scripts.prepare_stage_config import load_template

        config = load_template(
            ROOT / "configs/jpeg_eot_attack/qwen_collected_full.json"
        )
        config["ready_for_production"] = True
        rows = []
        for document_index in range(110):
            for epsilon in (2, 4, 8, 16):
                rows.append({
                    "record_id": (
                        "collected_receipts/external_test/"
                        f"c{document_index:05d}"
                    ),
                    "dataset_name": "collected_receipts",
                    "experiment_split": "external_test",
                    "clean_target": "10.00",
                    "attack_target": "100.00",
                    "status": "complete",
                    "epsilon_255": epsilon,
                    "model_id": config["model"]["model_id"],
                    "model_revision": config["model"]["revision"],
                })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.parquet"
            pd.DataFrame(rows).to_parquet(source, index=False)
            config["attack_sources"] = [{
                "name": "qwen_collected",
                "records": str(source),
                "sha256": sha256_file(source),
            }]
            selected = _load_plan(root, config)

        self.assertEqual(len(selected), 110)
        self.assertEqual(
            selected.iloc[0].prompt_text,
            config["semantic_prompts"]["receipt"][0],
        )

    def test_public_plan_does_not_invent_a_missing_prompt(self):
        import pandas as pd

        from checkpoints import sha256_file
        from stages.jpeg_eot_attack import _load_plan

        config = json.loads(
            (ROOT / "configs/jpeg_eot_attack/public_full.json").read_text()
        )
        config["ready_for_production"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.parquet"
            pd.DataFrame([{"record_id": "example"}]).to_parquet(
                source, index=False
            )
            config["attack_sources"] = [{
                "name": "public",
                "records": str(source),
                "sha256": sha256_file(source),
            }]
            with self.assertRaisesRegex(ValueError, "prompt_text"):
                _load_plan(root, config)

    def test_collected_defense_templates_are_valid(self):
        from stages.defenses import validate_config
        from scripts.prepare_stage_config import load_template

        for name, model_key, documents in (
            ("qwen_collected_final.json", "qwen", 110),
            ("donut_collected_final.json", "donut", 90),
        ):
            with self.subTest(name=name):
                config = load_template(ROOT / "configs/defenses" / name)
                config["ready_for_production"] = True
                validate_config(config)
                self.assertEqual(config.get("model_key", "qwen"), model_key)
                self.assertEqual(
                    config["expected_documents_by_dataset"],
                    {"collected_receipts": documents},
                )
                self.assertEqual(config["expected_attack_records"], documents * 3)
                self.assertEqual(config["attack_epsilons_255"], [4, 8, 16])

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
