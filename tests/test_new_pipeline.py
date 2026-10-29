import tempfile
import unittest
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class DefenseTests(unittest.TestCase):
    def setUp(self):
        from defenses import evaluate_defense, evaluate_semantic_consistency
        self.evaluate = evaluate_defense
        self.evaluate_semantic = evaluate_semantic_consistency
        self.page = Image.fromarray(np.full((24, 30, 3), 240, dtype=np.uint8))
        self.config = {
            "jpeg_quality": 85, "resize_votes": 5, "defense_seed": 566,
            "multiview_transforms": [
                "identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95"
            ],
            "multiview_minimum_votes": 3,
        }

    def test_selective_multiview_consensus(self):
        result, views = self.evaluate(
            pages=[self.page], defense_id="selective_multiview", record_id="x",
            predict=lambda pages: "10.00", normalize=lambda value: value,
            config=self.config,
        )
        self.assertEqual(result.final_prediction, "10.00")
        self.assertFalse(result.abstained)
        self.assertEqual(result.additional_vlm_passes, 4)
        self.assertEqual(len(views), 5)

    def test_selective_multiview_abstains(self):
        answers = iter(["1", "2", "3", "4", "5"])
        result, _ = self.evaluate(
            pages=[self.page], defense_id="selective_multiview", record_id="x",
            predict=lambda pages: next(answers), normalize=lambda value: value,
            config=self.config,
        )
        self.assertIsNone(result.final_prediction)
        self.assertTrue(result.abstained)

    def test_random_resize_is_reproducible(self):
        first, _ = self.evaluate(
            pages=[self.page], defense_id="resize_vote", record_id="x",
            predict=lambda pages: str(np.asarray(pages[0]).sum()), normalize=lambda value: value,
            config=self.config,
        )
        second, _ = self.evaluate(
            pages=[self.page], defense_id="resize_vote", record_id="x",
            predict=lambda pages: str(np.asarray(pages[0]).sum()), normalize=lambda value: value,
            config=self.config,
        )
        self.assertEqual(first.raw_predictions, second.raw_predictions)

    def test_semantic_consistency_pairs_views_and_prompts(self):
        observed = []
        prompts = ["direct", "paraphrase", "verify", "evidence", "alternate"]
        result, views = self.evaluate_semantic(
            pages=[self.page],
            record_id="x",
            predict=lambda pages, prompt: observed.append(prompt) or "10.00",
            normalize=lambda value: value,
            transforms=["identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95"],
            prompts=prompts,
            minimum_votes=3,
        )
        self.assertEqual(observed, prompts)
        self.assertEqual(result.final_prediction, "10.00")
        self.assertFalse(result.abstained)
        self.assertEqual(result.additional_vlm_passes, 4)
        self.assertEqual(len(views), 5)

    def test_semantic_consistency_abstains_without_majority(self):
        answers = iter(["1", "2", "3", "4", "5"])
        result, _ = self.evaluate_semantic(
            pages=[self.page],
            record_id="x",
            predict=lambda pages, prompt: next(answers),
            normalize=lambda value: value,
            transforms=["identity", "jpeg_q85", "resize_0.85", "blur_0.5", "brightness_0.95"],
            prompts=["a", "b", "c", "d", "e"],
            minimum_votes=3,
        )
        self.assertIsNone(result.final_prediction)
        self.assertTrue(result.abstained)

    def test_public_final_defense_config_is_valid_when_frozen(self):
        from stages.defenses import validate_config

        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs/defenses/qwen_public_final.json").read_text()
        )
        config["ready_for_production"] = True
        validate_config(config)
        self.assertEqual(config["attack_epsilons_255"], [4, 8, 16])
        self.assertIsNone(config["study_documents_per_dataset"])


class PatchTests(unittest.TestCase):
    def test_patch_changes_only_box(self):
        import torch
        from attacks.eot_patch import apply_patch, choose_patch_box

        clean = torch.zeros(3, 40, 60)
        clean[:, 25:, 45:] = 1
        box = choose_patch_box(clean, fraction=0.2, inset=0)
        patch = torch.full((3, box["side"], box["side"]), 0.5)
        attacked = apply_patch(clean, patch, box)
        mask = torch.zeros_like(clean, dtype=torch.bool)
        mask[:, box["top"]:box["top"] + box["side"], box["left"]:box["left"] + box["side"]] = True
        self.assertTrue(torch.equal(clean[~mask], attacked[~mask]))
        self.assertTrue(torch.all(attacked[mask] == 0.5))


class AnalysisTests(unittest.TestCase):
    def test_comparison_requires_matching_dimensions(self):
        from result_analysis import show_comparison

        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder) / "a.png"
            b = Path(folder) / "b.png"
            Image.new("RGB", (10, 10)).save(a)
            Image.new("RGB", (11, 10)).save(b)
            with self.assertRaises(ValueError):
                show_comparison(a, b)


if __name__ == "__main__":
    unittest.main()
