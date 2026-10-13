import random
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attacks.donut import (
    checkpoint_state,
    load_checkpoint_state,
    project_linf,
    targeted_image_attack,
    targeted_token_loss,
)
from models.donut import differentiable_donut_preprocess
from stages.donut_attack import (
    DONUT_REQUIRED_RECORD_FIELDS,
    _record_is_final,
    _validate_config,
)


class DonutAttackPrimitiveTests(unittest.TestCase):
    def smoke_config(self):
        return {
            "stage": "donut_attack",
            "mode": "smoke",
            "run_id": "smoke",
            "seed": 566,
            "split": "validation",
            "output_root": "outputs/donut_attack",
            "datasets": ["sroie", "cord_v2"],
            "smoke_per_dataset": 1,
            "baseline_predictions": "predictions.parquet",
            "baseline_merge_manifest": "merge_manifest.json",
            "baseline_predictions_sha256": "a" * 64,
            "baseline_config_sha256": "b" * 64,
            "baseline_git_commit": "c" * 40,
            "attack": {
                "epsilons_255": [8],
                "step_size_255": {"8": 2},
                "max_steps": 2,
                "loss_type": "paper_margin",
                "optimizer": "adam",
            },
        }

    def test_config_gates_production_until_smoke_is_reviewed(self):
        config = self.smoke_config()
        _validate_config(config)
        config["mode"] = "validation"
        config.pop("smoke_per_dataset")
        with self.assertRaisesRegex(ValueError, "ready_for_production"):
            _validate_config(config)
        config["ready_for_production"] = True
        _validate_config(config)

    def test_final_record_requires_exact_provenance(self):
        config = self.smoke_config()
        config_sha = "d" * 64
        git_commit = "e" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.json"
            record = {field: None for field in DONUT_REQUIRED_RECORD_FIELDS}
            record.update({
                "run_id": "smoke",
                "stage": "donut_attack",
                "experiment_key": "key",
                "record_id": "sroie/validation/example",
                "experiment_split": "validation",
                "model_id": "naver-clova-ix/donut-base-finetuned-docvqa",
                "model_revision": "b19d2e332684b0e2d35d9144ce34047767335cf8",
                "git_commit": git_commit,
                "config_sha256": config_sha,
                "baseline_predictions_sha256": "a" * 64,
                "worker_id": 0,
                "seed": 566,
                "epsilon_255": 8.0,
                "status": "complete",
                "error": None,
            })
            path.write_text(json.dumps(record))
            self.assertTrue(_record_is_final(
                path, config=config, config_sha256=config_sha,
                git_commit=git_commit, record_id="sroie/validation/example",
                epsilon=8.0, worker_id=0,
            ))
            record["git_commit"] = "f" * 40
            path.write_text(json.dumps(record))
            self.assertFalse(_record_is_final(
                path, config=config, config_sha256=config_sha,
                git_commit=git_commit, record_id="sroie/validation/example",
                epsilon=8.0, worker_id=0,
            ))

    def test_projection_enforces_pixel_space_bound_and_domain(self):
        clean = torch.tensor([[[0.0, 0.5, 1.0]]]).repeat(3, 1, 1)
        candidate = torch.tensor([[[-1.0, 0.9, 2.0]]]).repeat(3, 1, 1)
        projected = project_linf(candidate, clean, 8, quantize=True)
        self.assertGreaterEqual(float(projected.min()), 0.0)
        self.assertLessEqual(float(projected.max()), 1.0)
        self.assertLessEqual(float((projected - clean).abs().max() * 255), 8.0001)
        self.assertTrue(torch.equal(projected * 255, torch.round(projected * 255)))

    def test_zero_epsilon_is_exact(self):
        clean = torch.rand(3, 4, 5)
        self.assertTrue(torch.equal(project_linf(torch.rand_like(clean), clean, 0), clean))

    def test_losses_only_use_target_positions(self):
        logits = torch.zeros(1, 5, 4, requires_grad=True)
        target_ids = torch.tensor([2, 1])
        positions = torch.tensor([1, 3])
        ce, mean_logp = targeted_token_loss(
            logits, target_ids, positions, loss_type="cross_entropy"
        )
        ce.backward(retain_graph=True)
        self.assertTrue(torch.equal(logits.grad[0, 0], torch.zeros(4)))
        self.assertTrue(torch.equal(logits.grad[0, 2], torch.zeros(4)))
        self.assertTrue(torch.equal(logits.grad[0, 4], torch.zeros(4)))
        self.assertAlmostEqual(float(ce), np.log(4), places=5)
        self.assertAlmostEqual(float(mean_logp), -np.log(4), places=5)

        logits.grad = None
        margin, _ = targeted_token_loss(
            logits, target_ids, positions, loss_type="paper_margin"
        )
        self.assertEqual(float(margin), 0.0)  # Ties choose target IDs at these positions.

        separated = torch.zeros(1, 2, 3)
        separated[0, 0, 0] = 3
        separated[0, 0, 2] = 1
        margin, _ = targeted_token_loss(
            separated, torch.tensor([2]), torch.tensor([0]), loss_type="paper_margin"
        )
        self.assertEqual(float(margin), 2.0)

    def test_sign_attack_moves_toward_target_without_exceeding_epsilon(self):
        clean = torch.full((3, 3, 3), 0.5)

        def objective(image):
            return ((image - 0.8) ** 2).mean(), {}

        result = targeted_image_attack(
            clean,
            objective,
            epsilon_255=8,
            step_size_255=2,
            max_steps=10,
            optimizer_name="sign",
            quantize_each_step=True,
        )
        self.assertGreater(float(result.adversarial.mean()), float(clean.mean()))
        self.assertLessEqual(float((result.adversarial - clean).abs().max() * 255), 8.0001)
        self.assertEqual(result.completed_steps, 10)

    def test_checkpoint_metadata_contains_gradient_norms(self):
        clean = torch.full((3, 2, 2), 0.5)
        captured = {}

        def objective(image):
            return ((image - 0.7) ** 2).mean(), {}

        def checkpoint(step, adversarial, best_loss, loss, metadata, optimizer):
            captured.update(metadata)

        targeted_image_attack(
            clean,
            objective,
            epsilon_255=2,
            step_size_255=1,
            max_steps=1,
            checkpoint_interval=1,
            checkpoint_callback=checkpoint,
        )
        self.assertGreater(captured["gradient_l2"], 0)
        self.assertGreater(captured["gradient_linf"], 0)

    def test_model_parameters_are_not_part_of_attack_update(self):
        model = torch.nn.Conv2d(3, 1, 1, bias=False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = model.weight.detach().clone()
        clean = torch.rand(3, 4, 4)

        def objective(image):
            return model(image.unsqueeze(0)).square().mean(), {}

        targeted_image_attack(
            clean,
            objective,
            epsilon_255=2,
            step_size_255=1,
            max_steps=1,
            optimizer_name="sign",
        )
        self.assertTrue(torch.equal(before, model.weight))
        self.assertIsNone(model.weight.grad)

    def test_adam_resume_matches_uninterrupted_attack(self):
        clean = torch.full((3, 2, 2), 0.5)

        def objective(image):
            return ((image - 0.7) ** 2).mean(), {}

        uninterrupted = targeted_image_attack(
            clean,
            objective,
            epsilon_255=8,
            step_size_255=1,
            max_steps=4,
            optimizer_name="adam",
            quantize_each_step=False,
        )
        captured = {}

        def capture(step, adversarial, best_loss, loss, metadata, optimizer):
            captured.update(
                adversarial=adversarial.detach().clone(),
                best_loss=best_loss,
                optimizer_state=copy.deepcopy(optimizer.state_dict()),
            )

        targeted_image_attack(
            clean,
            objective,
            epsilon_255=8,
            step_size_255=1,
            max_steps=2,
            optimizer_name="adam",
            quantize_each_step=False,
            checkpoint_interval=2,
            checkpoint_callback=capture,
        )
        resumed = targeted_image_attack(
            clean,
            objective,
            epsilon_255=8,
            step_size_255=1,
            max_steps=4,
            optimizer_name="adam",
            quantize_each_step=False,
            start_adversarial=captured["adversarial"],
            start_step=2,
            start_best_loss=captured["best_loss"],
            optimizer_state=captured["optimizer_state"],
        )
        self.assertTrue(torch.allclose(uninterrupted.adversarial, resumed.adversarial))
        self.assertAlmostEqual(uninterrupted.best_loss, resumed.best_loss, places=7)

    def test_checkpoint_round_trip_and_provenance(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            checkpoint_state(
                path,
                adversarial=torch.rand(3, 2, 2),
                clean_sha256="clean",
                completed_steps=10,
                best_loss=0.5,
                optimizer=None,
                config_sha256="config",
                model_revision="revision",
            )
            state = load_checkpoint_state(
                path,
                clean_sha256="clean",
                config_sha256="config",
                model_revision="revision",
                device=torch.device("cpu"),
            )
            self.assertEqual(state["completed_steps"], 10)
            with self.assertRaises(ValueError):
                load_checkpoint_state(
                    path,
                    clean_sha256="different",
                    config_sha256="config",
                    model_revision="revision",
                    device=torch.device("cpu"),
                )

    def test_differentiable_preprocessing_preserves_gradients(self):
        image = torch.rand(3, 7, 5, requires_grad=True)
        output = differentiable_donut_preprocess(
            image,
            height=12,
            width=10,
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            align_long_axis=False,
        )
        self.assertEqual(tuple(output.shape), (3, 12, 10))
        output.sum().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
