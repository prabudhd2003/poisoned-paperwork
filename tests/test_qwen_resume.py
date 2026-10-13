"""CPU-only contract tests for resume candidate selection and sharding."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sharding import assigned_record_ids
from attacks.qwen import exact_forward_bpda, save_reload
from stages import qwen_attack, qwen_attack_resume


class ResumeCandidateTests(unittest.TestCase):
    def config(self, mode="smoke"):
        value = {
            "stage": "qwen_attack", "track": "resume", "mode": mode,
            "run_id": "resume_smoke", "owner_user_id": "user5",
            "split": "validation", "datasets": ["resume_parsing_vision"],
            "seed": 566, "baseline_run_id": "baseline_validation_v1",
            "baseline_config": "baseline.json", "baseline_config_sha256": "a" * 64,
            "baseline_predictions": "predictions.parquet",
            "baseline_predictions_sha256": "b" * 64,
            "baseline_manifest": "manifest.json", "baseline_summary": "summary.json",
            "baseline_correct_ids": "ids.txt", "baseline_environment": "environment.json",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "prompt_id": "resume_degree_v1", "question": "Question",
            "processor_tolerance": {"max": 1e-6, "mean": 1e-8, "surrogate_mean": 0.005},
            "attack": {"epsilons_255": [8], "max_steps": 1,
                "checkpoint_interval": 1, "step_size_255": {"8": 1},
                "loss_type": "cross_entropy",
                "multi_page_fallback": "gradient_accumulation_across_pages"},
        }
        if mode != "smoke":
            value["ready_for_production"] = True
        return value

    def test_candidate_selection_filters_and_checks_labels(self):
        import pandas as pd
        model = {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
                 "precision": "bfloat16", "min_pixels": 200704, "max_pixels": 1003520, "max_new_tokens": 32}
        prompt = {"id": "resume_degree_v1", "text": "Return a degree label."}
        baseline = {"run_id": "baseline_validation_v1", "split": "validation",
                    "models": {"qwen": model}, "prompts": {"resume_highest_degree": prompt}}
        def pred(rid, correct, status="complete", parse="ok", label="bachelor"):
            return {"record_id": rid, "model_key": "qwen", "dataset_name": "resume_parsing_vision",
                "experiment_split": "validation", "model_id": model["model_id"],
                "model_revision": model["revision"], "run_id": baseline["run_id"],
                "task": "resume_highest_degree", "prompt_id": prompt["id"], "prompt_text": prompt["text"],
                "clean_target": "bachelor", "attack_target": "master", "attack_eligible": True,
                "clean_correct": correct, "parse_status": parse, "status": status,
                "prediction_normalized": "bachelor" if correct else "master", "raw_response": "bachelor",
                "input_page_count": 1 if rid.endswith("1") else 2,
                "processor_settings": {"precision": "bfloat16", "min_pixels": 200704,
                    "max_pixels": 1003520, "max_new_tokens": 32, "do_sample": False, "batch_size": 1}}
        predictions = pd.DataFrame([pred("resume/a/1", True), pred("resume/a/2", False),
                                    pred("resume/a/3", True, status="failed")])
        manifest = pd.DataFrame([{"record_id": f"resume/a/{i}", "dataset_name": "resume_parsing_vision",
            "experiment_split": "validation", "usable": True, "attack_eligible": True,
            "clean_target": "bachelor", "attack_target": "master", "input_page_count": i}
            for i in (1, 2, 3)])
        selected = qwen_attack_resume.select_candidates(predictions, manifest, baseline)
        self.assertEqual(list(selected.index), ["resume/a/1"])

    def test_shards_are_disjoint_and_cover_all_ids(self):
        ids = [f"resume/{i:03d}" for i in range(31)]
        shards = [assigned_record_ids(ids, worker, 5) for worker in range(5)]
        self.assertEqual(sum(map(len, shards)), len(ids))
        self.assertEqual(set().union(*(set(shard) for shard in shards)), set(ids))
        for left in range(5):
            for right in range(left + 1, 5):
                self.assertTrue(set(shards[left]).isdisjoint(shards[right]))

    def test_smoke_selection_handles_record_id_as_index_and_column(self):
        import pandas as pd
        candidates = pd.DataFrame([
            {"record_id": "resume/z", "input_page_count": 2},
            {"record_id": "resume/a", "input_page_count": 1},
        ]).set_index("record_id", drop=False)
        selected = qwen_attack_resume.select_smoke_candidates(candidates, 1)
        self.assertEqual(list(selected.index), ["resume/a", "resume/z"])
        self.assertEqual(list(selected.record_id), ["resume/a", "resume/z"])

    def test_non_smoke_rejects_one_worker(self):
        with self.assertRaisesRegex(ValueError, "num_workers=5"):
            config = self.config("validation")
            qwen_attack_resume.run(config, Path("unused"), "user1", 0, 1)

    def test_production_is_gated_until_smoke_review(self):
        config = self.config("validation")
        config["ready_for_production"] = False
        with self.assertRaisesRegex(ValueError, "production is gated"):
            qwen_attack_resume._validate_attack_config(config)
        config["ready_for_production"] = True
        qwen_attack_resume._validate_attack_config(config)

    def test_test_mode_requires_validation_link(self):
        config = self.config("test")
        config["split"] = "test"
        config["frozen_for_test"] = True
        with self.assertRaisesRegex(ValueError, "validation_run_id"):
            qwen_attack_resume._validate_attack_config(config)
        config["validation_run_id"] = "qwen_resume_validation_v1"
        qwen_attack_resume._validate_attack_config(config)

    def test_loss_type_is_not_silently_ignored(self):
        config = self.config()
        config["attack"]["loss_type"] = "paper_margin"
        with self.assertRaisesRegex(ValueError, "cross_entropy"):
            qwen_attack_resume._validate_attack_config(config)

    def test_save_reload_is_atomic_and_bounded(self):
        clean = torch.full((3, 4, 5), 128 / 255)
        candidate = clean + 20 / 255
        with tempfile.TemporaryDirectory() as directory:
            pages, metadata = save_reload(
                [candidate], [clean], Path(directory) / "experiment", 8
            )
            self.assertEqual(len(pages), 1)
            self.assertLessEqual(metadata[0]["linf_255"], 8.0001)
            self.assertTrue((Path(directory) / "experiment/page_000.png").is_file())
            self.assertFalse((Path(directory) / "experiment/page_000.png.tmp").exists())

    def test_bpda_uses_official_forward_and_surrogate_gradient(self):
        surrogate = torch.tensor([1.0, 2.0], requires_grad=True)
        official = torch.tensor([10.0, 20.0])
        result = exact_forward_bpda(surrogate, official)
        torch.testing.assert_close(result, official)
        result.sum().backward()
        torch.testing.assert_close(surrogate.grad, torch.ones_like(surrogate))

    def test_router_rejects_receipt_dataset_for_resume(self):
        with self.assertRaisesRegex(ValueError, "accepts datasets"):
            qwen_attack._validate({"stage": "qwen_attack", "track": "resume", "datasets": ["sroie"]})


if __name__ == "__main__":
    unittest.main()
