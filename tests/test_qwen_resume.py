"""CPU-only contract tests for resume candidate selection and sharding."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sharding import assigned_record_ids
from stages import qwen_attack, qwen_attack_resume


class ResumeCandidateTests(unittest.TestCase):
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

    def test_non_smoke_rejects_one_worker(self):
        with self.assertRaisesRegex(ValueError, "num_workers=5"):
            config = {"stage": "qwen_attack", "track": "resume", "datasets": ["resume_parsing_vision"],
                "split": "validation", "mode": "validation", "attack": {"epsilons_255": [8],
                    "max_steps": 1, "checkpoint_interval": 1, "step_size_255": {"8": 1},
                    "multi_page_fallback": "gradient_accumulation_across_pages"},
                "processor_tolerance": {"max": 0.1, "mean": 0.005},
                "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
                "prompt_id": "resume_degree_v1"}
            qwen_attack_resume.run(config, Path("unused"), "user1", 0, 1)

    def test_router_rejects_receipt_dataset_for_resume(self):
        with self.assertRaisesRegex(ValueError, "accepts datasets"):
            qwen_attack._validate({"stage": "qwen_attack", "track": "resume", "datasets": ["sroie"]})


if __name__ == "__main__":
    unittest.main()
