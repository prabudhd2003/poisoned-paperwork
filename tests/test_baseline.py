import copy
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import row_images, select_baseline_manifest
from models.qwen import build_messages
from stages import baseline


def config():
    return json.loads((REPO_ROOT / "configs" / "baseline" / "smoke.json").read_text())


def synthetic_manifest():
    rows = []
    definitions = (
        ("sroie", "receipt_total", "10.00"),
        ("cord_v2", "receipt_total", "10000"),
        ("resume_parsing_vision", "resume_highest_degree", "bachelor"),
    )
    for dataset_name, task, target in definitions:
        for index in range(3):
            document_id = f"{dataset_name}-{index}"
            rows.append({
                "record_id": f"{dataset_name}/validation/{document_id}",
                "dataset_name": dataset_name,
                "document_id": document_id,
                "source_split": "validation",
                "source_index": index,
                "experiment_split": "validation",
                "task": task,
                "clean_target": target,
                "attack_target": "other",
                "usable": True,
                "attack_eligible": True,
            })
    return pd.DataFrame(rows)


class ConfigAndPlanTests(unittest.TestCase):
    def test_smoke_plan_has_exact_model_scope(self):
        cfg = config()
        baseline.validate_config(cfg)
        selected = select_baseline_manifest(synthetic_manifest(), cfg)
        experiments = baseline.build_experiments(selected, cfg)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(experiments), 10)
        qwen = [item for item in experiments.values() if item["model_key"] == "qwen"]
        donut = [item for item in experiments.values() if item["model_key"] == "donut"]
        self.assertEqual(len(qwen), 6)
        self.assertEqual(len(donut), 4)
        self.assertFalse(any(item["dataset_name"] == "resume_parsing_vision" for item in donut))

    def test_resume_manifest_task_is_the_real_processed_name(self):
        cfg = config()
        selected = select_baseline_manifest(synthetic_manifest(), cfg)
        tasks = set(selected.loc[
            selected["dataset_name"].eq("resume_parsing_vision"), "task"
        ])
        self.assertEqual(tasks, {"resume_highest_degree"})

    def test_test_split_must_be_frozen_and_link_validation(self):
        cfg = config()
        cfg.pop("smoke_per_dataset")
        cfg["split"] = "test"
        with self.assertRaises(ValueError):
            baseline.validate_config(cfg)
        cfg["frozen_for_test"] = True
        with self.assertRaises(ValueError):
            baseline.validate_config(cfg)
        cfg["validation_run_id"] = "baseline_validation_v1"
        baseline.validate_config(cfg)

    def test_changed_model_or_prompt_is_rejected(self):
        cfg = config()
        wrong_model = copy.deepcopy(cfg)
        wrong_model["models"]["qwen"]["revision"] = "wrong"
        with self.assertRaises(ValueError):
            baseline.validate_config(wrong_model)
        wrong_prompt = copy.deepcopy(cfg)
        wrong_prompt["prompts"]["receipt_total"]["text"] = "Total?"
        with self.assertRaises(ValueError):
            baseline.validate_config(wrong_prompt)


class ImagesAndMessagesTests(unittest.TestCase):
    def test_all_resume_pages_precede_question(self):
        self.assertEqual(row_images({"images": ["one", "two"]}), ["one", "two"])
        content = build_messages(2, "question")[0]["content"]
        self.assertEqual([item["type"] for item in content], ["image", "image", "text"])

    def test_receipt_has_one_image(self):
        self.assertEqual(row_images({"image": "receipt"}), ["receipt"])


if __name__ == "__main__":
    unittest.main()
