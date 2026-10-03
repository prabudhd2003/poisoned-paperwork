import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from models.qwen import DEFAULT_MODEL_ID, DEFAULT_REVISION, build_messages
from normalization import normalize_resume_degree
from stages import baseline_qwen as bq


def base_config():
    return {
        "run_id": "t",
        "seed": 1,
        "split": "validation",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_REVISION,
        "jobs": [
            {"task": "receipt_total", "datasets": ["sroie"], "prompt_id": "receipt_total_v1"},
            {"task": "resume_degree", "datasets": ["r"], "prompt_id": "resume_degree_v1"},
        ],
    }


def manifest_row(dataset="sroie", target="250.00"):
    return {
        "record_id": f"{dataset}/train/1",
        "dataset_name": dataset,
        "source_split": "train",
        "source_index": 3,
        "clean_target": target,
        "attack_target": "2500.00",
        "attack_eligible": True,
    }


def make_record(raw, error=None, dataset="sroie", target="250.00", task="receipt_total"):
    job = {"task": task, "prompt_id": "receipt_total_v1"}
    return bq.build_record(
        config=base_config(), config_sha256="abc", job=job,
        manifest_row=manifest_row(dataset, target), raw_response=raw,
        runtime_seconds=0.5, peak_gpu_memory_mb=100.0, page_count=1, worker_id=0,
        git_commit="deadbeef", settings={}, started_at="s", ended_at="e", error=error,
    )


class ResumeNormalizationTests(unittest.TestCase):
    def test_accepts_exact_labels(self):
        self.assertEqual(normalize_resume_degree("Bachelor"), "bachelor")
        self.assertEqual(normalize_resume_degree("  master \n"), "master")
        self.assertEqual(normalize_resume_degree("certificate_or_diploma"), "certificate_or_diploma")

    def test_rejects_everything_else(self):
        for text in ("Master of Science", "bachelor and master", "", None, "phd"):
            self.assertIsNone(normalize_resume_degree(text))


class ConfigTests(unittest.TestCase):
    def test_valid(self):
        bq.validate_config(base_config())

    def test_test_split_requires_frozen(self):
        config = base_config()
        config["split"] = "test"
        with self.assertRaises(ValueError):
            bq.validate_config(config)
        config["frozen_for_test"] = True
        bq.validate_config(config)

    def test_wrong_revision_rejected(self):
        config = base_config()
        config["model_revision"] = "0" * 40
        with self.assertRaises(ValueError):
            bq.validate_config(config)

    def test_editing_existing_prompt_rejected(self):
        config = base_config()
        config["jobs"][0]["prompt_text"] = "something else"
        with self.assertRaises(ValueError):
            bq.validate_config(config)

    def test_new_prompt_needs_text(self):
        config = copy.deepcopy(base_config())
        config["jobs"][0]["prompt_id"] = "receipt_total_v2"
        with self.assertRaises(ValueError):
            bq.validate_config(config)
        config["jobs"][0]["prompt_text"] = "Total?"
        bq.validate_config(config)


class KeyAndRecordTests(unittest.TestCase):
    def test_key_is_safe_and_distinguishes_factors(self):
        a = bq.experiment_key("sroie/train/1", "receipt_total", "p1", "validation")
        b = bq.experiment_key("sroie/train/1", "receipt_total", "p2", "validation")
        c = bq.experiment_key("sroie/train/1", "receipt_total", "p1", "test")
        self.assertNotIn("/", a)
        self.assertEqual(len({a, b, c}), 3)

    def test_clean_correct_requires_exact_parse(self):
        self.assertTrue(make_record("Total: 250")["clean_correct"])
        self.assertFalse(make_record("10 or 20")["clean_correct"])
        self.assertFalse(make_record("251")["clean_correct"])

    def test_parse_failure_is_explicit(self):
        record = make_record("no idea")
        self.assertIsNone(record["prediction_normalized"])
        self.assertEqual(record["parse_status"], "failed")
        self.assertEqual(record["status"], "complete")

    def test_inference_error_is_failed_record(self):
        record = make_record(None, error="RuntimeError: boom")
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["clean_correct"])
        self.assertEqual(record["error"], "RuntimeError: boom")

    def test_record_has_all_required_fields(self):
        record = make_record("250")
        for field in bq.REQUIRED_RECORD_FIELDS:
            self.assertIn(field, record)


class ResumeSkipTests(unittest.TestCase):
    def test_valid_skipped_corrupt_and_failed_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            self.assertFalse(bq.record_is_valid(path, "abc"))
            path.write_text(json.dumps(make_record("250")))
            self.assertTrue(bq.record_is_valid(path, "abc"))
            self.assertFalse(bq.record_is_valid(path, "different-config"))
            path.write_text(json.dumps(make_record(None, error="x")))
            self.assertFalse(bq.record_is_valid(path, "abc"))
            self.assertTrue(bq.record_is_final(path, "abc"))
            path.write_text("{not json")
            self.assertFalse(bq.record_is_valid(path, "abc"))
            self.assertFalse(bq.record_is_final(path, "abc"))

    def test_atomic_write_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "r.json"
            bq._atomic_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["r.json"])


class MessageTests(unittest.TestCase):
    def test_pages_precede_question_in_order(self):
        content = build_messages(3, "q")[0]["content"]
        self.assertEqual([c["type"] for c in content], ["image", "image", "image", "text"])
        with self.assertRaises(ValueError):
            build_messages(0, "q")

    def test_row_images(self):
        self.assertEqual(bq.row_images({"image": "a"}), ["a"])
        self.assertEqual(bq.row_images({"images": ["a", "b"]}), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
