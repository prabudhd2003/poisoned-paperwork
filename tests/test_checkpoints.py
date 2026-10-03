import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoints import (
    BASELINE_REQUIRED_FIELDS,
    atomic_json,
    experiment_key,
    record_is_final,
)


def expected():
    return {
        "experiment_key": experiment_key("sroie/train/1", "model/id", "receipt_total", "p1", "validation"),
        "record_id": "sroie/train/1",
        "dataset_name": "sroie",
        "experiment_split": "validation",
        "model_key": "qwen",
        "model_id": "model/id",
        "model_revision": "revision",
        "task": "receipt_total",
        "prompt_id": "p1",
        "prompt_sha256": "prompt-hash",
    }


def record(status="complete"):
    value = {field: None for field in BASELINE_REQUIRED_FIELDS}
    value.update(expected())
    value.update({
        "run_id": "run",
        "stage": "baseline",
        "config_sha256": "config-hash",
        "status": status,
        "clean_correct": False,
    })
    return value


class CheckpointTests(unittest.TestCase):
    def test_key_changes_with_any_experimental_factor(self):
        keys = {
            experiment_key("a", "m", "t", "p", "validation"),
            experiment_key("b", "m", "t", "p", "validation"),
            experiment_key("a", "other", "t", "p", "validation"),
            experiment_key("a", "m", "t", "other", "validation"),
            experiment_key("a", "m", "t", "p", "test"),
        }
        self.assertEqual(len(keys), 5)

    def test_atomic_write_and_strict_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records" / "record.json"
            atomic_json(path, record())
            self.assertTrue(record_is_final(
                path, expected=expected(), config_sha256="config-hash"
            ))
            self.assertEqual(list(path.parent.glob("*.tmp.*")), [])
            self.assertFalse(record_is_final(
                path, expected=expected(), config_sha256="other"
            ))
            stale = json.loads(path.read_text())
            stale["model_revision"] = "other"
            path.write_text(json.dumps(stale))
            self.assertFalse(record_is_final(
                path, expected=expected(), config_sha256="config-hash"
            ))

    def test_explicit_failure_is_final_but_corrupt_json_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            atomic_json(path, record(status="failed"))
            self.assertTrue(record_is_final(
                path, expected=expected(), config_sha256="config-hash"
            ))
            path.write_text("{broken")
            self.assertFalse(record_is_final(
                path, expected=expected(), config_sha256="config-hash"
            ))


if __name__ == "__main__":
    unittest.main()
