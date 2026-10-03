import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from checkpoints import BASELINE_REQUIRED_FIELDS, sha256_bytes
from data import select_baseline_manifest
from stages import baseline


def smoke_config():
    config = json.loads(
        (REPO_ROOT / "configs" / "baseline" / "smoke.json").read_text()
    )
    config["run_id"] = "merge_test"
    return config


def manifest_frame():
    rows = []
    definitions = (
        ("sroie", "receipt_total", "10.00"),
        ("cord_v2", "receipt_total", "10000"),
        ("resume_parsing_vision", "resume_highest_degree", "bachelor"),
    )
    for dataset_name, task, target in definitions:
        for index in range(2):
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


class MergeTests(unittest.TestCase):
    def _prepared_run(self, root: Path):
        config = smoke_config()
        config_path = root / "configs" / "baseline" / "smoke.json"
        config_path.parent.mkdir(parents=True)
        config_text = json.dumps(config, indent=2) + "\n"
        config_path.write_text(config_text)
        manifest_path = root / "data" / "processed" / "master_manifest.parquet"
        manifest_path.parent.mkdir(parents=True)
        manifest_frame().to_parquet(manifest_path, index=False)

        manifest = select_baseline_manifest(manifest_frame(), config)
        experiments = baseline.build_experiments(manifest, config)
        output = root / "outputs" / "baseline" / config["run_id"]
        (output / "assignments").mkdir(parents=True)
        (output / "shards" / "worker_00" / "records").mkdir(parents=True)
        (output / "config.json").write_text(config_text)
        (output / "assignments" / "worker_00.txt").write_text(
            "".join(f"{record_id}\n" for record_id in sorted(manifest.index))
        )
        config_hash = sha256_bytes(config_text.encode())
        for key, expected in experiments.items():
            record = {field: None for field in BASELINE_REQUIRED_FIELDS}
            record.update(expected)
            record.update({
                "run_id": config["run_id"],
                "stage": "baseline",
                "clean_target": "target",
                "attack_target": "other",
                "attack_eligible": True,
                "raw_response": "target",
                "prediction_normalized": "target",
                "parse_status": "ok",
                "clean_correct": True,
                "runtime_seconds": 1.0,
                "peak_gpu_memory_mb": 1.0,
                "worker_id": 0,
                "seed": 566,
                "git_commit": "commit-a",
                "config_sha256": config_hash,
                "input_page_count": 1,
                "status": "complete",
                "error": None,
            })
            (output / "shards" / "worker_00" / "records" / f"{key}.json").write_text(
                json.dumps(record)
            )
        marker = {
            "run_id": config["run_id"],
            "worker_id": 0,
            "expected_records": len(experiments),
            "config_sha256": config_hash,
            "git_commit": "commit-a",
        }
        (output / "shards" / "worker_00" / "WORKER_COMPLETE.json").write_text(
            json.dumps(marker)
        )
        return config, config_path, output

    def test_exact_smoke_merge_writes_ten_rows_and_model_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path, output = self._prepared_run(Path(directory))
            self.assertTrue(baseline.merge_if_complete(
                config, config_path, num_workers=1
            ))
            predictions = pd.read_parquet(output / "merged" / "predictions.parquet")
            self.assertEqual(len(predictions), 10)
            self.assertEqual((predictions["model_key"] == "qwen").sum(), 6)
            self.assertEqual((predictions["model_key"] == "donut").sum(), 4)
            self.assertTrue((output / "merged" / "qwen_clean_correct_ids.txt").is_file())
            self.assertTrue((output / "merged" / "donut_clean_correct_ids.txt").is_file())
            self.assertTrue(baseline.merge_if_complete(
                config, config_path, num_workers=1
            ))

    def test_missing_marker_does_not_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            config, config_path, output = self._prepared_run(Path(directory))
            (output / "shards" / "worker_00" / "WORKER_COMPLETE.json").unlink()
            self.assertFalse(baseline.merge_if_complete(
                config, config_path, num_workers=1
            ))


if __name__ == "__main__":
    unittest.main()
