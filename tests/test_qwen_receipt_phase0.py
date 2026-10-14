"""CPU-only tests for the Qwen receipt Phase-0 artifact verifier."""

from __future__ import annotations

import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import verify_qwen_receipt_phase0 as phase0


def baseline_config() -> dict:
    return {
        "run_id": "baseline_validation_v1",
        "stage": "baseline",
        "split": "validation",
        "frozen_for_test": False,
        "require_clean_git": True,
        "seed": 566,
        "output_root": "outputs/baseline",
        "datasets": ["sroie", "cord_v2", "resume_parsing_vision"],
        "models": {
            "qwen": {
                "model_id": phase0.QWEN_MODEL_ID,
                "revision": phase0.QWEN_MODEL_REVISION,
                "precision": "bfloat16",
                "max_new_tokens": 32,
                "min_pixels": 200704,
                "max_pixels": 1003520,
            },
            "donut": {
                "model_id": "donut",
                "revision": "revision",
                "precision": "float32",
                "max_length": 64,
            },
        },
        "prompts": {
            "receipt_total": {
                "id": phase0.RECEIPT_PROMPT_ID,
                "text": phase0.RECEIPT_PROMPT_TEXT,
            },
            "resume_highest_degree": {
                "id": "resume_degree_v1",
                "text": "Return one degree label.",
            },
        },
    }


def processor_settings() -> dict:
    return {
        "precision": "bfloat16",
        "min_pixels": 200704,
        "max_pixels": 1003520,
        "max_new_tokens": 32,
        "do_sample": False,
        "batch_size": 1,
    }


def receipt_prediction(
    record_id: str,
    dataset: str,
    clean: str,
    attack: str,
    *,
    correct: bool = True,
    eligible: bool = True,
) -> dict:
    return {
        "record_id": record_id,
        "experiment_key": f"qwen::{record_id}",
        "run_id": "baseline_validation_v1",
        "stage": "baseline",
        "model_key": "qwen",
        "dataset_name": dataset,
        "experiment_split": "validation",
        "model_id": phase0.QWEN_MODEL_ID,
        "model_revision": phase0.QWEN_MODEL_REVISION,
        "task": phase0.RECEIPT_TASK,
        "prompt_id": phase0.RECEIPT_PROMPT_ID,
        "prompt_text": phase0.RECEIPT_PROMPT_TEXT,
        "prompt_sha256": phase0.sha256_text(phase0.RECEIPT_PROMPT_TEXT),
        "clean_target": clean,
        "attack_target": attack,
        "attack_eligible": eligible,
        "clean_correct": correct,
        "parse_status": "ok",
        "status": "complete",
        "prediction_normalized": clean if correct else "999",
        "processor_settings": processor_settings(),
        "input_page_count": 1,
        "worker_id": 0,
        "config_sha256": "placeholder",
        "git_commit": "baseline-commit",
    }


def receipt_manifest(record: dict) -> dict:
    return {
        "record_id": record["record_id"],
        "dataset_name": record["dataset_name"],
        "experiment_split": "validation",
        "task": phase0.RECEIPT_TASK,
        "usable": True,
        "attack_eligible": record["attack_eligible"],
        "clean_target": record["clean_target"],
        "attack_target": record["attack_target"],
        "num_pages": 1,
    }


def candidate_frames():
    rows = [
        receipt_prediction("sroie/validation/valid", "sroie", "1.00", "10.00"),
        receipt_prediction(
            "sroie/validation/incorrect", "sroie", "2.00", "20.00", correct=False
        ),
        # This mirrors a known-safe condition for Phase 0: a negative source
        # row outside the clean-correct candidate pool is not artifact corruption.
        receipt_prediction(
            "sroie/validation/negative", "sroie", "-1.00", "-10.00",
            correct=False, eligible=True,
        ),
        receipt_prediction("cord_v2/validation/valid", "cord_v2", "100", "1000"),
        receipt_prediction("cord_v2/validation/zero", "cord_v2", "0", "0"),
        receipt_prediction("cord_v2/validation/equal", "cord_v2", "20", "20"),
    ]
    return pd.DataFrame(rows), pd.DataFrame([receipt_manifest(row) for row in rows])


class TargetPolicyTests(unittest.TestCase):
    def test_sroie_and_cord_use_exact_decimal_ten_times_arithmetic(self):
        sroie = phase0.evaluate_target_policy(
            record_id="s", dataset_name="sroie",
            clean_target="12.34", attack_target="123.40",
        )
        cord = phase0.evaluate_target_policy(
            record_id="c", dataset_name="cord_v2",
            clean_target="123", attack_target="1230",
        )
        self.assertTrue(sroie["eligible_under_policy"])
        self.assertTrue(cord["eligible_under_policy"])
        self.assertTrue(sroie["ten_x_exact"])

    def test_noncanonical_or_non_ten_times_target_is_rejected(self):
        with self.assertRaisesRegex(phase0.VerificationError, "not canonical"):
            phase0.evaluate_target_policy(
                record_id="s", dataset_name="sroie",
                clean_target="12.3", attack_target="123.00",
            )
        with self.assertRaisesRegex(phase0.VerificationError, "not exactly 10x"):
            phase0.evaluate_target_policy(
                record_id="c", dataset_name="cord_v2",
                clean_target="12", attack_target="121",
            )

    def test_zero_negative_and_equal_targets_are_classified(self):
        zero = phase0.evaluate_target_policy(
            record_id="zero", dataset_name="cord_v2",
            clean_target="0", attack_target="0",
        )
        negative = phase0.evaluate_target_policy(
            record_id="negative", dataset_name="sroie",
            clean_target="-1.00", attack_target="-10.00",
        )
        equal = phase0.evaluate_target_policy(
            record_id="equal", dataset_name="cord_v2",
            clean_target="20", attack_target="20",
        )
        self.assertEqual(
            zero["reasons"], ["non_positive_clean_total", "equal_target"]
        )
        self.assertEqual(negative["reasons"], ["non_positive_clean_total"])
        self.assertEqual(equal["reasons"], ["equal_target"])
        self.assertFalse(zero["eligible_under_policy"])


class CandidateAnalysisTests(unittest.TestCase):
    def test_donut_and_resume_rows_do_not_enter_receipt_pool(self):
        predictions, manifest = candidate_frames()
        donut = receipt_prediction(
            "sroie/validation/valid", "sroie", "1.00", "10.00"
        )
        donut["model_key"] = "donut"
        resume = receipt_prediction(
            "resume/validation/decoy", "resume_parsing_vision", "1", "10"
        )
        predictions = pd.concat(
            [predictions, pd.DataFrame([donut, resume])], ignore_index=True
        )

        report = phase0.analyze_receipt_candidates(
            predictions, manifest, baseline_config()
        )

        self.assertEqual(report["candidate_counts"]["pre_policy"]["total"], 4)

    def test_counts_and_exclusions_are_per_dataset_and_deterministic(self):
        predictions, manifest = candidate_frames()
        report = phase0.analyze_receipt_candidates(
            predictions, manifest, baseline_config()
        )
        self.assertEqual(
            report["coverage"]["usable_validation_manifest"],
            {"sroie": 3, "cord_v2": 3, "total": 6},
        )
        self.assertEqual(
            report["candidate_counts"]["pre_policy"],
            {"sroie": 1, "cord_v2": 3, "total": 4},
        )
        self.assertEqual(
            report["candidate_counts"]["target_valid"],
            {"sroie": 1, "cord_v2": 1, "total": 2},
        )
        self.assertEqual(
            [row["record_id"] for row in report["exclusions"]],
            ["cord_v2/validation/equal", "cord_v2/validation/zero"],
        )
        self.assertTrue(report["coverage"]["metadata_page_count_verified"])

    def test_missing_or_duplicate_qwen_receipt_row_is_rejected(self):
        predictions, manifest = candidate_frames()
        with self.assertRaisesRegex(phase0.VerificationError, "coverage differs"):
            phase0.analyze_receipt_candidates(
                predictions.iloc[:-1].copy(), manifest, baseline_config()
            )
        duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(phase0.VerificationError, "duplicate record IDs"):
            phase0.analyze_receipt_candidates(duplicate, manifest, baseline_config())

    def test_manifest_target_and_page_count_must_match_baseline(self):
        predictions, manifest = candidate_frames()
        changed = manifest.copy()
        changed.loc[changed["record_id"].eq("sroie/validation/valid"), "attack_target"] = "11.00"
        with self.assertRaisesRegex(phase0.VerificationError, "baseline/manifest mismatch"):
            phase0.analyze_receipt_candidates(predictions, changed, baseline_config())
        changed = manifest.copy()
        changed.loc[changed["record_id"].eq("sroie/validation/valid"), "num_pages"] = 2
        with self.assertRaisesRegex(phase0.VerificationError, "input_page_count"):
            phase0.analyze_receipt_candidates(predictions, changed, baseline_config())

    def test_model_prompt_and_processor_drift_are_rejected(self):
        predictions, manifest = candidate_frames()
        changed = predictions.copy()
        changed.loc[0, "model_revision"] = "wrong"
        with self.assertRaisesRegex(phase0.VerificationError, "model_revision"):
            phase0.analyze_receipt_candidates(changed, manifest, baseline_config())
        changed = predictions.copy()
        changed.loc[0, "prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(phase0.VerificationError, "prompt_sha256"):
            phase0.analyze_receipt_candidates(changed, manifest, baseline_config())
        changed = predictions.copy()
        settings = processor_settings()
        settings["do_sample"] = True
        changed.loc[0, "processor_settings"] = json.dumps(settings)
        with self.assertRaisesRegex(phase0.VerificationError, "greedy"):
            phase0.analyze_receipt_candidates(changed, manifest, baseline_config())


class ArtifactIntegrationTests(unittest.TestCase):
    def test_complete_frozen_artifact_set_produces_pass_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs/baseline/active.json"
            frozen_path = root / "outputs/baseline/baseline_validation_v1/config.json"
            predictions_path = root / "outputs/baseline/baseline_validation_v1/merged/predictions.parquet"
            merge_path = root / "outputs/baseline/baseline_validation_v1/merged/merge_manifest.json"
            summary_path = root / "outputs/baseline/baseline_validation_v1/merged/summary.json"
            ids_path = root / "outputs/baseline/baseline_validation_v1/merged/qwen_clean_correct_ids.txt"
            environment_path = root / "outputs/baseline/baseline_validation_v1/shards/worker_00/environment.json"
            manifest_path = root / "data/processed/master_manifest.parquet"
            reference_path = root / "configs/qwen_attack/resume_active.json"
            for path in (
                config_path, frozen_path, predictions_path, merge_path,
                summary_path, ids_path, environment_path, manifest_path,
                reference_path,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)

            config_text = json.dumps(baseline_config(), indent=2) + "\n"
            config_path.write_text(config_text, encoding="utf-8")
            frozen_path.write_text(config_text, encoding="utf-8")
            config_hash = phase0.sha256_file(config_path)

            predictions, manifest = candidate_frames()
            predictions["config_sha256"] = config_hash
            predictions.to_parquet(predictions_path, index=False)
            manifest.to_parquet(manifest_path, index=False)
            predictions_hash = phase0.sha256_file(predictions_path)
            reference_path.write_text(json.dumps({
                "baseline_run_id": "baseline_validation_v1",
                "baseline_config_sha256": config_hash,
                "baseline_predictions_sha256": predictions_hash,
            }), encoding="utf-8")
            merge_path.write_text(json.dumps({
                "config_sha256": config_hash,
                "git_commit": "baseline-commit",
                "records": len(predictions),
                "records_fingerprint": "a" * 64,
            }), encoding="utf-8")
            summary_path.write_text(json.dumps({
                "records": len(predictions), "expected_records": len(predictions)
            }), encoding="utf-8")
            correct_ids = sorted(
                predictions.loc[predictions["clean_correct"].eq(True), "record_id"]
            )
            ids_path.write_text(
                "".join(f"{record_id}\n" for record_id in correct_ids), encoding="utf-8"
            )
            environment_path.write_text(json.dumps({
                "python": "3.11", "torch": "2", "cuda": "12", "transformers": "4",
                "datasets": "3", "pillow": "11", "gpu": "synthetic",
                "git_commit": "baseline-commit",
            }), encoding="utf-8")

            with self.assertRaisesRegex(
                phase0.VerificationError, "must contain 527 records"
            ):
                phase0.verify_phase0(root)
            report = phase0.verify_phase0(root, enforce_accepted_counts=False)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["candidate_counts"]["target_valid"]["total"], 2)
            self.assertEqual(
                report["provenance"]["baseline_environment_scope"],
                "worker_00_representative",
            )
            self.assertIn("master_manifest", report["artifacts"])

    def test_import_does_not_load_torch(self):
        source = (ROOT / "scripts/verify_qwen_receipt_phase0.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertNotIn("torch", imported_roots)


if __name__ == "__main__":
    unittest.main()
