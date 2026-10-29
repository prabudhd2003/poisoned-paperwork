"""CPU-only contract tests for the Qwen receipt attack track."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoints import sha256_bytes, sha256_file, sha256_text
from sharding import assigned_record_ids
from stages import qwen_attack, qwen_attack_receipt


PROMPT = (
    "What is the final total amount shown on this receipt? Return only the "
    "amount, with no currency symbol, label, or explanation."
)
MODEL = {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "precision": "bfloat16",
    "min_pixels": 200704,
    "max_pixels": 1003520,
    "max_new_tokens": 32,
}
TEST_RUNTIME_VERSIONS = {
    "python": "3.11.12",
    "torch": "2.6.0+cu124",
    "transformers": "5.17.0",
    "datasets": "5.0.1",
    "pillow": "12.3.0",
    "numpy": "2.4.6",
    "lpips": "0.1.4",
    "scikit_image": "0.26.0",
}
TEST_GPU_RUNTIME_CONTRACT = {
    "python": "3.11.12",
    "cuda": "12.4",
    "gpu": "NVIDIA A100-SXM4-40GB",
    "gpu_memory_bytes": 42949672960,
}
TEST_SOURCE_EVIDENCE = {
    "schema_version": 1,
    "verification": "qwen_receipt_source_pixels_v1",
    "scope": "all_validation_candidates",
    "records": 219,
    "candidate_counts": {"sroie": 123, "cord_v2": 96},
    "source_records_sha256": (
        "77480634c85f37271c1927798bd975858877da6c1ebb06442c549730d58da277"
    ),
    "accepted_report_sha256": (
        "7682d9fc5f5601fa1cc9d4dd368b75d9df44950772185d8a9a9a85cb61cf609d"
    ),
    "verified_git_commit": "fa5934fd6d3a1345da1fd0130998c297ace2e61a",
}


def _tracked_config(name: str) -> dict:
    return json.loads((ROOT / "configs/qwen_attack" / name).read_text(encoding="utf-8"))


def _baseline() -> dict:
    return {
        "run_id": "baseline_validation_v1",
        "split": "validation",
        "models": {"qwen": dict(MODEL)},
        "prompts": {"receipt_total": {"id": "receipt_total_v1", "text": PROMPT}},
    }


def _prediction(
    dataset: str,
    index: int,
    *,
    correct: bool = True,
    page_count: int = 1,
    attack_target: str | None = None,
) -> dict:
    clean_target = f"{index + 1}.25" if dataset == "sroie" else str((index + 1) * 100)
    if attack_target is None:
        attack_target = (
            f"{Decimal(clean_target) * Decimal(10):.2f}"
            if dataset == "sroie"
            else str(int(clean_target) * 10)
        )
    document_id = f"{dataset}-{index:03d}"
    record_id = f"{dataset}/validation/{document_id}"
    normalized = clean_target if correct else ("999.00" if dataset == "sroie" else "999")
    return {
        "record_id": record_id,
        "run_id": "baseline_validation_v1",
        "stage": "baseline",
        "model_key": "qwen",
        "dataset_name": dataset,
        "experiment_split": "validation",
        "model_id": MODEL["model_id"],
        "model_revision": MODEL["revision"],
        "task": "receipt_total",
        "prompt_id": "receipt_total_v1",
        "prompt_text": PROMPT,
        "prompt_sha256": sha256_text(PROMPT),
        "clean_target": clean_target,
        "attack_target": attack_target,
        "attack_eligible": True,
        "clean_correct": correct,
        "parse_status": "ok",
        "status": "complete",
        "prediction_normalized": normalized,
        "processor_settings": {
            "precision": MODEL["precision"],
            "min_pixels": MODEL["min_pixels"],
            "max_pixels": MODEL["max_pixels"],
            "max_new_tokens": MODEL["max_new_tokens"],
            "do_sample": False,
            "batch_size": 1,
        },
        "raw_response": normalized,
        "input_page_count": page_count,
        "source_split": "validation",
        "source_index": index,
    }


def _manifest_row(prediction: dict, *, page_count: int = 1) -> dict:
    return {
        "record_id": prediction["record_id"],
        "dataset_name": prediction["dataset_name"],
        "experiment_split": prediction["experiment_split"],
        "task": prediction["task"],
        "usable": True,
        "attack_eligible": prediction["attack_eligible"],
        "clean_target": prediction["clean_target"],
        "attack_target": prediction["attack_target"],
        "num_pages": page_count,
    }


def _candidate_frames(per_dataset: int = 6):
    records = [
        _prediction(dataset, index)
        for dataset in reversed(qwen_attack_receipt.RECEIPT_DATASETS)
        for index in reversed(range(per_dataset))
    ]
    return pd.DataFrame(records), pd.DataFrame([_manifest_row(record) for record in records])


def _processed_rows(candidates: pd.DataFrame) -> dict[str, dict]:
    rows = {}
    for index, (record_id, row) in enumerate(candidates.sort_index().iterrows()):
        rows[record_id] = {
            "clean_target": row.clean_target,
            "attack_target": row.attack_target,
            "usable": True,
            "attack_eligible": True,
            "source_split": row.source_split,
            "source_index": int(row.source_index),
            "image": Image.new(
                "RGB", (12 + index, 8 + (index % 2)), color=(index, 2, 3)
            ),
        }
    return rows


def _synthetic_source_evidence(
    config: dict, candidates: pd.DataFrame, rows: dict[str, dict]
) -> dict:
    records = qwen_attack_receipt._build_source_records(config, candidates, rows)
    return {
        **TEST_SOURCE_EVIDENCE,
        "records": len(records),
        "candidate_counts": {
            dataset: int(candidates.dataset_name.eq(dataset).sum())
            for dataset in qwen_attack_receipt.RECEIPT_DATASETS
        },
        "source_records_sha256": qwen_attack_receipt._source_records_sha256(records),
        "accepted_report_sha256": "f" * 64,
        "verified_git_commit": "a" * 40,
    }


class ReceiptConfigTests(unittest.TestCase):
    def test_tracked_smoke_and_enabled_validation_configs(self):
        smoke = _tracked_config("receipt_smoke_saaketh.json")
        qwen_attack_receipt._validate_attack_config(smoke)
        self.assertIs(smoke["ready_for_gpu_smoke"], True)
        self.assertEqual(smoke["source_evidence"], TEST_SOURCE_EVIDENCE)
        self.assertEqual(qwen_attack_receipt.EXPECTED_SOURCE_EVIDENCE, TEST_SOURCE_EVIDENCE)
        self.assertEqual(smoke["datasets"], ["sroie", "cord_v2"])
        self.assertEqual(smoke["smoke_per_dataset"], 5)
        self.assertEqual(smoke["attack"]["epsilons_255"], [8])
        self.assertEqual(smoke["attack"]["step_size_255"], {"8": 2.0})
        self.assertEqual(smoke["attack"]["max_steps"], 25)
        self.assertEqual(smoke["attack"]["checkpoint_interval"], 10)
        self.assertEqual(smoke["shared_artifact_root_env"], "PP_SHARED_ARTIFACT_ROOT")
        self.assertEqual(smoke["output_root_env"], "PP_RECEIPT_OUTPUT_ROOT")

        active = _tracked_config("receipt_active.json")
        self.assertEqual(active["source_evidence"], TEST_SOURCE_EVIDENCE)
        self.assertIs(active["ready_for_production"], True)
        qwen_attack_receipt._validate_attack_config(active)

        gated = copy.deepcopy(active)
        gated["ready_for_production"] = False
        with self.assertRaisesRegex(ValueError, "production is gated"):
            qwen_attack_receipt._validate_attack_config(gated)
        self.assertEqual(active["expected_validation_candidates"], {"sroie": 123, "cord_v2": 96})
        self.assertEqual(active["attack"]["epsilons_255"], [2, 4, 8, 16])
        self.assertEqual(
            active["attack"]["step_size_255"],
            {"2": 0.5, "4": 1.0, "8": 2.0, "16": 4.0},
        )

    def test_config_rejects_missing_or_mutated_source_evidence_contract(self):
        base = _tracked_config("receipt_smoke_saaketh.json")
        missing = copy.deepcopy(base)
        del missing["source_evidence"]
        with self.assertRaisesRegex(ValueError, "source_evidence"):
            qwen_attack_receipt._validate_attack_config(missing)

        mutations = {
            "schema_version": 2,
            "verification": "different_verifier",
            "scope": "smoke_subset_only",
            "records": 10,
            "candidate_counts": {"sroie": 5, "cord_v2": 5},
            "source_records_sha256": "0" * 64,
            "accepted_report_sha256": "1" * 64,
            "verified_git_commit": "2" * 40,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(base)
                changed["source_evidence"][field] = value
                with self.assertRaisesRegex(ValueError, "source_evidence"):
                    qwen_attack_receipt._validate_attack_config(changed)

        for field in TEST_SOURCE_EVIDENCE:
            with self.subTest(missing_nested_field=field):
                changed = copy.deepcopy(base)
                del changed["source_evidence"][field]
                with self.assertRaisesRegex(ValueError, "source_evidence"):
                    qwen_attack_receipt._validate_attack_config(changed)

    def test_config_requires_the_frozen_receipt_output_environment_name(self):
        base = _tracked_config("receipt_smoke_saaketh.json")
        missing = copy.deepcopy(base)
        del missing["output_root_env"]
        with self.assertRaisesRegex(ValueError, "output_root_env"):
            qwen_attack_receipt._validate_attack_config(missing)

        changed = copy.deepcopy(base)
        changed["output_root_env"] = "A_DIFFERENT_OUTPUT_ROOT"
        with self.assertRaisesRegex(ValueError, "PP_RECEIPT_OUTPUT_ROOT"):
            qwen_attack_receipt._validate_attack_config(changed)

    def test_config_rejects_dataset_model_prompt_policy_and_attack_drift(self):
        base = _tracked_config("receipt_smoke_saaketh.json")
        mutations = [
            ("datasets", ["cord_v2", "sroie"], "datasets must be exactly"),
            ("model_id", "different/model", "pinned Qwen"),
            ("model_revision", "0" * 40, "pinned Qwen"),
            ("prompt_id", "different_prompt", "prompt_id"),
            ("question", "Different receipt question", "prompt"),
        ]
        for field, value, message in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(base)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    qwen_attack_receipt._validate_attack_config(changed)

        changed = copy.deepcopy(base)
        changed["target_policy"]["decimal_arithmetic"] = False
        with self.assertRaisesRegex(ValueError, "target_policy"):
            qwen_attack_receipt._validate_attack_config(changed)
        changed = copy.deepcopy(base)
        changed["attack"]["multi_page_fallback"] = "gradient_accumulation_across_pages"
        with self.assertRaisesRegex(ValueError, "one-page"):
            qwen_attack_receipt._validate_attack_config(changed)
        changed = copy.deepcopy(base)
        changed["baseline_summary_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ValueError, "baseline_summary_sha256"):
            qwen_attack_receipt._validate_attack_config(changed)
        changed = copy.deepcopy(base)
        changed["run_id"] = "../../escape"
        with self.assertRaisesRegex(ValueError, "filesystem-safe"):
            qwen_attack_receipt._validate_attack_config(changed)
        changed = copy.deepcopy(base)
        changed["processed_data_root"] = "../outside"
        with self.assertRaisesRegex(ValueError, "artifact root"):
            qwen_attack_receipt._validate_attack_config(changed)

    def test_phase1_explicitly_rejects_unfrozen_test_mode(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        config.update(mode="test", split="test", frozen_for_test=True)
        with self.assertRaisesRegex(ValueError, "later phase"):
            qwen_attack_receipt._validate_attack_config(config)


class ReceiptTargetPolicyTests(unittest.TestCase):
    def test_exact_decimal_ten_times_for_both_datasets(self):
        sroie = qwen_attack_receipt.evaluate_target_policy(
            record_id="sroie/r", dataset_name="sroie",
            clean_target="12.34", attack_target="123.40",
        )
        cord = qwen_attack_receipt.evaluate_target_policy(
            record_id="cord/r", dataset_name="cord_v2",
            clean_target="123", attack_target="1230",
        )
        self.assertEqual(sroie["attack_value"], Decimal("123.40"))
        self.assertEqual(cord["attack_value"], Decimal("1230"))

    def test_noncanonical_nonpositive_equal_and_non_ten_times_targets_fail(self):
        cases = [
            ("sroie", "12.3", "123.00", "not canonical"),
            ("cord_v2", "1,000", "10000", "not canonical"),
            ("sroie", "0.00", "0.00", "positive"),
            ("sroie", "1.00", "1.00", "differ"),
            ("cord_v2", "100", "1001", "exactly 10x"),
        ]
        for dataset, clean, target, message in cases:
            with self.subTest(dataset=dataset, clean=clean, target=target):
                with self.assertRaisesRegex(ValueError, message):
                    qwen_attack_receipt.evaluate_target_policy(
                        record_id=f"{dataset}/r", dataset_name=dataset,
                        clean_target=clean, attack_target=target,
                    )

    def test_dataset_aware_output_normalization(self):
        sroie = qwen_attack_receipt._receipt_normalizer("sroie")
        cord = qwen_attack_receipt._receipt_normalizer("cord_v2")
        self.assertEqual(sroie("Total: $12.3"), "12.30")
        self.assertEqual(cord("Rp 45.500"), "45500")
        self.assertIsNone(cord("10 or 20"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            qwen_attack_receipt._receipt_normalizer("resume_parsing_vision")


class ReceiptCandidateTests(unittest.TestCase):
    def test_coverage_candidate_filter_and_one_page_contract(self):
        predictions, manifest = _candidate_frames(2)
        predictions.loc[predictions.index[-1], "clean_correct"] = False
        predictions.loc[predictions.index[-1], "prediction_normalized"] = "999.00"
        selected = qwen_attack_receipt.select_candidates(predictions, manifest, _baseline())
        self.assertEqual(len(selected), 3)
        self.assertEqual(list(selected.index), sorted(selected.index))
        self.assertNotIn(predictions.iloc[-1].record_id, selected.index)

        missing = manifest.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "IDs do not equal"):
            qwen_attack_receipt.select_candidates(predictions, missing, _baseline())

        bad_pages = predictions.copy()
        bad_pages.loc[bad_pages.index[0], "input_page_count"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one page"):
            qwen_attack_receipt.select_candidates(bad_pages, manifest, _baseline())

    def test_candidate_rejects_target_and_processor_drift(self):
        predictions, manifest = _candidate_frames(1)
        changed = predictions.copy()
        wrong_target = "99.00" if changed.loc[changed.index[0], "dataset_name"] == "sroie" else "999"
        changed.loc[changed.index[0], "attack_target"] = wrong_target
        changed_manifest = manifest.copy()
        changed_manifest.loc[changed_manifest.index[0], "attack_target"] = wrong_target
        with self.assertRaisesRegex(ValueError, "exactly 10x"):
            qwen_attack_receipt.select_candidates(changed, changed_manifest, _baseline())

        changed = predictions.copy()
        settings = dict(changed.loc[changed.index[0], "processor_settings"])
        settings["do_sample"] = True
        changed.at[changed.index[0], "processor_settings"] = settings
        with self.assertRaisesRegex(ValueError, "greedy"):
            qwen_attack_receipt.select_candidates(changed, manifest, _baseline())

    def test_smoke_selection_is_deterministic_five_per_dataset(self):
        predictions, manifest = _candidate_frames(6)
        candidates = qwen_attack_receipt.select_candidates(predictions, manifest, _baseline())
        first = qwen_attack_receipt.select_smoke_candidates(candidates, 5)
        second = qwen_attack_receipt.select_smoke_candidates(candidates.sample(frac=1, random_state=566), 5)
        self.assertEqual(list(first.index), list(second.index))
        self.assertEqual(first.groupby("dataset_name").size().to_dict(), {"cord_v2": 5, "sroie": 5})
        self.assertNotIn("sroie/validation/sroie-005", first.index)
        self.assertNotIn("cord_v2/validation/cord_v2-005", first.index)

        without_cord = candidates.loc[candidates.dataset_name.eq("sroie")]
        with self.assertRaisesRegex(ValueError, "smoke needs"):
            qwen_attack_receipt.select_smoke_candidates(without_cord, 5)


class ReceiptPathRouterAndShardingTests(unittest.TestCase):
    def test_shared_output_root_resolves_one_run_path(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            output = temporary / "receipt-output"
            for path in (repo, shared, output):
                path.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {"PP_RECEIPT_OUTPUT_ROOT": str(output)},
                    clear=False,
                ),
                patch.object(qwen_attack_receipt, "_require_shared_output_mode"),
            ):
                run_root = qwen_attack_receipt.resolve_output_root(
                    config, repo, shared_root=shared
                )
            self.assertEqual(
                run_root,
                (output / "qwen_attack" / config["run_id"]).resolve(),
            )
            self.assertTrue(run_root.is_relative_to(output.resolve()))

    def test_shared_output_root_rejects_missing_relative_and_nonexistent_values(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            repo.mkdir()
            shared.mkdir()

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "PP_RECEIPT_OUTPUT_ROOT"):
                    qwen_attack_receipt.resolve_output_root(
                        config, repo, shared_root=shared
                    )

            invalid_values = (
                ("relative/output", "absolute"),
                (str(temporary / "missing"), "does not exist"),
            )
            for value, message in invalid_values:
                with self.subTest(value=value), patch.dict(
                    os.environ,
                    {"PP_RECEIPT_OUTPUT_ROOT": value},
                    clear=False,
                ):
                    with self.assertRaisesRegex(
                        (RuntimeError, ValueError, FileNotFoundError), message
                    ):
                        qwen_attack_receipt.resolve_output_root(
                            config, repo, shared_root=shared
                        )

    def test_shared_output_root_rejects_code_input_and_symlink_overlap(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            repo.mkdir()
            shared.mkdir()
            repo_child = repo / "output"
            shared_child = shared / "output"
            repo_child.mkdir()
            shared_child.mkdir()
            candidates = (repo, repo_child, shared, shared_child, temporary)
            for candidate in candidates:
                with self.subTest(candidate=candidate), patch.dict(
                    os.environ,
                    {"PP_RECEIPT_OUTPUT_ROOT": str(candidate)},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "separate"):
                        qwen_attack_receipt.resolve_output_root(
                            config, repo, shared_root=shared
                        )

            alias = temporary / "output-alias"
            alias.symlink_to(repo, target_is_directory=True)
            with patch.dict(
                os.environ,
                {"PP_RECEIPT_OUTPUT_ROOT": str(alias)},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "separate"):
                    qwen_attack_receipt.resolve_output_root(
                        config, repo, shared_root=shared
                    )

    def test_shared_output_run_path_rejects_symlink_escape(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            output = temporary / "receipt-output"
            escape = temporary / "escape"
            for path in (repo, shared, output, escape):
                path.mkdir()
            (output / "qwen_attack").symlink_to(escape, target_is_directory=True)
            with patch.dict(
                os.environ,
                {"PP_RECEIPT_OUTPUT_ROOT": str(output)},
                clear=False,
            ), patch.object(qwen_attack_receipt, "_require_shared_output_mode"):
                with self.assertRaisesRegex(ValueError, "escapes"):
                    qwen_attack_receipt.resolve_output_root(
                        config, repo, shared_root=shared
                    )

    def test_shared_output_root_requires_group_write_search_and_setgid(self):
        path = Path("/shared/receipt-output")
        qwen_attack_receipt._require_shared_output_mode(0o2770, path)
        for mode, message in (
            (0o2750, "group-writable"),
            (0o2760, "group-writable/searchable"),
            (0o0770, "setgid"),
        ):
            with self.subTest(mode=oct(mode)), self.assertRaisesRegex(
                PermissionError, message
            ):
                qwen_attack_receipt._require_shared_output_mode(mode, path)

    def test_shared_output_root_rejects_nonwritable_directory(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            output = temporary / "receipt-output"
            for path in (repo, shared, output):
                path.mkdir()
            with (
                patch.dict(
                    os.environ,
                    {"PP_RECEIPT_OUTPUT_ROOT": str(output)},
                    clear=False,
                ),
                patch.object(
                    qwen_attack_receipt.os,
                    "access",
                    side_effect=lambda path, mode: False
                    if Path(path).resolve() == output.resolve()
                    else True,
                ),
                self.assertRaisesRegex((RuntimeError, PermissionError), "writable"),
            ):
                qwen_attack_receipt.resolve_output_root(
                    config, repo, shared_root=shared
                )

    def test_shared_input_root_and_local_config_resolution_are_split(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            repo = temporary / "personal"
            shared = temporary / "shared"
            local_config = repo / "configs/baseline/active.json"
            frozen_config = shared / "outputs/baseline/run/config.json"
            local_config.parent.mkdir(parents=True)
            frozen_config.parent.mkdir(parents=True)
            content = '{"frozen": true}\n'
            local_config.write_text(content, encoding="utf-8")
            frozen_config.write_text(content, encoding="utf-8")
            config = {
                "shared_artifact_root_env": "TEST_SHARED_RECEIPTS",
                "baseline_config": "configs/baseline/active.json",
                "baseline_frozen_config": "outputs/baseline/run/config.json",
                "baseline_config_sha256": sha256_file(local_config),
            }
            with patch.dict(os.environ, {"TEST_SHARED_RECEIPTS": str(shared)}, clear=False):
                self.assertEqual(qwen_attack_receipt._shared_artifact_root(config), shared.resolve())
                selected = qwen_attack_receipt._resolve_baseline_config(config, repo, shared)
            self.assertEqual(selected, local_config.resolve())
            self.assertNotEqual(selected.parent, frozen_config.parent)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TEST_SHARED_RECEIPTS"):
                qwen_attack_receipt._shared_artifact_root(config)

    def test_cpu_preflight_loads_and_validates_real_image_rows(self):
        predictions, manifest = _candidate_frames(1)
        candidates = qwen_attack_receipt.select_candidates(
            predictions, manifest, _baseline()
        )
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary = Path(temporary_directory.name)
        shared = temporary / "shared"
        output = temporary / "output"
        shared.mkdir()
        output.mkdir()
        rows = {}
        for record_id, row in candidates.iterrows():
            rows[record_id] = {
                "clean_target": row.clean_target,
                "attack_target": row.attack_target,
                "usable": True,
                "attack_eligible": True,
                "image": Image.new("RGB", (12, 8), color=(1, 2, 3)),
            }
        paths = {
            "processed_data_root": shared / "processed",
            "shared_artifact_root": shared,
        }
        config = _tracked_config("receipt_smoke_saaketh.json")
        with (
            patch.dict(
                os.environ,
                {"PP_RECEIPT_OUTPUT_ROOT": str(output)},
                clear=False,
            ),
            patch.object(
                qwen_attack_receipt,
                "_load_plan",
                return_value=(_baseline(), candidates, paths),
            ),
            patch.object(qwen_attack_receipt, "_require_shared_output_mode"),
            patch.object(
                qwen_attack_receipt, "_load_processed_rows", return_value=rows
            ) as load_rows,
        ):
            selected = qwen_attack_receipt.preflight(config, ROOT)
        self.assertEqual(set(selected.index), set(candidates.index))
        load_rows.assert_called_once_with(
            paths["processed_data_root"], "validation", set(candidates.index)
        )

        bad_rows = copy.deepcopy(rows)
        first_id = next(iter(bad_rows))
        bad_rows[first_id]["image"] = None
        bad_rows[first_id]["images"] = [
            Image.new("RGB", (2, 2)),
            Image.new("RGB", (2, 2)),
        ]
        with (
            patch.dict(
                os.environ,
                {"PP_RECEIPT_OUTPUT_ROOT": str(output)},
                clear=False,
            ),
            patch.object(
                qwen_attack_receipt,
                "_load_plan",
                return_value=(_baseline(), candidates, paths),
            ),
            patch.object(qwen_attack_receipt, "_require_shared_output_mode"),
            patch.object(
                qwen_attack_receipt, "_load_processed_rows", return_value=bad_rows
            ),
            self.assertRaisesRegex(RuntimeError, "exactly one page"),
        ):
            qwen_attack_receipt.preflight(config, ROOT)

    def test_source_evidence_covers_untruncated_candidates_and_rgb_hashes(self):
        predictions, manifest = _candidate_frames(1)
        candidates = qwen_attack_receipt.select_candidates(
            predictions, manifest, _baseline()
        )
        rows = _processed_rows(candidates)
        config = _tracked_config("receipt_smoke_saaketh.json")
        config["expected_validation_candidates"] = {"sroie": 1, "cord_v2": 1}
        source_evidence = _synthetic_source_evidence(config, candidates, rows)
        config["source_evidence"] = source_evidence
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs/qwen_attack/receipt_smoke_saaketh.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with (
                patch.object(qwen_attack_receipt, "_require_clean_git"),
                patch.object(
                    qwen_attack_receipt,
                    "_load_plan",
                    return_value=(_baseline(), candidates, {"processed_data_root": root}),
                ) as load_plan,
                patch.object(qwen_attack_receipt, "_load_processed_rows", return_value=rows),
                patch.object(
                    qwen_attack_receipt,
                    "EXPECTED_VALIDATION_CANDIDATES",
                    {"sroie": 1, "cord_v2": 1},
                ),
                patch.object(
                    qwen_attack_receipt,
                    "EXPECTED_SOURCE_EVIDENCE",
                    source_evidence,
                ),
                patch.object(
                    qwen_attack_receipt.subprocess,
                    "check_output",
                    side_effect=lambda command, **_: (
                        "a" * 40 + "\n" if "rev-parse" in command else "qwen-receipt\n"
                    ),
                ),
            ):
                report = qwen_attack_receipt.build_source_evidence(config, config_path)
        load_plan.assert_called_once_with(
            config,
            root.resolve(),
            apply_smoke_selection=False,
            verify_source_evidence=False,
        )
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["candidate_counts"], {"sroie": 1, "cord_v2": 1})
        self.assertEqual(
            {record["record_id"] for record in report["source_records"]},
            set(candidates.index),
        )
        for record in report["source_records"]:
            self.assertEqual(record["mode"], "RGB")
            self.assertEqual(len(record["rgb_sha256"]), 64)
        self.assertEqual(
            report["source_records_sha256"],
            source_evidence["source_records_sha256"],
        )

    def test_bound_source_evidence_uses_full_pool_not_smoke_subset(self):
        predictions, manifest = _candidate_frames(6)
        candidates = qwen_attack_receipt.select_candidates(
            predictions, manifest, _baseline()
        )
        rows = _processed_rows(candidates)
        config = _tracked_config("receipt_smoke_saaketh.json")
        expected_counts = {"sroie": 6, "cord_v2": 6}
        config["expected_validation_candidates"] = expected_counts
        source_evidence = _synthetic_source_evidence(config, candidates, rows)
        config["source_evidence"] = source_evidence
        paths = {"processed_data_root": Path("/read-only/processed")}

        with (
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_VALIDATION_CANDIDATES",
                expected_counts,
            ),
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_SOURCE_EVIDENCE",
                source_evidence,
            ),
            patch.object(
                qwen_attack_receipt,
                "_load_processed_rows",
                return_value=rows,
            ) as load_rows,
        ):
            verified = qwen_attack_receipt._verify_source_evidence(
                config, candidates, paths
            )

        smoke = qwen_attack_receipt.select_smoke_candidates(candidates, 5)
        self.assertEqual(len(smoke), 10)
        self.assertEqual(verified["records"], 12)
        self.assertEqual(verified["candidate_counts"], expected_counts)
        self.assertEqual(
            verified["source_records_sha256"],
            source_evidence["source_records_sha256"],
        )
        self.assertEqual(
            qwen_attack_receipt._source_records_sha256(
                list(reversed(verified["source_records"]))
            ),
            source_evidence["source_records_sha256"],
        )
        load_rows.assert_called_once_with(
            paths["processed_data_root"], "validation", set(candidates.index)
        )

    def test_bound_source_evidence_rejects_truncation_pixel_and_dimension_drift(self):
        predictions, manifest = _candidate_frames(6)
        candidates = qwen_attack_receipt.select_candidates(
            predictions, manifest, _baseline()
        )
        rows = _processed_rows(candidates)
        config = _tracked_config("receipt_smoke_saaketh.json")
        expected_counts = {"sroie": 6, "cord_v2": 6}
        config["expected_validation_candidates"] = expected_counts
        source_evidence = _synthetic_source_evidence(config, candidates, rows)
        config["source_evidence"] = source_evidence
        paths = {"processed_data_root": Path("/read-only/processed")}

        truncated = qwen_attack_receipt.select_smoke_candidates(candidates, 5)
        drifted_rows = copy.deepcopy(rows)
        changed_id = sorted(drifted_rows)[0]
        original = drifted_rows[changed_id]["image"]
        changed = original.copy()
        changed.putpixel((0, 0), (255, 255, 255))
        drifted_rows[changed_id]["image"] = changed

        with (
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_VALIDATION_CANDIDATES",
                expected_counts,
            ),
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_SOURCE_EVIDENCE",
                source_evidence,
            ),
            patch.object(qwen_attack_receipt, "_load_processed_rows", return_value=rows),
            self.assertRaisesRegex(RuntimeError, "source evidence"),
        ):
            qwen_attack_receipt._verify_source_evidence(config, truncated, paths)

        with (
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_VALIDATION_CANDIDATES",
                expected_counts,
            ),
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_SOURCE_EVIDENCE",
                source_evidence,
            ),
            patch.object(
                qwen_attack_receipt,
                "_load_processed_rows",
                return_value=drifted_rows,
            ),
            self.assertRaisesRegex(RuntimeError, "source evidence"),
        ):
            qwen_attack_receipt._verify_source_evidence(config, candidates, paths)

        resized_rows = copy.deepcopy(rows)
        resized_rows[changed_id]["image"] = Image.new(
            "RGB",
            (original.width + 1, original.height),
            color=(0, 2, 3),
        )
        with (
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_VALIDATION_CANDIDATES",
                expected_counts,
            ),
            patch.object(
                qwen_attack_receipt,
                "EXPECTED_SOURCE_EVIDENCE",
                source_evidence,
            ),
            patch.object(
                qwen_attack_receipt,
                "_load_processed_rows",
                return_value=resized_rows,
            ),
            self.assertRaisesRegex(RuntimeError, "source evidence"),
        ):
            qwen_attack_receipt._verify_source_evidence(config, candidates, paths)

    def test_source_record_builder_rejects_missing_or_mutated_source_rows(self):
        predictions, manifest = _candidate_frames(1)
        candidates = qwen_attack_receipt.select_candidates(
            predictions, manifest, _baseline()
        )
        rows = _processed_rows(candidates)
        config = _tracked_config("receipt_smoke_saaketh.json")

        missing_rows = copy.deepcopy(rows)
        del missing_rows[sorted(missing_rows)[0]]
        with self.assertRaisesRegex(RuntimeError, "source evidence"):
            qwen_attack_receipt._build_source_records(
                config, candidates, missing_rows
            )

        record_id = sorted(rows)[0]
        mutations = {
            "clean_target": "999.99",
            "attack_target": "9999.90",
            "usable": False,
            "attack_eligible": False,
            "source_split": "test",
            "source_index": 999,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(rows)
                changed[record_id][field] = value
                with self.assertRaisesRegex(RuntimeError, "source evidence"):
                    qwen_attack_receipt._build_source_records(
                        config, candidates, changed
                    )

    def test_source_evidence_mismatch_stops_before_qwen_creation(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True)
        )
        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "test-job"}, clear=False),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(
                qwen_attack_receipt,
                "_load_plan",
                side_effect=RuntimeError("source evidence digest mismatch"),
            ),
            patch.object(qwen_attack_receipt, "QwenVL") as qwen,
            self.assertRaisesRegex(RuntimeError, "source evidence"),
        ):
            qwen_attack_receipt.run(
                config,
                ROOT / "configs/qwen_attack/receipt_smoke_saaketh.json",
                user_id="user3",
                worker_id=0,
                num_workers=1,
            )
        qwen.assert_not_called()

    def test_router_smoke_counts_and_five_way_shards(self):
        track, module = qwen_attack._validate({
            "stage": "qwen_attack", "track": "receipt",
            "datasets": ["sroie", "cord_v2"],
        })
        self.assertEqual((track, module), ("receipt", "stages.qwen_attack_receipt"))
        with self.assertRaisesRegex(ValueError, "accepts datasets"):
            qwen_attack._validate({
                "stage": "qwen_attack", "track": "receipt",
                "datasets": ["resume_parsing_vision"],
            })

        spec = importlib.util.spec_from_file_location(
            "smoke_qwen_attack_for_test", ROOT / "scripts/smoke_qwen_attack.py"
        )
        smoke_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(smoke_module)
        self.assertEqual(smoke_module._expected_smoke_documents({
            "track": "receipt", "datasets": ["sroie", "cord_v2"],
            "smoke_per_dataset": 5,
        }), 10)
        self.assertEqual(smoke_module._expected_smoke_documents({
            "track": "resume", "datasets": ["resume_parsing_vision"],
            "smoke_per_page_group": 5,
        }), 10)
        self.assertNotEqual(
            qwen_attack_receipt._record_key("sroie/a/b_c", 8, 566),
            qwen_attack_receipt._record_key("sroie/a_b/c", 8, 566),
        )

        ids = [f"sroie/validation/{index:03d}" for index in range(123)]
        ids += [f"cord_v2/validation/{index:03d}" for index in range(96)]
        shards = [assigned_record_ids(ids, worker, 5) for worker in range(5)]
        self.assertEqual(sum(map(len, shards)), 219)
        self.assertEqual(set().union(*(set(shard) for shard in shards)), set(ids))
        for left in range(5):
            for right in range(left + 1, 5):
                self.assertTrue(set(shards[left]).isdisjoint(shards[right]))

    def test_processor_parity_resolver_cannot_escape_shared_root(self):
        spec = importlib.util.spec_from_file_location(
            "check_qwen_processor_parity_for_test",
            ROOT / "scripts/check_qwen_processor_parity.py",
        )
        parity_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(parity_module)
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / "shared"
            shared.mkdir()
            config = {
                "shared_artifact_root_env": "TEST_SHARED_RECEIPTS",
                "processed_data_root": "data/processed",
            }
            with patch.dict(
                os.environ, {"TEST_SHARED_RECEIPTS": str(shared)}, clear=False
            ):
                self.assertEqual(
                    parity_module._processed_data_root(config),
                    (shared / "data/processed").resolve(),
                )
                config["processed_data_root"] = "../outside"
                with self.assertRaisesRegex(SystemExit, "must stay relative"):
                    parity_module._processed_data_root(config)

    def test_clean_git_guard_applies_to_cpu_merge_entrypoint(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with patch("subprocess.check_output", return_value=" M changed.py\n"):
            with self.assertRaisesRegex(RuntimeError, "dirty Git worktree"):
                qwen_attack_receipt._require_clean_git(config, ROOT)
        config["require_clean_git"] = False
        with patch("subprocess.check_output") as status:
            qwen_attack_receipt._require_clean_git(config, ROOT)
        status.assert_not_called()

    def test_source_evidence_is_part_of_every_run_provenance(self):
        config = _tracked_config("receipt_smoke_saaketh.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "configs/qwen_attack/receipt_smoke_saaketh.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            paths = {}
            for name in (
                "baseline_manifest",
                "baseline_summary",
                "baseline_correct_ids",
                "baseline_environment",
                "master_manifest",
            ):
                path = root / f"{name}.artifact"
                path.write_text(name, encoding="utf-8")
                paths[name] = path
            with patch.object(
                qwen_attack_receipt.subprocess,
                "check_output",
                return_value="b" * 40 + "\n",
            ):
                output = root / "shared-output/qwen_attack/qwen_receipt_smoke_saaketh_v1"
                provenance = qwen_attack_receipt._build_provenance(
                    config, config_path, root, TEST_RUNTIME_VERSIONS, paths, output
                )

        self.assertEqual(provenance["source_evidence"], TEST_SOURCE_EVIDENCE)
        self.assertEqual(
            provenance["source_records_sha256"],
            TEST_SOURCE_EVIDENCE["source_records_sha256"],
        )
        self.assertEqual(
            provenance["source_evidence_report_sha256"],
            TEST_SOURCE_EVIDENCE["accepted_report_sha256"],
        )
        self.assertEqual(
            provenance["source_evidence_verified_git_commit"],
            TEST_SOURCE_EVIDENCE["verified_git_commit"],
        )
        self.assertEqual(provenance["output_root_env"], "PP_RECEIPT_OUTPUT_ROOT")
        self.assertEqual(provenance["output_base"], str(output.parents[1]))
        self.assertEqual(provenance["output_run_root"], str(output))


class ReceiptStorageAndLauncherTests(unittest.TestCase):
    @staticmethod
    def _load_storage_preflight_module():
        spec = importlib.util.spec_from_file_location(
            "preflight_qwen_receipt_storage_for_test",
            ROOT / "scripts/preflight_qwen_receipt_storage.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_storage_preflight_contract_and_five_synthetic_markers(self):
        module = self._load_storage_preflight_module()
        args = module._parser().parse_args(
            [
                "--config",
                "configs/qwen_attack/receipt_smoke_saaketh.json",
                "--user-id",
                "user3",
                "--session-id",
                "task9-20261014t230000z",
                "--expected-git-commit",
                "a" * 40,
                "--require-all",
            ]
        )
        self.assertEqual(args.user_id, "user3")
        self.assertTrue(args.require_all)
        self.assertIsNotNone(module.SESSION_PATTERN.fullmatch(args.session_id))
        self.assertIsNone(module.SESSION_PATTERN.fullmatch("Unsafe Session"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session"
            markers = session / "markers"
            probes = session / "probes"
            for path, parent in (
                (session, root),
                (markers, session),
                (probes, session),
            ):
                path.mkdir()
                path.chmod(0o2770)
                # Some local filesystems clear setgid on directories even when
                # chmod requests it. CARC verifies the real mode; this unit test
                # exercises the resolver and I/O path with the accepted mode.
                with patch.object(module.stat, "S_IMODE", return_value=0o2770):
                    module._shared_directory(path, parent=parent)

            lock_path = session / ".storage.lock"
            with lock_path.open("a+", encoding="utf-8") as lock:
                module.fcntl.flock(
                    lock, module.fcntl.LOCK_EX | module.fcntl.LOCK_NB
                )
                for worker_id in range(5):
                    marker = {
                        "schema_version": 1,
                        "verification": "qwen_receipt_shared_storage_v1",
                        "session_id": args.session_id,
                        "user_id": f"user{worker_id + 1}",
                        "worker_id": worker_id,
                        "git_commit": args.expected_git_commit,
                        "unix_uid": 1000 + worker_id,
                    }
                    marker_path = markers / f"worker_{worker_id:02d}.json"
                    module._atomic_json(marker_path, marker)
                    self.assertEqual(
                        module._verify_marker(
                            marker_path,
                            {
                                "session_id": args.session_id,
                                "worker_id": worker_id,
                                "git_commit": args.expected_git_commit,
                            },
                        )["unix_uid"],
                        1000 + worker_id,
                    )

                temporary = probes / ".rename.tmp"
                published = probes / "atomic-directory-rename-complete"
                temporary.mkdir()
                (temporary / "payload.txt").write_text("ok\n", encoding="utf-8")
                os.replace(temporary, published)
                self.assertEqual(
                    (published / "payload.txt").read_text(encoding="utf-8"),
                    "ok\n",
                )

            saved = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(markers.glob("worker_*.json"))
            ]
            self.assertEqual(len(saved), 5)
            self.assertEqual({value["worker_id"] for value in saved}, set(range(5)))
            self.assertEqual(len({value["unix_uid"] for value in saved}), 5)

            first = markers / "worker_00.json"
            with self.assertRaisesRegex(RuntimeError, "differs in worker_id"):
                module._verify_marker(first, {"worker_id": 4})

    def test_run_metadata_publication_is_globally_locked_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            record_ids = ["sroie/validation/a", "cord_v2/validation/b"]
            qwen_attack_receipt._initialize_run_metadata(
                output,
                config_text='{"run_id":"test"}\n',
                worker_id=0,
                record_ids=record_ids,
            )
            qwen_attack_receipt._initialize_run_metadata(
                output,
                config_text='{"run_id":"test"}\n',
                worker_id=0,
                record_ids=record_ids,
            )

            self.assertTrue((output / ".metadata.lock").is_file())
            self.assertEqual(
                (output / "assignments/worker_00.txt").read_text(encoding="utf-8"),
                "".join(f"{record_id}\n" for record_id in record_ids),
            )
            with self.assertRaisesRegex(RuntimeError, "metadata differs"):
                qwen_attack_receipt._initialize_run_metadata(
                    output,
                    config_text='{"run_id":"changed"}\n',
                    worker_id=1,
                    record_ids=["sroie/validation/c"],
                )

    def test_receipt_launchers_are_syntax_valid_and_use_explicit_runtime(self):
        paths = (
            ROOT / "scripts/lib/qwen_receipt_runtime.sh",
            ROOT / "scripts/submit_qwen_smoke.sh",
            ROOT / "scripts/submit_stage.sh",
            ROOT / "scripts/slurm/qwen_receipt_smoke.sbatch",
            ROOT / "scripts/slurm/gpu_stage.sbatch",
            ROOT / "scripts/slurm/qwen_smoke.sbatch",
        )
        for path in paths:
            with self.subTest(path=path):
                subprocess.run(["bash", "-n", str(path)], check=True)

        runtime = paths[0].read_text(encoding="utf-8")
        smoke_submit = paths[1].read_text(encoding="utf-8")
        production_submit = paths[2].read_text(encoding="utf-8")
        smoke_batch = paths[3].read_text(encoding="utf-8")
        production_batch = paths[4].read_text(encoding="utf-8")
        for variable in (
            "PP_PYTHON_BIN",
            "PP_SHARED_ARTIFACT_ROOT",
            "PP_RECEIPT_OUTPUT_ROOT",
        ):
            self.assertIn(variable, runtime)
            self.assertIn(variable, smoke_submit)
        self.assertIn("git -C \"$repo_real\" status --porcelain", runtime)
        self.assertIn("PP_EXPECTED_GIT_SHA", smoke_submit)
        self.assertIn('if [[ "$DRY_RUN" -eq 1 ]]', smoke_submit)
        self.assertIn('"$PP_PYTHON_BIN" -E -s -u', smoke_batch)
        self.assertIn('PYTHON_COMMAND=("$PP_PYTHON_BIN" -E -s -u)', production_batch)
        self.assertIn("receipt production remains gated", production_batch)
        self.assertIn(
            'qwen_attack)\n    GPU_TYPE="a100"; CONSTRAINT="a100-80gb"',
            production_submit,
        )
        self.assertIn(
            'if [[ "$IS_RECEIPT" -eq 1 ]]; then\n'
            '  GPU_TYPE="l40s"\n'
            '  CONSTRAINT=""\n'
            '  MEMORY="64G"\n'
            '  TIME_LIMIT="12:00:00"\n'
            "fi",
            production_submit,
        )

        legacy_smoke = paths[5].read_text(encoding="utf-8")
        self.assertIn("Receipt smoke must use scripts/submit_qwen_smoke.sh", legacy_smoke)

    def test_receipt_launcher_classification_parses_json_and_fails_closed(self):
        runtime = ROOT / "scripts/lib/qwen_receipt_runtime.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            receipt = temporary / "receipt.json"
            resume = temporary / "resume.json"
            malformed = temporary / "malformed.json"
            receipt.write_text('{"track":"receipt"}\n', encoding="utf-8")
            resume.write_text('{"track":"resume"}\n', encoding="utf-8")
            malformed.write_text('{"track":', encoding="utf-8")

            command = 'source "$1"; pp_receipt_is_config "$2"'
            receipt_result = subprocess.run(
                ["bash", "-c", command, "bash", str(runtime), str(receipt)],
                check=False,
            )
            resume_result = subprocess.run(
                ["bash", "-c", command, "bash", str(runtime), str(resume)],
                check=False,
            )
            malformed_result = subprocess.run(
                ["bash", "-c", command, "bash", str(runtime), str(malformed)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(receipt_result.returncode, 0)
        self.assertEqual(resume_result.returncode, 1)
        self.assertEqual(malformed_result.returncode, 2)
        self.assertIn("Cannot classify config", malformed_result.stderr)

    def test_manual_receipt_merge_enforces_group_friendly_umask(self):
        merge_script = (ROOT / "scripts/merge_qwen_receipt.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("os.umask(0o002)", merge_script)

    def test_receipt_merge_recovery_launcher_is_cpu_only_and_explicit(self):
        script = ROOT / "scripts/recover_qwen_receipt_merge.py"
        source = script.read_text(encoding="utf-8")
        compile(source, str(script), "exec")
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        for option in (
            "--expected-attack-git-commit",
            "--expected-config-sha256",
            "--expected-merger-git-commit",
            "--evidence-dir",
            "--num-workers",
        ):
            self.assertIn(option, result.stdout)
        self.assertIn("CPU-only merge recovery", source)
        self.assertIn("recover_merge_if_complete", source)
        self.assertIn("os.umask(0o002)", source)
        self.assertNotIn("sbatch", source)
        self.assertNotIn("QwenVLWrapper", source)


class ReceiptProcessorSettingsRecoveryTests(unittest.TestCase):
    @staticmethod
    def _attack_settings() -> dict:
        return {
            "precision": MODEL["precision"],
            "min_pixels": MODEL["min_pixels"],
            "max_pixels": MODEL["max_pixels"],
            "max_new_tokens": MODEL["max_new_tokens"],
            "do_sample": False,
            "batch_size": 1,
        }

    def _parquet_padded_qwen_settings(self, root: Path) -> dict:
        path = root / "mixed-baseline.parquet"
        donut_settings = {
            "precision": "float32",
            "height": 1280,
            "width": 960,
            "align_long_axis": False,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "max_length": 512,
        }
        pd.DataFrame([
            {"model_key": "donut", "processor_settings": donut_settings},
            {"model_key": "qwen", "processor_settings": self._attack_settings()},
        ]).to_parquet(path, index=False)
        restored = pd.read_parquet(path)
        return dict(
            restored.loc[restored.model_key.eq("qwen"), "processor_settings"].iloc[0]
        )

    def test_mixed_parquet_null_padding_and_integer_valued_floats_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            padded = self._parquet_padded_qwen_settings(Path(directory))

        self.assertEqual(
            set(padded),
            set(qwen_attack_receipt.QWEN_PROCESSOR_REQUIRED_FIELDS)
            | set(qwen_attack_receipt.QWEN_PROCESSOR_NULL_PADDING_FIELDS),
        )
        for field in qwen_attack_receipt.QWEN_PROCESSOR_NULL_PADDING_FIELDS:
            self.assertIsNone(padded[field])
        for field in ("min_pixels", "max_pixels", "max_new_tokens", "batch_size"):
            self.assertIsInstance(padded[field], float)
        qwen_attack_receipt._require_matching_qwen_processor_settings(
            padded,
            self._attack_settings(),
            "sroie/validation/observed",
        )
        qwen_attack_receipt._require_matching_qwen_processor_settings(
            self._attack_settings(),
            padded,
            "sroie/validation/observed",
        )

    def test_required_qwen_processor_setting_drift_is_rejected(self):
        required = self._attack_settings()
        cases = {
            "changed value": ({**required, "max_new_tokens": 64}, "setting drift"),
            "missing required": (
                {key: value for key, value in required.items() if key != "batch_size"},
                "missing required fields",
            ),
            "null required": ({**required, "batch_size": None}, "positive integer"),
            "unknown null": ({**required, "future_setting": None}, "unknown fields"),
            "unknown false": ({**required, "future_setting": False}, "unknown fields"),
            "non-null padding": ({**required, "width": 960}, "non-null padding"),
            "boolean as number": ({**required, "max_pixels": True}, "positive integer"),
            "number as boolean": ({**required, "do_sample": 0}, "must be boolean"),
            "numeric string": ({**required, "min_pixels": "200704"}, "positive integer"),
            "fractional number": ({**required, "batch_size": 1.5}, "positive integer"),
        }
        for name, (observed, message) in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                qwen_attack_receipt._require_matching_qwen_processor_settings(
                    required,
                    observed,
                    "cord_v2/validation/negative",
                )

    def test_baseline_unknown_and_non_null_padding_are_also_rejected(self):
        required = self._attack_settings()
        cases = (
            ({**required, "future_setting": None}, "unknown fields"),
            ({**required, "height": 1280}, "non-null padding"),
            ({**required, "precision": None}, "non-empty string"),
        )
        for baseline, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                qwen_attack_receipt._require_matching_qwen_processor_settings(
                    baseline,
                    required,
                    "sroie/validation/baseline-negative",
                )


class ReceiptSavedRecordAndMergeTests(unittest.TestCase):
    provenance = {
        "run_id": "receipt-smoke-test",
        "stage": "qwen_attack",
        "track": "receipt",
        "config_sha256": "c" * 64,
        "git_commit": "d" * 40,
        "environment_hash": sha256_text(json.dumps(TEST_RUNTIME_VERSIONS, sort_keys=True)),
        "source_evidence": copy.deepcopy(TEST_SOURCE_EVIDENCE),
        "source_records_sha256": TEST_SOURCE_EVIDENCE["source_records_sha256"],
        "source_evidence_report_sha256": TEST_SOURCE_EVIDENCE["accepted_report_sha256"],
        "source_evidence_verified_git_commit": TEST_SOURCE_EVIDENCE["verified_git_commit"],
    }

    @staticmethod
    def _attack_settings() -> dict:
        return {
            "epsilons_255": [8],
            "max_steps": 25,
            "step_size_255": {"8": 2.0},
            "checkpoint_interval": 10,
            "loss_type": "cross_entropy",
            "multi_page_fallback": "none",
        }

    def _contracts(self, ids: set[str]) -> dict[str, dict]:
        processor_settings = {
            "precision": MODEL["precision"],
            "min_pixels": MODEL["min_pixels"],
            "max_pixels": MODEL["max_pixels"],
            "max_new_tokens": MODEL["max_new_tokens"],
            "do_sample": False,
            "batch_size": 1,
        }
        contracts = {}
        for record_id in ids:
            dataset = record_id.split("/", 1)[0]
            clean = "12.34" if dataset == "sroie" else "123"
            target = "123.40" if dataset == "sroie" else "1230"
            source_image = Image.new("RGB", (10, 20), color=(10, 20, 30))
            source_bytes = source_image.tobytes()
            source_hashes = [sha256_bytes(source_bytes)]
            contracts[record_id] = {
                "_source_rgb_bytes": source_bytes,
                "run_id": "receipt-smoke-test",
                "stage": "qwen_attack",
                "track": "receipt",
                "dataset_name": dataset,
                "experiment_split": "validation",
                "model_key": "qwen",
                "model_id": MODEL["model_id"],
                "model_revision": MODEL["revision"],
                "task": "receipt_total",
                "prompt_id": "receipt_total_v1",
                "prompt_text": PROMPT,
                "question": PROMPT,
                "prompt_sha256": sha256_text(PROMPT),
                "clean_target": clean,
                "attack_target": target,
                "target_text": target,
                "usable": True,
                "attack_eligible": True,
                "clean_correct": True,
                "clean_prediction": clean,
                "processor_settings": processor_settings,
                "processor_tolerance": {
                    "max": 0.000001,
                    "mean": 0.00000001,
                    "surrogate_mean": 0.005,
                },
                "source_split": "validation",
                "source_index": 0,
                "source_image_hashes": source_hashes,
                "clean_image_sha256": sha256_text(json.dumps(source_hashes)),
                "source_image_widths": [10],
                "source_image_heights": [20],
                "document_page_count": 1,
                "attacked_page_count": 1,
                "input_page_count": 1,
                "seed": 566,
                "max_steps": 25,
                "loss_type": "cross_entropy",
                "optimizer": "sign",
                "quantize_each_step": True,
                "attack_settings": self._attack_settings(),
                "target_policy": dict(qwen_attack_receipt.TARGET_POLICY),
                "status": "complete",
                "error": None,
            }
        return contracts

    def _write_record(
        self, output: Path, record_id: str, dataset: str, *, success: bool,
        epsilon: int = 8, worker_id: int = 0,
    ) -> Path:
        key = qwen_attack_receipt._record_key(record_id, epsilon, 566)
        image_dir = output / "shards" / f"worker_{worker_id:02d}" / "images" / key
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / "page_000.png"
        Image.new("RGB", (10, 20), color=(10, 20, 30)).save(image_path, format="PNG")
        image_hash = sha256_file(image_path)
        record_path = output / "shards" / f"worker_{worker_id:02d}" / "records" / f"{key}.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        clean = "12.34" if dataset == "sroie" else "123"
        target = "123.40" if dataset == "sroie" else "1230"
        raw_response = target if success else clean
        source_image = Image.new("RGB", (10, 20), color=(10, 20, 30))
        source_hashes = [sha256_bytes(source_image.tobytes())]
        attack_settings = self._attack_settings()
        processor_settings = {
            "precision": MODEL["precision"],
            "min_pixels": MODEL["min_pixels"],
            "max_pixels": MODEL["max_pixels"],
            "max_new_tokens": MODEL["max_new_tokens"],
            "do_sample": False,
            "batch_size": 1,
        }
        record = {
            **self.provenance,
            "experiment_key": key,
            "record_id": record_id,
            "dataset_name": dataset,
            "experiment_split": "validation",
            "model_key": "qwen",
            "model_id": MODEL["model_id"],
            "model_revision": MODEL["revision"],
            "task": "receipt_total",
            "prompt_id": "receipt_total_v1",
            "prompt_text": PROMPT,
            "question": PROMPT,
            "prompt_sha256": sha256_text(PROMPT),
            "clean_target": clean,
            "attack_target": target,
            "target_text": target,
            "usable": True,
            "attack_eligible": True,
            "clean_correct": True,
            "clean_prediction": clean,
            "processor_settings": processor_settings,
            "processor_tolerance": {
                "max": 0.000001,
                "mean": 0.00000001,
                "surrogate_mean": 0.005,
            },
            "source_split": "validation",
            "source_index": 0,
            "source_image_hashes": source_hashes,
            "clean_image_sha256": sha256_text(json.dumps(source_hashes)),
            "source_image_widths": [10],
            "source_image_heights": [20],
            "target_policy": dict(qwen_attack_receipt.TARGET_POLICY),
            "epsilon_255": epsilon,
            "step_size_255": 2.0,
            "max_steps": 25,
            "loss_type": "cross_entropy",
            "optimizer": "sign",
            "quantize_each_step": True,
            "attack_settings": attack_settings,
            "targeted_success_reloaded": success,
            "saved_image_success": success,
            "targeted_success_memory": success,
            "in_memory_success": success,
            "raw_response": raw_response,
            "final_raw_response": raw_response,
            "prediction_normalized": raw_response,
            "final_prediction_normalized": raw_response,
            "parse_status": "ok",
            "linf_255": 0.0,
            "lpips": 0.0,
            "ssim": 1.0,
            "initial_loss": 4.0,
            "final_loss": 2.0,
            "best_loss": 1.9,
            "mean_target_log_probability": -0.5,
            "gradient_norm_final": 2.5,
            "runtime_seconds": 3.0,
            "peak_gpu_memory_mb": 100.0,
            "target_token_ids_hash": "e" * 64,
            "official_processor_shape": [4, 8],
            "differentiable_processor_shape": [4, 8],
            "processor_max_abs_difference": 0.0,
            "processor_mean_abs_difference": 0.0,
            "processor_max_abs_difference_255": 0.0,
            "processor_mean_abs_difference_255": 0.0,
            "processor_surrogate_max_abs_difference": 0.01,
            "processor_surrogate_mean_abs_difference": 0.001,
            "processor_surrogate_max_abs_difference_255": 1.0,
            "processor_surrogate_mean_abs_difference_255": 0.1,
            "processor_gradient_method": "bpda_exact_official_forward",
            "worker_id": worker_id,
            "seed": 566,
            "document_page_count": 1,
            "attacked_page_count": 1,
            "input_page_count": 1,
            "status": "complete",
            "error": None,
            "images": [{
                "page": 0,
                "path": str(image_path),
                "sha256": image_hash,
                "linf_255": 0.0,
                "lpips": 0.0,
                "ssim": 1.0,
                "width": 10,
                "height": 20,
            }],
            "attacked_image_sha256": [image_hash],
            "image_widths": [10],
            "image_heights": [20],
            "optimization_trace": [{
                "target_loss": 2.0,
                "mean_target_log_probability": -0.5,
                "linf_255": 7.0,
                "gradient_norm": 2.5,
                "peak_gpu_memory_mb": 100.0,
                "raw_in_memory": raw_response,
                "raw_reloaded": raw_response,
                "in_memory_success": success,
                "saved_image_success": success,
            }],
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record_path

    def _complete_output(self, root: Path) -> tuple[Path, set[str]]:
        output = root / "outputs/qwen_attack/receipt-smoke-test"
        ids = {
            "sroie/validation/sroie-000",
            "cord_v2/validation/cord_v2-000",
        }
        assignment = sorted(ids)
        assignment_path = output / "assignments/worker_00.txt"
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_path.write_text("".join(f"{record_id}\n" for record_id in assignment), encoding="utf-8")
        self._write_record(output, assignment[0], "cord_v2", success=False)
        self._write_record(output, assignment[1], "sroie", success=True)
        marker = {
            **self.provenance,
            "runtime_contract": dict(TEST_GPU_RUNTIME_CONTRACT),
            "worker_id": 0,
            "record_ids": assignment,
            "num_experiments": 2,
        }
        shard = output / "shards/worker_00"
        environment = {
            **TEST_RUNTIME_VERSIONS,
            **TEST_GPU_RUNTIME_CONTRACT,
            "runtime_versions": dict(TEST_RUNTIME_VERSIONS),
            "runtime_contract": dict(TEST_GPU_RUNTIME_CONTRACT),
            **self.provenance,
        }
        environment_path = shard / "environment.json"
        environment_path.write_text(json.dumps(environment), encoding="utf-8")
        marker_path = shard / "WORKER_COMPLETE.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        return output, ids

    @staticmethod
    def _inventory_paths(root: Path, output: Path) -> dict[str, Path]:
        (output / "config.json").write_text(
            '{"run_id":"receipt-smoke-test"}\n', encoding="utf-8"
        )
        checkpoint = output / "shards/worker_00/checkpoints/state.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('{"step":10}\n', encoding="utf-8")
        (output / "shards/worker_00/progress.json").write_text(
            '{"records":2}\n', encoding="utf-8"
        )

        frozen = root / "frozen"
        frozen.mkdir()
        paths: dict[str, Path] = {}
        for index, key in enumerate((
            "baseline_config",
            "baseline_frozen_config",
            "baseline_predictions",
            "baseline_manifest",
            "baseline_summary",
            "baseline_correct_ids",
            "baseline_environment",
            "master_manifest",
        )):
            path = frozen / f"{index:02d}-{key}.artifact"
            path.write_text(f"{key}\n", encoding="utf-8")
            paths[key] = path
        return paths

    def test_saved_record_validation_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            expected = {
                **self.provenance,
                "experiment_key": key,
                "record_id": record_id,
                "epsilon_255": 8,
                "worker_id": 0,
            }
            self.assertTrue(qwen_attack_receipt._validate_saved_record(record_path, expected))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            Path(record["images"][0]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                qwen_attack_receipt._validate_saved_record(record_path, expected)

    def test_strict_merge_reports_conditional_and_harmful_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            merged = qwen_attack_receipt._merge_if_complete(
                output,
                expected_ids=ids,
                epsilons=[8],
                num_workers=1,
                provenance=dict(self.provenance),
                candidate_counts={"sroie": 1, "cord_v2": 1},
                usable_counts={"sroie": 2, "cord_v2": 4},
                record_contracts=self._contracts(ids),
            )
            self.assertTrue(merged)
            summary = json.loads((output / "merged/summary.json").read_text(encoding="utf-8"))
            groups = {(row["dataset_name"], row["epsilon_255"]): row for row in summary["groups"]}
            sroie = groups[("sroie", 8)]
            cord = groups[("cord_v2", 8)]
            self.assertEqual(sroie["reloaded_successes"], 1)
            self.assertEqual(sroie["conditional_denominator"], 1)
            self.assertEqual(sroie["usable_denominator"], 2)
            self.assertEqual(sroie["conditional_asr"], 1.0)
            self.assertEqual(sroie["harmful_output_rate"], 0.5)
            self.assertEqual(cord["reloaded_successes"], 0)
            self.assertEqual(cord["conditional_asr"], 0.0)
            self.assertEqual(cord["harmful_output_rate"], 0.0)
            self.assertTrue(qwen_attack_receipt._merge_if_complete(
                output,
                expected_ids=ids,
                epsilons=[8],
                num_workers=1,
                provenance=dict(self.provenance),
                candidate_counts={"sroie": 1, "cord_v2": 1},
                usable_counts={"sroie": 2, "cord_v2": 4},
                record_contracts=self._contracts(ids),
            ))
            summary_path = output / "merged/summary.json"
            summary_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outputs changed"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

    def test_strict_merge_accepts_only_observed_qwen_null_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            padded = (
                ReceiptProcessorSettingsRecoveryTests()
                ._parquet_padded_qwen_settings(root)
            )
            output, ids = self._complete_output(root)
            contracts = self._contracts(ids)
            for contract in contracts.values():
                contract["processor_settings"] = copy.deepcopy(padded)
            self.assertTrue(qwen_attack_receipt._merge_if_complete(
                output,
                expected_ids=ids,
                epsilons=[8],
                num_workers=1,
                provenance=dict(self.provenance),
                candidate_counts={"sroie": 1, "cord_v2": 1},
                usable_counts={"sroie": 2, "cord_v2": 4},
                record_contracts=contracts,
            ))

    def test_strict_merge_still_rejects_real_processor_setting_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["processor_settings"]["max_new_tokens"] = 64
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "processor_settings.*drift"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

    def test_recovery_inventory_and_manifest_are_stable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, ids = self._complete_output(root)
            paths = self._inventory_paths(root, output)
            before = qwen_attack_receipt._merge_recovery_inventory(output, paths)
            categories = before["categories"]
            for category in (
                "run_config",
                "assignments",
                "records",
                "images",
                "checkpoints",
                "completion_markers",
                "environments",
                "progress",
                "frozen_inputs",
            ):
                self.assertIn(category, categories)
            self.assertEqual(categories["frozen_inputs"]["file_count"], 8)
            self.assertNotIn(
                "run/.merge.lock",
                {entry["path"] for entry in before["entries"]},
            )

            snapshots = []

            def inventory():
                value = qwen_attack_receipt._merge_recovery_inventory(output, paths)
                snapshots.append(copy.deepcopy(value))
                return value

            recovery = {
                "schema_version": 1,
                "mode": "qwen_receipt_cpu_merge_recovery_v1",
                "attack_git_commit": self.provenance["git_commit"],
                "attack_config_sha256": self.provenance["config_sha256"],
                "merger_git_commit": "e" * 40,
            }
            arguments = {
                "expected_ids": ids,
                "epsilons": [8],
                "num_workers": 1,
                "provenance": dict(self.provenance),
                "candidate_counts": {"sroie": 1, "cord_v2": 1},
                "usable_counts": {"sroie": 2, "cord_v2": 4},
                "record_contracts": self._contracts(ids),
                "merge_recovery": recovery,
                "recovery_inventory": inventory,
            }
            self.assertTrue(qwen_attack_receipt._merge_if_complete(output, **arguments))
            self.assertTrue(qwen_attack_receipt._merge_if_complete(output, **arguments))
            self.assertEqual(len(snapshots), 4)
            self.assertTrue(all(snapshot == before for snapshot in snapshots))

            manifest = json.loads(
                (output / "merged/merge_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["git_commit"], self.provenance["git_commit"])
            self.assertEqual(
                manifest["merge_recovery"]["merger_git_commit"],
                recovery["merger_git_commit"],
            )
            self.assertTrue(manifest["merge_recovery"]["source_artifacts_unchanged"])
            self.assertEqual(
                manifest["merge_recovery"]["source_artifacts"],
                qwen_attack_receipt._inventory_summary(before),
            )
            self.assertEqual(
                qwen_attack_receipt._merge_recovery_inventory(output, paths),
                before,
            )

    def test_recovery_merge_rejects_changed_before_after_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, ids = self._complete_output(root)
            paths = self._inventory_paths(root, output)
            baseline = qwen_attack_receipt._merge_recovery_inventory(output, paths)
            calls = 0

            def changing_inventory():
                nonlocal calls
                calls += 1
                result = copy.deepcopy(baseline)
                if calls > 1:
                    result["manifest_sha256"] = "0" * 64
                return result

            with self.assertRaisesRegex(RuntimeError, "source artifacts changed"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                    merge_recovery={
                        "schema_version": 1,
                        "mode": "qwen_receipt_cpu_merge_recovery_v1",
                        "attack_git_commit": self.provenance["git_commit"],
                        "attack_config_sha256": self.provenance["config_sha256"],
                        "merger_git_commit": "e" * 40,
                    },
                    recovery_inventory=changing_inventory,
                )

    def test_merge_rejects_tampered_deterministic_manifest_fields(self):
        mutations = {
            "records": 0,
            "unique_documents": 0,
            "expected_ids_sha256": "0" * 64,
            "source_records_sha256": "0" * 64,
            "source_evidence_report_sha256": "1" * 64,
            "source_evidence_verified_git_commit": "2" * 40,
            "source_evidence": {
                **TEST_SOURCE_EVIDENCE,
                "scope": "smoke_subset_only",
            },
            "runtime_contract": {
                **TEST_GPU_RUNTIME_CONTRACT,
                "gpu_memory_bytes": TEST_GPU_RUNTIME_CONTRACT["gpu_memory_bytes"] + 1,
            },
        }
        for field, changed_value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output, ids = self._complete_output(Path(directory))
                arguments = {
                    "expected_ids": ids,
                    "epsilons": [8],
                    "num_workers": 1,
                    "provenance": dict(self.provenance),
                    "candidate_counts": {"sroie": 1, "cord_v2": 1},
                    "usable_counts": {"sroie": 2, "cord_v2": 4},
                    "record_contracts": self._contracts(ids),
                }
                self.assertTrue(qwen_attack_receipt._merge_if_complete(output, **arguments))
                manifest_path = output / "merged/merge_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest[field] = changed_value
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "different inputs"):
                    qwen_attack_receipt._merge_if_complete(output, **arguments)

    def test_merge_rejects_worker_environment_and_gpu_drift(self):
        mutations = (
            ("runtime_versions", {**TEST_RUNTIME_VERSIONS, "numpy": "different"}),
            ("source_records_sha256", "0" * 64),
            ("source_evidence_report_sha256", "1" * 64),
            (
                "source_evidence",
                {**TEST_SOURCE_EVIDENCE, "records": 10},
            ),
            (
                "runtime_contract",
                {
                    **TEST_GPU_RUNTIME_CONTRACT,
                    "gpu_memory_bytes": TEST_GPU_RUNTIME_CONTRACT["gpu_memory_bytes"] + 1,
                },
            ),
        )
        for field, changed_value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output, ids = self._complete_output(Path(directory))
                environment_path = output / "shards/worker_00/environment.json"
                environment = json.loads(environment_path.read_text(encoding="utf-8"))
                environment[field] = changed_value
                environment_path.write_text(json.dumps(environment), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    qwen_attack_receipt._merge_if_complete(
                        output,
                        expected_ids=ids,
                        epsilons=[8],
                        num_workers=1,
                        provenance=dict(self.provenance),
                        candidate_counts={"sroie": 1, "cord_v2": 1},
                        usable_counts={"sroie": 2, "cord_v2": 4},
                        record_contracts=self._contracts(ids),
                    )

    def test_merge_rejects_source_evidence_drift_in_marker_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            marker_path = output / "shards/worker_00/WORKER_COMPLETE.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["source_records_sha256"] = "0" * 64
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completion provenance"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["source_evidence"] = {
                **TEST_SOURCE_EVIDENCE,
                "source_records_sha256": "0" * 64,
            }
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "record provenance"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

    def test_merge_rejects_scientific_provenance_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["model_revision"] = "wrong"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "model_revision"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

    def test_merge_recomputes_linf_from_clean_and_saved_rgb_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            image_path = Path(record["images"][0]["path"])
            Image.new("RGB", (10, 20), color=(11, 20, 30)).save(image_path, format="PNG")
            changed_hash = sha256_file(image_path)
            record["images"][0]["sha256"] = changed_hash
            record["attacked_image_sha256"] = [changed_hash]
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "recorded L-infinity differs"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 1, "cord_v2": 1},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )

    def test_saved_record_rejects_false_success_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["raw_response"] = record["clean_target"]
            record["final_raw_response"] = record["clean_target"]
            record["prediction_normalized"] = record["clean_target"]
            record["final_prediction_normalized"] = record["clean_target"]
            record_path.write_text(json.dumps(record), encoding="utf-8")
            expected = {
                **self.provenance,
                "experiment_key": key,
                "record_id": record_id,
                "epsilon_255": 8,
                "worker_id": 0,
            }
            with self.assertRaisesRegex(RuntimeError, "success does not match"):
                qwen_attack_receipt._validate_saved_record(record_path, expected)

    def test_saved_record_rejects_processor_parity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output, _ = self._complete_output(Path(directory))
            record_id = "sroie/validation/sroie-000"
            key = qwen_attack_receipt._record_key(record_id, 8, 566)
            record_path = output / "shards/worker_00/records" / f"{key}.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["processor_mean_abs_difference"] = 1.0
            record_path.write_text(json.dumps(record), encoding="utf-8")
            expected = {
                **self.provenance,
                "experiment_key": key,
                "record_id": record_id,
                "epsilon_255": 8,
                "worker_id": 0,
            }
            with self.assertRaisesRegex(RuntimeError, "exceeds frozen tolerance"):
                qwen_attack_receipt._validate_saved_record(record_path, expected)

    def test_merge_waits_for_marker_and_rejects_denominator_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "empty"
            self.assertFalse(qwen_attack_receipt._merge_if_complete(
                output,
                expected_ids={"sroie/validation/a"},
                epsilons=[8],
                num_workers=1,
                provenance=dict(self.provenance),
                candidate_counts={"sroie": 1, "cord_v2": 0},
                usable_counts={"sroie": 1, "cord_v2": 0},
                record_contracts={
                    "sroie/validation/a": {"dataset_name": "sroie"}
                },
            ))

        with tempfile.TemporaryDirectory() as directory:
            output, ids = self._complete_output(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "candidate denominators differ"):
                qwen_attack_receipt._merge_if_complete(
                    output,
                    expected_ids=ids,
                    epsilons=[8],
                    num_workers=1,
                    provenance=dict(self.provenance),
                    candidate_counts={"sroie": 2, "cord_v2": 0},
                    usable_counts={"sroie": 2, "cord_v2": 4},
                    record_contracts=self._contracts(ids),
                )


if __name__ == "__main__":
    unittest.main()
