"""CPU-only tests for Qwen receipt merge-recovery provenance and evidence."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from checkpoints import sha256_bytes
from stages import qwen_attack_receipt


class ReceiptMergeRecoveryIdentityTests(unittest.TestCase):
    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True
        ).strip()

    def _repository(self, temporary: Path) -> tuple[Path, Path, Path, str, str, str]:
        root = temporary / "recovery-checkout"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "receipt-tests@example.invalid"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Receipt Tests"],
            cwd=root,
            check=True,
        )
        config_path = root / "configs/qwen_attack/receipt_active.json"
        config_path.parent.mkdir(parents=True)
        config_bytes = b'{"run_id":"qwen_receipt_validation_v1"}\n'
        config_path.write_bytes(config_bytes)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "attack"], cwd=root, check=True
        )
        attack_commit = self._git(root, "rev-parse", "HEAD")

        implementation = root / "merge-recovery.txt"
        implementation.write_text("narrow merge repair\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "recovery"], cwd=root, check=True
        )
        merger_commit = self._git(root, "rev-parse", "HEAD")

        output = temporary / "existing-run"
        output.mkdir()
        (output / "config.json").write_bytes(config_bytes)
        return (
            root,
            config_path,
            output,
            attack_commit,
            merger_commit,
            sha256_bytes(config_bytes),
        )

    def test_identity_binds_attack_config_run_config_and_real_merger_head(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._repository(Path(directory))
            root, config_path, output, attack, merger, config_hash = values
            identity = qwen_attack_receipt._validate_merge_recovery_identity(
                config_path=config_path,
                root=root,
                output=output,
                expected_attack_git_commit=attack,
                expected_config_sha256=config_hash,
                expected_merger_git_commit=merger,
            )
            self.assertEqual(identity["attack_git_commit"], attack)
            self.assertEqual(identity["merger_git_commit"], merger)
            self.assertTrue(identity["config_bytes_match_exactly"])
            self.assertTrue(identity["attack_is_ancestor_of_merger"])
            self.assertEqual(
                set(identity["config_hashes"].values()), {config_hash}
            )

            (output / "config.json").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config hashes"):
                qwen_attack_receipt._validate_merge_recovery_identity(
                    config_path=config_path,
                    root=root,
                    output=output,
                    expected_attack_git_commit=attack,
                    expected_config_sha256=config_hash,
                    expected_merger_git_commit=merger,
                )

    def test_identity_rejects_dirty_checkout_and_wrong_merger_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._repository(Path(directory))
            root, config_path, output, attack, merger, config_hash = values
            with self.assertRaisesRegex(RuntimeError, "HEAD differs"):
                qwen_attack_receipt._validate_merge_recovery_identity(
                    config_path=config_path,
                    root=root,
                    output=output,
                    expected_attack_git_commit=attack,
                    expected_config_sha256=config_hash,
                    expected_merger_git_commit="f" * 40,
                )

            (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "clean Git worktree"):
                qwen_attack_receipt._validate_merge_recovery_identity(
                    config_path=config_path,
                    root=root,
                    output=output,
                    expected_attack_git_commit=attack,
                    expected_config_sha256=config_hash,
                    expected_merger_git_commit=merger,
                )

    def test_recovery_code_and_config_must_share_one_checkout(self):
        qwen_attack_receipt._require_recovery_implementation_root(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "same checkout"):
                qwen_attack_receipt._require_recovery_implementation_root(
                    Path(directory)
                )


class ReceiptMergeRecoveryInventoryTests(unittest.TestCase):
    def test_inventory_covers_sources_and_frozen_inputs_but_not_merge_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            output = temporary / "run"
            run_files = {
                "config.json": b"config\n",
                "assignments/worker_00.txt": b"sroie/validation/a\n",
                "shards/worker_00/records/a.json": b"{}\n",
                "shards/worker_00/images/a/page_000.png": b"png",
                "shards/worker_00/checkpoints/a/state.pt": b"checkpoint",
                "shards/worker_00/WORKER_COMPLETE.json": b"{}\n",
                "shards/worker_00/environment.json": b"{}\n",
                "shards/worker_00/progress.json": b"{}\n",
                "shards/worker_00/failures/failed.json": b"{}\n",
                ".metadata.lock": b"metadata-lock-content\n",
            }
            for relative, contents in run_files.items():
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            (output / ".merge.lock").write_text(
                "must-not-be-truncated\n", encoding="utf-8"
            )
            merged_output = output / "merged/attacks.csv"
            merged_output.parent.mkdir(parents=True)
            merged_output.write_text("new output\n", encoding="utf-8")
            temporary_output = output / ".merged.tmp.old/partial.csv"
            temporary_output.parent.mkdir(parents=True)
            temporary_output.write_text("partial\n", encoding="utf-8")

            frozen_names = (
                "baseline_config",
                "baseline_frozen_config",
                "baseline_predictions",
                "baseline_manifest",
                "baseline_summary",
                "baseline_correct_ids",
                "baseline_environment",
                "master_manifest",
            )
            frozen = temporary / "frozen"
            frozen.mkdir()
            paths = {}
            for name in frozen_names:
                path = frozen / name
                path.write_text(f"{name}\n", encoding="utf-8")
                paths[name] = path

            before = qwen_attack_receipt._merge_recovery_inventory(output, paths)
            after = qwen_attack_receipt._merge_recovery_inventory(output, paths)
            self.assertEqual(before, after)
            self.assertEqual(before["file_count"], len(run_files) + len(frozen_names))
            self.assertEqual(
                set(before["categories"]),
                {
                    "assignments",
                    "checkpoints",
                    "completion_markers",
                    "environments",
                    "failures",
                    "frozen_inputs",
                    "images",
                    "progress",
                    "records",
                    "run_config",
                    "run_other",
                },
            )
            inventory_paths = {entry["path"] for entry in before["entries"]}
            self.assertNotIn("run/.merge.lock", inventory_paths)
            self.assertFalse(any("/merged/" in path for path in inventory_paths))
            self.assertFalse(any(".merged.tmp." in path for path in inventory_paths))
            self.assertEqual(
                (output / ".merge.lock").read_text(encoding="utf-8"),
                "must-not-be-truncated\n",
            )

            record = output / "shards/worker_00/records/a.json"
            record.write_text('{"changed":true}\n', encoding="utf-8")
            changed = qwen_attack_receipt._merge_recovery_inventory(output, paths)
            self.assertNotEqual(
                before["manifest_sha256"], changed["manifest_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
