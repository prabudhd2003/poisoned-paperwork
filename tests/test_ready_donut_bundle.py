"""CPU checks for dependency submission and immutable runtime inputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/ready_donut" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("queue_ready_donut")
smoke = load("smoke_ready_donut")


class ReadyDonutTests(unittest.TestCase):
    def test_real_launcher_dependencies(self):
        templates = [
            "configs/donut_attack/collected_external_test.json",
            "configs/jpeg_eot_attack/donut_public_full.json",
            "configs/jpeg_eot_attack/donut_public_frequency_hardview.json",
        ]
        for job, config in zip(queue.JOBS, templates):
            rendered = subprocess.check_output(
                ["bash", "scripts/submit_stage.sh", job["stage"], "all", config, "--dry-run"],
                cwd=ROOT, text=True,
            )
            args = queue.sbatch_from_launcher(rendered, "12345", "abcdef")
            self.assertIn("--dependency=afterok:12345", args)
            self.assertIn("--parsable", args)
            self.assertIn("--array=0-4", args)
            self.assertIn("--job-name=" + job["name"], args)
            self.assertTrue(any("PP_EXPECTED_GIT_SHA=abcdef" in arg for arg in args))
            self.assertTrue(args[-1].endswith("gpu_stage.sbatch"))

    def test_reject_changed_resource_contract(self):
        with self.assertRaises(RuntimeError):
            queue.sbatch_from_launcher("sbatch --array=0-3 --export=ALL worker.sh", "1", "x")

    def test_active_worker_and_array_detection(self):
        job = queue.JOBS[0]
        rows = [dict(name=name) for name in [job["name"], job["name"] + "_w3", "pp_qwen_attack_collected_receipts"]]
        self.assertEqual(queue.matching_active(job, rows), rows[:2])

    def test_checkpoint_blocks_fresh_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(queue.has_progress(root))
            checkpoint = root / "shards/worker_00/checkpoints/resume.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertTrue(queue.has_progress(root))

    def test_manifest_rejects_config_and_code_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text("{}")
            helper = Path(smoke.__file__)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(
                root=str(root), git_commit="abc", helper=str(helper),
                helper_sha256=smoke.digest(helper),
                jobs=[dict(config_path=str(config), config_sha256=smoke.digest(config))],
            )))
            with patch.object(smoke.subprocess, "check_output", side_effect=["abc\n", ""]):
                smoke.validate_manifest(manifest)
            config.write_text('{"changed":true}')
            with patch.object(smoke.subprocess, "check_output", side_effect=["abc\n", ""]):
                with self.assertRaisesRegex(RuntimeError, "config changed"):
                    smoke.validate_manifest(manifest)
            with patch.object(smoke.subprocess, "check_output", return_value="different\n"):
                with self.assertRaisesRegex(RuntimeError, "commit changed"):
                    smoke.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
