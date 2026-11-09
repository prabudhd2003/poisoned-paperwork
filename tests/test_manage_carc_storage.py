import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManageCarcStorageTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "manage_carc_storage", ROOT / "scripts/manage_carc_storage.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.module)

    def test_completed_run_requires_merged_output_and_every_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "merged").mkdir()
            (run / "merged/summary.json").write_text("{}")
            for worker in range(2):
                shard = run / f"shards/worker_{worker:02d}"
                shard.mkdir(parents=True)
                (shard / "WORKER_COMPLETE.json").write_text("{}")
            self.assertEqual(self.module.completed_run(run, 2), (True, "complete"))
            self.assertFalse(self.module.completed_run(run, 3)[0])

    def test_migrate_directory_copies_verifies_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "repo/outputs/stage/run"
            destination = temporary / "scratch/outputs/stage/run"
            source.mkdir(parents=True)
            (source / "record.json").write_text('{"ok": true}\n')
            self.module.migrate_directory(source, destination, apply=True)
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(), destination.resolve())
            self.assertEqual((source / "record.json").read_text(), '{"ok": true}\n')

    def test_relative_repo_path_rejects_escape(self):
        with self.assertRaises(SystemExit):
            self.module.relative_repo_path("../outside")
        with self.assertRaises(SystemExit):
            self.module.relative_repo_path("/absolute")

    def test_archive_copy_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "source"
            destination = temporary / "destination"
            (source / "merged").mkdir(parents=True)
            (source / "merged/summary.json").write_text("{}")
            (source / "shards/worker_00/checkpoints").mkdir(parents=True)
            (source / "shards/worker_00/checkpoints/state.pt").write_bytes(b"large")
            (source / "shards/worker_00/records").mkdir(parents=True)
            (source / "shards/worker_00/records/one.json").write_text("{}")
            self.module.rsync_copy(source, destination, exclude_checkpoints=True)
            self.assertTrue((destination / "merged/summary.json").is_file())
            self.assertTrue((destination / "shards/worker_00/records/one.json").is_file())
            self.assertFalse((destination / "shards/worker_00/checkpoints").exists())

    def test_every_collected_followup_has_a_scratch_run_definition(self):
        plan = json.loads(
            (ROOT / "configs/storage/carc_scratch_plan.json").read_text()
        )
        links = set(plan["active_run_links"])
        expected = {
            "outputs/defenses/defenses_qwen_collected_final_v1",
            "outputs/defenses/defenses_donut_collected_final_v1",
            "outputs/jpeg_eot_attack/qwen_collected_jpeg_eot_full_v1",
            "outputs/jpeg_eot_attack/qwen_collected_frequency_hardview_full_v1",
            "outputs/jpeg_eot_attack/donut_collected_jpeg_eot_full_v1",
            "outputs/jpeg_eot_attack/donut_collected_frequency_hardview_full_v1",
        }
        self.assertTrue(expected.issubset(links))


if __name__ == "__main__":
    unittest.main()
