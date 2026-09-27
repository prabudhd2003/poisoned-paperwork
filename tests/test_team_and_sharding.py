import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sharding import assigned_record_ids
from team import get_team_member


class TeamAndShardingTests(unittest.TestCase):
    def test_fixed_team_mapping(self):
        self.assertEqual(get_team_member("user1").name, "Prabudhd")
        self.assertEqual(get_team_member("user1").worker_id, 0)
        self.assertEqual(get_team_member("user5").name, "Shail")
        self.assertEqual(get_team_member("user5").worker_id, 4)

    def test_five_shards_are_disjoint_and_complete(self):
        record_ids = [f"record-{index:03d}" for index in range(23)]
        shards = [assigned_record_ids(record_ids, worker_id) for worker_id in range(5)]

        combined = [record_id for shard in shards for record_id in shard]
        self.assertEqual(sorted(combined), sorted(record_ids))
        self.assertEqual(len(combined), len(set(combined)))

    def test_assignment_is_order_independent(self):
        record_ids = ["c", "a", "b", "e", "d"]
        self.assertEqual(
            assigned_record_ids(record_ids, worker_id=1),
            assigned_record_ids(reversed(record_ids), worker_id=1),
        )


if __name__ == "__main__":
    unittest.main()
