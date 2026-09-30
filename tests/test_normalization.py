import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from normalization import normalize_receipt_total


class ReceiptNormalizationTests(unittest.TestCase):
    def test_sroie(self):
        self.assertEqual(normalize_receipt_total("$250", "sroie"), "250.00")
        self.assertEqual(normalize_receipt_total("Total: 1,234.50", "sroie"), "1234.50")

    def test_cord(self):
        self.assertEqual(normalize_receipt_total("45,500", "cord_v2"), "45500")
        self.assertEqual(normalize_receipt_total("Rp 45.500", "cord_v2"), "45500")

    def test_rejects_ambiguous_or_invalid_output(self):
        self.assertIsNone(normalize_receipt_total("10 or 20", "sroie"))
        self.assertIsNone(normalize_receipt_total("no answer", "cord_v2"))
        self.assertIsNone(normalize_receipt_total("-1", "cord_v2"))


if __name__ == "__main__":
    unittest.main()
