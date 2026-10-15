import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attacks.donut_validation import validate_initial_gradient


class GradientAcceptanceTests(unittest.TestCase):
    def test_converged_resume_is_allowed(self):
        validate_initial_gradient(dict(finite=True, nonzero=False), resumed=True,
                                  loss_type="paper_margin", initial_loss=0.0)

    def test_zero_gradient_elsewhere_is_rejected(self):
        for resumed, loss_type, loss in [(False, "paper_margin", 0.0),
                                          (True, "paper_margin", 0.1),
                                          (True, "cross_entropy", 0.0)]:
            with self.subTest(resumed=resumed, loss_type=loss_type, loss=loss):
                with self.assertRaises(RuntimeError):
                    validate_initial_gradient(dict(finite=True, nonzero=False),
                                              resumed=resumed, loss_type=loss_type,
                                              initial_loss=loss)

    def test_nonfinite_always_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_initial_gradient(dict(finite=False, nonzero=True), resumed=True,
                                      loss_type="paper_margin", initial_loss=0.0)

    def test_nonzero_finite_gradient_is_allowed(self):
        validate_initial_gradient(dict(finite=True, nonzero=True), resumed=False,
                                  loss_type="paper_margin", initial_loss=1.0)
