"""Acceptance checks shared by Donut audits (no model loading)."""

import math


def validate_initial_gradient(check, *, resumed, loss_type, initial_loss):
    """A converged margin objective may have zero gradient after resume."""
    if not check or not check["finite"] or not math.isfinite(initial_loss):
        raise RuntimeError("Missing/nonfinite initial gradient or loss")
    if not check["nonzero"] and not (
        resumed and loss_type == "paper_margin" and initial_loss == 0.0
    ):
        raise RuntimeError("Unexpected zero initial image gradient")
