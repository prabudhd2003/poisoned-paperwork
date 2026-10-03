"""Conservative output normalization shared by receipt experiments."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_AMOUNT = re.compile(r"[-+]?\d[\d,.]*")


def normalize_receipt_total(text: str | None, dataset_name: str) -> str | None:
    """Normalize exactly one unambiguous receipt amount.

    SROIE totals are represented with two decimal places. CORD totals are
    non-negative integer rupiah values. Explanatory output with multiple
    numeric candidates is rejected rather than guessed.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^(?:total|amount|answer)\s*[:=-]?\s*", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" \t\r\n$€£¥₹")
    matches = _AMOUNT.findall(cleaned)
    if len(matches) != 1:
        return None
    value = matches[0]

    try:
        if dataset_name == "cord_v2":
            compact = value.replace(",", "").replace(".", "")
            if value.startswith("-") or not compact.isdigit():
                return None
            return str(int(compact))
        if dataset_name == "sroie":
            # SROIE uses a decimal amount; commas are thousands separators.
            amount = Decimal(value.replace(",", ""))
            if amount < 0:
                return None
            return f"{amount:.2f}"
    except (InvalidOperation, ValueError):
        return None
    raise ValueError(f"unsupported receipt dataset: {dataset_name}")

RESUME_DEGREE_LABELS = (
    "secondary",
    "certificate_or_diploma",
    "associate",
    "bachelor",
    "postgraduate",
    "master",
    "doctorate",
)


def normalize_resume_degree(text: str | None) -> str | None:
    """Accept only one exact label from the seven-label vocabulary.

    The response is lowercased and trimmed; nothing is inferred from sentences
    or degree names ("Master of Science" is a parse failure, not "master").
    """
    if not isinstance(text, str):
        return None
    cleaned = text.strip().lower()
    return cleaned if cleaned in RESUME_DEGREE_LABELS else None
