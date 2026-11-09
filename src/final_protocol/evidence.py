"""Evidence-grounded selective inference. No labels or attack targets enter inference."""

from __future__ import annotations
import csv
import io
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from normalization import (
    RESUME_DEGREE_LABELS,
    normalize_receipt_total,
    normalize_resume_degree,
)


@dataclass(frozen=True)
class Line:
    page: int
    text: str
    box: tuple[int, int, int, int]
    confidence: float


class TesseractOCR:
    def __init__(self, settings):
        self.settings = settings

    def __call__(self, pages):
        lines = []
        with tempfile.TemporaryDirectory() as directory:
            for index, page in enumerate(pages):
                path = Path(directory) / "page.png"
                page.save(path)
                args = [
                    self.settings["executable"],
                    str(path),
                    "stdout",
                    "--tessdata-dir",
                    self.settings["tessdata_dir"],
                    "-l",
                    self.settings["language"],
                    "--oem",
                    "1",
                    "--psm",
                    "3",
                    "tsv",
                ]
                output = subprocess.run(
                    args,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.settings["timeout_seconds"],
                ).stdout
                groups = {}
                for word in csv.DictReader(io.StringIO(output), delimiter="\t"):
                    if word["level"] != "5" or not word["text"].strip():
                        continue
                    key = tuple(word[k] for k in ("block_num", "par_num", "line_num"))
                    groups.setdefault(key, []).append(word)
                for words in groups.values():
                    left = min(int(w["left"]) for w in words)
                    top = min(int(w["top"]) for w in words)
                    right = max(int(w["left"]) + int(w["width"]) for w in words)
                    bottom = max(int(w["top"]) + int(w["height"]) for w in words)
                    lines.append(
                        Line(
                            index,
                            " ".join(w["text"] for w in words),
                            (left, top, right, bottom),
                            min(float(w["conf"]) for w in words),
                        )
                    )
        return sorted(lines, key=lambda line: (line.page, line.box[1], line.box[0]))


TOTAL = re.compile(
    r"\b(?:grand\s+total|total|amount\s+due|balance\s+due|jumlah)\b", re.I
)
EXCLUDE = re.compile(
    r"\b(?:sub\s*total|tax|tender|cash|change|saving|discount|item)\b", re.I
)
AMOUNT = re.compile(r"(?<!\w)\d[\d,.]*(?!\w)")
DEGREES = (
    ("doctorate", r"\b(?:ph\.?\s*d\.?|doctorate|doctor of)\b"),
    ("master", r"\b(?:master(?:s|\x27s)?|m\.?sc\.?|mba|m\.?tech\.?)\b"),
    ("postgraduate", r"\bpostgraduate\b"),
    ("bachelor", r"\b(?:bachelor(?:s|\x27s)?|b\.?sc\.?|b\.?tech\.?|b\.?eng\.?)\b"),
    ("associate", r"\bassociate(?:s|\x27s)?\b"),
    ("certificate_or_diploma", r"\b(?:diploma|certificate)\b"),
    ("secondary", r"\b(?:high school|secondary school)\b"),
)
UNCERTAIN = re.compile(
    r"\b(?:pursuing|expected|candidate|enrolled|ongoing|in progress|incomplete|not completed|require[ds]?|preferred|students?|teaching|supervis|honorary)\w*\b",
    re.I,
)


def localize(lines, dataset, minimum_confidence=60):
    """Conservative rules. Ambiguous evidence causes abstention, never label lookup."""
    candidates, uncertain = [], False
    if dataset != "resume_parsing_vision":
        for line in lines:
            if not TOTAL.search(line.text) or EXCLUDE.search(line.text):
                continue
            amounts = AMOUNT.findall(line.text)
            values = {normalize_receipt_total(a, dataset) for a in amounts}
            values.discard(None)
            if len(values) != 1 or line.confidence < minimum_confidence:
                uncertain = True
            else:
                candidates.append((next(iter(values)), line))
        values = {value for value, _ in candidates}
        answer = next(iter(values)) if len(values) == 1 and not uncertain else None
    else:
        # A degree mention alone is not evidence of an earned qualification.
        # Scan all pages; refuse uncertain degree statements even outside Education.
        education = False
        for line in lines:
            text = line.text
            if re.search(
                r"\b(?:education|academic qualifications|qualifications)\b", text, re.I
            ):
                education = True
            elif re.fullmatch(
                r"\s*(?:experience|skills|projects|publications|employment|references)\s*",
                text,
                re.I,
            ):
                education = False
            matches = [
                label for label, pattern in DEGREES if re.search(pattern, text, re.I)
            ]
            if not matches:
                continue
            if UNCERTAIN.search(text):
                uncertain = True
                continue
            earned = re.search(
                r"\b(?:graduated|awarded|earned|completed)\b", text, re.I
            )
            dated = re.search(r"\b(?:19|20)\d{2}\b", text)
            if line.confidence < minimum_confidence or not (
                earned or (education and dated)
            ):
                uncertain = True
                continue
            candidates.extend((label, line) for label in matches)
        answer = (
            max((v for v, _ in candidates), key=RESUME_DEGREE_LABELS.index)
            if candidates and not uncertain
            else None
        )
    return (
        answer,
        [line for _, line in candidates],
        "ambiguous_or_missing_evidence" if answer is None else "evidence_found",
    )


def infer(pages, dataset, prompt, predict, ocr, settings, variant="evidence"):
    """Only pixels, task identity, prompt and frozen settings cross this boundary."""
    if variant not in {"none", "ocr_only", "full_ocr", "full_crop", "evidence"}:
        raise ValueError(variant)
    started = time.perf_counter()
    normalize = (
        normalize_resume_degree
        if dataset == "resume_parsing_vision"
        else lambda s: normalize_receipt_total(s, dataset)
    )
    full = crop = evidence = None
    calls = 0
    ocr_pages = 0
    lines = []
    regions = []
    reason = "model_prediction"
    if variant != "ocr_only":
        full = normalize(predict(pages, prompt))
        calls += 1
    if variant != "none":
        lines = ocr(pages)
        ocr_pages = len(pages)
        evidence, regions, reason = localize(
            lines, dataset, settings["minimum_confidence"]
        )
    if variant in {"full_crop", "evidence"} and regions:
        crops = []
        for region in regions:
            page = pages[region.page]
            l, t, r, b = region.box
            pad = max(settings["crop_padding_pixels"], b - t)
            crops.append(
                page.crop(
                    (
                        max(0, l - pad),
                        max(0, t - pad),
                        min(page.width, r + pad),
                        min(page.height, b + pad),
                    )
                )
            )
        if dataset == "resume_parsing_vision":
            crop = normalize(predict(crops, prompt))
            calls += 1
        else:
            predictions = [normalize(predict([image], prompt)) for image in crops]
            calls += len(crops)
            crop = (
                predictions[0]
                if predictions
                and predictions[0] is not None
                and len(set(predictions)) == 1
                else None
            )
    if variant == "none":
        answer = full
    elif variant == "ocr_only":
        answer = evidence
    elif variant == "full_ocr":
        answer = full if full is not None and full == evidence else None
    elif variant == "full_crop":
        answer = full if full is not None and full == crop else None
    else:
        answer = full if full is not None and full == crop == evidence else None
    return dict(
        prediction=answer,
        abstained=answer is None,
        full_prediction=full,
        crop_prediction=crop,
        ocr_prediction=evidence,
        reason=reason,
        vlm_calls=calls,
        ocr_pages=ocr_pages,
        runtime_seconds=time.perf_counter() - started,
        evidence_lines=[asdict(line) for line in lines],
        regions=[asdict(line) for line in regions],
    )
