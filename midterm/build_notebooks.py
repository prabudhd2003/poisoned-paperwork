"""Regenerate the small report notebooks from the definitions below."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(("markdown:" + source).encode()).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(("code:" + source).encode()).hexdigest()[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INTRO = """The analysis is CPU-only: it does not load Qwen or Donut and is safe to run while GPU jobs are active. It displays the table in the notebook and saves report-ready 300-DPI PNG figures and CSV tables under `midterm/generated/`. If a future run is missing, the final cell explains which environment variable can point to it; rerun all cells after the run finishes."""

BOOTSTRAP = """from pathlib import Path
import sys

REPO_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'midterm' / 'reporting.py').is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from midterm.reporting import run_attack_report
"""


DEFINITIONS = [
    {
        "file": "01_qwen_sroie.ipynb",
        "title": "Qwen × SROIE results",
        "stem": "qwen_sroie",
        "candidates": [
            "/project2/yzhao010_1531/qwen-receipt-runs/qwen_attack/qwen_receipt_validation_v1",
            "outputs/qwen_attack/qwen_receipt_validation_v1",
        ],
        "env": "PP_QWEN_RECEIPT_RUN_DIR",
        "dataset": "sroie",
        "epsilon": 16,
    },
    {
        "file": "02_qwen_cord.ipynb",
        "title": "Qwen × CORD v2 results",
        "stem": "qwen_cord",
        "candidates": [
            "/project2/yzhao010_1531/qwen-receipt-runs/qwen_attack/qwen_receipt_validation_v1",
            "outputs/qwen_attack/qwen_receipt_validation_v1",
        ],
        "env": "PP_QWEN_RECEIPT_RUN_DIR",
        "dataset": "cord",
        "epsilon": 16,
    },
    {
        "file": "03_qwen_resume.ipynb",
        "title": "Qwen × resume parsing results",
        "stem": "qwen_resume",
        "candidates": [
            "/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork/outputs/qwen_attack/qwen_resume_validation_v1",
            "outputs/qwen_attack/qwen_resume_validation_v1",
        ],
        "env": "PP_QWEN_RESUME_RUN_DIR",
        "dataset": "resume",
        "epsilon": 16,
    },
    {
        "file": "04_donut_sroie.ipynb",
        "title": "Donut × SROIE results",
        "stem": "donut_sroie",
        "candidates": [
            "/project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree/outputs/donut_attack/donut_paper_full_validation_v1",
        ],
        "extra": [["/project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree/outputs/donut_attack/donut_paper_eps8_16_20_v1"]],
        "env": "PP_DONUT_RUN_DIR",
        "dataset": "sroie",
        "epsilon": 32,
    },
    {
        "file": "05_donut_cord.ipynb",
        "title": "Donut × CORD v2 results",
        "stem": "donut_cord",
        "candidates": [
            "/project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree/outputs/donut_attack/donut_paper_full_validation_v1",
        ],
        "extra": [["/project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree/outputs/donut_attack/donut_paper_eps8_16_20_v1"]],
        "env": "PP_DONUT_RUN_DIR",
        "dataset": "cord",
        "epsilon": 32,
    },
    {
        "file": "06_qwen_collected_receipts.ipynb",
        "title": "Qwen × team-collected receipts",
        "stem": "qwen_collected",
        "candidates": ["outputs/qwen_attack/qwen_collected_external_v1"],
        "env": "PP_QWEN_COLLECTED_RUN_DIR",
        "dataset": None,
        "epsilon": 16,
    },
    {
        "file": "07_donut_collected_receipts.ipynb",
        "title": "Donut × team-collected receipts",
        "stem": "donut_collected",
        "candidates": ["outputs/donut_attack/donut_collected_external_v1"],
        "env": "PP_DONUT_COLLECTED_RUN_DIR",
        "dataset": None,
        "epsilon": 16,
    },
]


def build() -> None:
    overview = notebook(
        [
            markdown("# Current authoritative results overview\n\n" + INTRO + "\n\nThis notebook reads the exact merged baseline, Qwen, and Donut outputs on CARC. It does not use a copied results snapshot. Run it after pulling this repository into the main CARC folder."),
            code("from pathlib import Path\nimport sys\nREPO_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'midterm' / 'reporting.py').is_file())\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\nfrom midterm.reporting import render_authoritative_overview"),
            markdown("## Report tables and figures\n\nThe first figure compares clean accuracy and attack success. The second makes the effectiveness–perceptibility trade-off explicit."),
            code("baseline_table, attack_table = render_authoritative_overview()\ndisplay(baseline_table)\ndisplay(attack_table)"),
            markdown("## Interpretation\n\nQwen is highly accurate on clean public documents, so the attacks are measured on a meaningful clean-correct set. Attack success grows sharply with ε on both receipt datasets, while resumes are substantially harder. Donut's lower clean accuracy must be reported alongside its attack results: unconditional failure is not the same as a successful targeted attack. These findings establish a non-trivial robustness–visibility trade-off and motivate transfer tests and defenses."),
        ]
    )
    (HERE / "00_current_results_overview.ipynb").write_text(json.dumps(overview, indent=1) + "\n")

    for definition in DEFINITIONS:
        call = (
            "result = run_attack_report(\n"
            f"    title={definition['title']!r},\n"
            f"    stem={definition['stem']!r},\n"
            f"    run_candidates={definition['candidates']!r},\n"
            f"    env_var={definition['env']!r},\n"
            f"    dataset_filter={definition['dataset']!r},\n"
            f"    qualitative_epsilon={definition['epsilon']!r},\n"
            f"    extra_run_candidates={definition.get('extra', [])!r},\n"
            ")"
        )
        cells = [
            markdown(f"# {definition['title']}\n\n{INTRO}"),
            code(BOOTSTRAP),
            markdown("## Quantitative results\n\nThe table includes targeted attack success rate (ASR), Wilson 95% confidence intervals, perceptual quality, runtime, and optimization steps. The four-panel figure is the main quantitative report figure."),
            code(call),
            markdown("## How to read the qualitative panel\n\nThe original and attacked images are rendered at native resolution. The third panel is a heatmap of the largest RGB-channel change at each pixel; it intentionally exaggerates where the attack acts and is not an image shown to the model. A visually similar attacked image with changed model output is evidence of a successful adversarial example."),
            markdown("## Report checklist\n\nBefore quoting final values, confirm the expected number of trials at every ε, inspect any failure logs, and check that the notebook loaded the intended run directory printed above. Report clean baseline accuracy beside conditional ASR."),
        ]
        (HERE / definition["file"]).write_text(json.dumps(notebook(cells), indent=1) + "\n")

    defense = notebook(
        [
            markdown("# Defense results\n\n" + INTRO + "\n\nThis future-ready notebook analyzes effectiveness, retained clean utility, abstention, compute overhead, and adaptive-attack robustness. These dimensions are reported together so a defense cannot look good merely by rejecting everything."),
            code("from pathlib import Path\nimport sys\nREPO_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'midterm' / 'reporting.py').is_file())\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\nfrom midterm.reporting import run_defense_report"),
            code("defense_result = run_defense_report()"),
            markdown("## Interpretation checklist\n\nA useful defense should lower targeted ASR without collapsing clean accuracy, excessive abstention, or prohibitive extra model calls. Adaptive attack results are the stronger test: they measure whether the defense still helps when the attacker knows it is present."),
        ]
    )
    (HERE / "08_defense_results.ipynb").write_text(json.dumps(defense, indent=1) + "\n")

    report = notebook(
        [
            markdown("# Midterm report figures\n\nThis notebook regenerates cross-experiment figures and CSV tables directly from the authoritative CARC result directories. After new experiments finish, run their dataset notebooks for updated per-combination figures."),
            code("from pathlib import Path\nimport sys\nREPO_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'midterm' / 'reporting.py').is_file())\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\nfrom midterm.reporting import render_authoritative_overview"),
            code("baseline_table, attack_table = render_authoritative_overview()\ndisplay(baseline_table)\ndisplay(attack_table)"),
            markdown("## Suggested report placement\n\nUse `current_results_overview.png` in the main Results section and `current_attack_tradeoff.png` when discussing stealth versus effectiveness. Use the per-dataset quantitative and native-resolution qualitative figures from notebooks 01–07 as supporting figures. CSV exports provide the exact values for prose and captions."),
        ]
    )
    (HERE / "09_report_figures.ipynb").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    build()
    print(f"Wrote 10 notebooks under {HERE}")

