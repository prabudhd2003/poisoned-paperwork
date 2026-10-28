# Midterm results notebooks

This directory is the report-facing analysis package for Poisoned Paperwork.
It is intentionally separate from the terminal/GPU runbook: every notebook is
CPU-only, reads completed result files, displays report tables, and exports
300-DPI PNG figures plus CSV tables under `midterm/generated/`.

## Notebook order

1. `00_current_results_overview.ipynb` — rebuilds the overview from authoritative CARC outputs.
2. `01_qwen_sroie.ipynb` — Qwen × SROIE.
3. `02_qwen_cord.ipynb` — Qwen × CORD v2.
4. `03_qwen_resume.ipynb` — Qwen × resume parsing.
5. `04_donut_sroie.ipynb` — Donut × SROIE.
6. `05_donut_cord.ipynb` — Donut × CORD v2.
7. `06_qwen_collected_receipts.ipynb` — Qwen × team-collected receipts.
8. `07_donut_collected_receipts.ipynb` — Donut × team-collected receipts.
9. `08_defense_results.ipynb` — future-ready defense and adaptive-attack analysis.
10. `09_report_figures.ipynb` — regenerates the cross-experiment report figures.

The notebooks use these authoritative CARC result roots: the main project for
clean baselines and Qwen resumes, the shared `qwen-receipt-runs` directory for
Qwen receipts, and Khalid's `kbali-donut-worktree` for Donut. Dataset notebooks
discover these paths automatically. If a run is moved, set the environment
variable printed by that notebook. Missing future results are reported clearly
rather than causing the notebook to fail; rerun all cells when the run completes.

## Run on CARC or the Mac

From the repository root:

```bash
jupyter lab midterm/
```

Use **Run All** in the desired notebook. Generated artifacts are deliberately
ignored by Git because they can always be recreated from the source results.
The notebook itself displays every table, so CSV files are only for later
copying or additional calculations.

## Rebuilding notebook files

Notebook JSON is generated from `build_notebooks.py` so all notebooks keep the
same short, readable structure. Run this only when changing notebook templates:

```bash
python midterm/build_notebooks.py
```

