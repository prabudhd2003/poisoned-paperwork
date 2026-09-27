# 10 - Final results, tables, and figures

## Notebook

`notebooks/10_results_and_figures.ipynb`

## Purpose

Perform the final read-only merge and statistical analysis across notebooks
04-09. Create the authoritative tables and figures for the report and poster.
This notebook does not load a vision-language model and must never rerun or
silently repair an experiment.

**GPU required:** no. Run on a CARC CPU node or locally after copying the merged
result files. Do not consume a GPU allocation for this notebook.

## Questions answered

1. What is clean model accuracy on each task and dataset?
2. How does targeted Qwen success change with epsilon and perceptibility?
3. How much do attacks transfer or survive digital and physical processing?
4. Does EOT improve patch robustness?
5. How effective and costly are the three defenses, including adaptive attacks?
6. Which proposal claims are supported, unsupported, or inconclusive?

## Inputs

Use only completed merged outputs:

```text
outputs/04_clean_baselines/<frozen_run>/merged/
outputs/05_donut_attack/<frozen_run>/merged/
outputs/06_qwen_attacks/<frozen_run>/merged/
outputs/07_transfer_robustness/<frozen_run>/merged/
outputs/08_eot_patch/<frozen_run>/merged/
outputs/09_defenses/<frozen_run>/merged/
```

Create one tracked registry before analysis:

```text
configs/10_results/final_run_registry.json
```

It maps each stage to exactly one run ID and config SHA-256. Notebook 10 must
refuse ambiguous “latest run” discovery. Updating the registry requires an
explicit Git change.

## Read-only rule

Never modify files under `outputs/`. If a schema, missing-record, or hash check
fails, stop and fix or rerun the producing notebook. Do not drop failures,
duplicates, or inconvenient examples inside the plotting code.

## Notebook procedure

### 1. Provenance audit

Load all stage configs and verify:

- expected model IDs and revisions;
- expected dataset split and counts;
- frozen prompt and processor hashes;
- config hashes match result records;
- all five worker completion markers exist for GPU stages;
- no duplicate experiment keys;
- no unreported failures;
- source image and attacked image hashes resolve;
- test configurations point to validation runs used for selection.

Write a compact provenance table with Git commit, run ID, date, model revision,
split, and number of records for every stage.

### 2. Build analysis tables

Create normalized analysis dataframes with one row per valid statistical unit:

- `clean_predictions` - one row per document/model;
- `full_page_attacks` - one row per document/epsilon;
- `transform_trials` - one row per document/attack/transform/replicate;
- `patch_attacks` - one row per document/EOT condition;
- `physical_trials` - one row per printed document;
- `defense_trials` - one row per document/attack/defense;
- `adaptive_attacks` - one row per document/defense/epsilon.

Join using `record_id` and explicit experiment keys. Assert that no join changes
row counts unexpectedly.

### 3. Statistical policy

- Report integer numerators and denominators beside every percentage.
- Use 95% document-level bootstrap confidence intervals with 10,000 resamples
  and seed 566 for accuracy, ASR, transfer, and defense rates.
- Resample source documents, not transformation replicates.
- For the small physical pilot, also show exact counts and Wilson intervals.
- Report median and interquartile range for LPIPS, SSIM, runtime, and latency;
  include p95 for operational latency.
- Treat SROIE, CORD, and resumes separately before showing pooled results.
- Do not claim statistical significance from overlapping confidence intervals
  alone. Add a paired document-level comparison only when needed and declared.

### 4. Required final tables

1. **Data and eligibility:** raw, usable, clean-correct, and attacked counts.
2. **Clean baselines:** exact-match accuracy by model/dataset/task.
3. **Donut reproduction:** ASR and perceptibility by epsilon.
4. **Qwen full-page attacks:** conditional ASR, harmful-output rate, LPIPS,
   SSIM, and runtime by epsilon.
5. **Transfer and robustness:** Qwen survival and InternVL transfer by
   transformation.
6. **Patch and physical pilot:** non-EOT, EOT, and physical survival.
7. **Predictors:** AUROC and Brier score versus constant baselines.
8. **Defenses:** clean accuracy, non-adaptive ASR, adaptive ASR, latency,
   routing fraction, and additional passes.
9. **Failures and exclusions:** every excluded, failed, or incomplete category.

### 5. Required final figures

1. Clean exact-match accuracy with confidence intervals.
2. Qwen conditional ASR versus epsilon, separated by dataset/task.
3. LPIPS and SSIM versus epsilon.
4. Transformation-survival heatmap.
5. Direct InternVL transfer versus source epsilon.
6. Non-EOT versus EOT patch robustness.
7. Physical predictor ROC and calibration, clearly labeled as a pilot.
8. Defense tradeoff: adaptive ASR versus clean accuracy.
9. Defense cost: latency and additional VLM passes.
10. A small qualitative panel of deterministically selected successes and
    failures; never cherry-pick without stating the selection rule.

Every figure must have readable labels, units, denominator information in the
caption, a colorblind-safe palette, and both PNG and PDF exports.

### 6. Proposal-question matrix

End the notebook with a table mapping each research question to its evidence:

| Research question | Primary metric | Table/figure | Conclusion |
|---|---|---|---|
| Effect of perturbation budget | Conditional ASR | assigned output | supported / unsupported / inconclusive |
| Predict transfer/survival | AUROC, Brier | assigned output | ... |
| Lightweight detector/defense | adaptive ASR, clean accuracy, latency | assigned output | ... |
| Resume vulnerability | resume ASR | assigned output | ... |

Conclusions must be generated only after metrics are computed. Do not place an
expected direction into analysis code.

## Five-person division of work

This notebook is CPU-only, but the team can prepare independent sections after
the run registry and shared plotting style are frozen:

- Person 0: provenance, data counts, clean baselines, and failure appendix.
- Person 1: Donut and Qwen full-page attack tables/figures.
- Person 2: transfer and digital-robustness tables/figures.
- Person 3: EOT patch, physical pilot, and predictor tables/figures.
- Person 4: defenses, adaptive attacks, and cost tables/figures.

Each person writes derived files into a separate temporary subdirectory. The
stage owner performs the final notebook merge and applies the shared style.
Only one person edits the canonical notebook at a time to avoid Git conflicts.

## Exact outputs

Large and intermediate outputs remain gitignored:

```text
results/10_final/<run_id>/
├── analysis_tables/*.parquet
├── bootstrap_samples/
├── validation_report.json
└── complete_results.xlsx
```

Small final artifacts are copied to the tracked report directory:

```text
reports/final/
├── tables/*.csv
├── figures/*.png
├── figures/*.pdf
├── captions.md
├── provenance.csv
└── project_summary.md
```

`project_summary.md` contains the final numerators, denominators, metrics,
limitations, and one-paragraph answer to every research question. It is the
single source used when writing the paper and poster.

## Results shown in the notebook

The notebook displays all required final tables and low-resolution previews of
all figures, followed by:

- provenance and completeness status;
- primary findings;
- negative or inconclusive findings;
- physical-pilot limitations;
- compute and timeout limitations;
- deviations from the submitted proposal, including the resume dataset change.

## Completion criteria

- The frozen run registry identifies one complete run per stage.
- Every GPU-stage output passes completeness, duplicate, and provenance checks.
- All percentages include integer numerators and denominators.
- Confidence intervals resample at the document level.
- All proposal metrics and questions appear in a final table or figure.
- Both non-adaptive and adaptive defense results are included.
- Final files are exported to `reports/final/` and are reproducible from the
  registry without loading a VLM.
- Limitations and deviations are explicit rather than hidden.
