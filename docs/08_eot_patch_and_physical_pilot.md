# 08 - EOT patch, physical pilot, and survival predictors

## GPU stage and analysis notebook

- Stage command: `eot_patch`
- Implementation: `src/stages/eot_patch.py`
- Active config: `configs/eot_patch/active.json`
- CPU analysis: `notebooks/05_attack_analysis.ipynb`

## Purpose

Evaluate a second, explicitly visible threat model: a small margin patch
optimized with expectation over transformation (EOT). Use a fixed subset of
patch attacks for a print-and-photograph pilot, then test whether inexpensive
digital features predict InternVL transfer and physical survival.

**GPU required:** yes, on CARC, for patch optimization and model inference.
Printing and photographing are manual. Logistic-regression fitting and result
merging are CPU tasks.

## Questions answered

1. Can a localized patch produce the exact target under realistic processing?
2. Does EOT improve survival through resizing, JPEG, blur, and brightness?
3. Which digitally successful patches remain successful after printing and
   photographing?
4. Can Qwen target likelihood and digital transformation survival predict
   InternVL transfer or physical survival?

## Inputs

```text
data/processed/sroie/{validation,test}/
data/processed/cord_v2/{validation,test}/
outputs/baseline/<qwen_run>/merged/predictions.parquet
outputs/qwen_attack/<qwen_run>/merged/attacks.parquet
outputs/transfer_robustness/<run_id>/merged/document_features.parquet
outputs/transfer_robustness/<run_id>/merged/direct_transfer.parquet
```

The primary patch study uses receipt documents only. Select usable,
attack-eligible, Qwen-clean-correct receipts. Resume patches are an optional
extension only after the receipt and physical pipelines finish.

## Model

- `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Same frozen prompt and inference pipeline as the `baseline` and
  `qwen_attack` stages.
- Freeze model weights; optimize patch pixels only.

InternVL labels are loaded from `transfer_robustness`. Do not optimize a patch using
InternVL feedback.

## Patch threat model

This is separate from the imperceptible full-page attack. A patch may be
visible, so never combine its success rate with the L-infinity PGD ASR.

### Patch geometry and placement

Start with one square patch whose side is 12.5% of the image's shorter side.
Choose one corner deterministically:

1. Define four candidate corner regions with a 2% inset.
2. Convert each candidate region to grayscale.
3. Select the region with the highest mean brightness; break ties in the order
   bottom-right, top-right, bottom-left, top-left.
4. Record placement and ensure the patch does not extend outside the image.

The patch is clipped to valid RGB values but is not constrained by the
full-page epsilon. Report its image-area percentage. If the patch obscures the
printed total, mark the example invalid rather than counting it as an attack.

### EOT distribution

At every optimization step, sample transformations from this frozen candidate
distribution:

```text
resize scale: Uniform(0.70, 1.00), then return to original dimensions
JPEG quality: Uniform integer [70, 95] using a differentiable approximation
Gaussian blur sigma: Uniform(0.0, 1.0)
brightness multiplier: Uniform(0.85, 1.15)
translation: up to +/- 2% of image width and height
EOT samples per step: 4 initially
iterations: 200 initially
seed: 566 plus deterministic experiment-key offset
```

Use the real Pillow JPEG/resize/blur/brightness pipeline for final evaluation.
The differentiable approximation exists only during optimization and must be
compared with the real pipeline on a smoke subset.

### Objective

Minimize expected target-token cross-entropy across EOT samples. Track the
untransformed exact target and exact target under a fixed validation transform
set. Early stopping is allowed only when the target succeeds on the original
PNG and at least 80% of the fixed transform set.

## Stage implementation procedure

### 1. Patch and EOT unit tests

- Pixels outside the patch mask remain bit-identical to the clean image.
- Patch placement and area match the recorded geometry.
- EOT samples are reproducible from the saved RNG state.
- Gradients pass through every differentiable approximation.
- The real transformed image remains valid RGB and retains expected dimensions.
- A resumed attack matches an uninterrupted attack with the same seed.

### 2. Non-EOT control

For the smoke subset, optimize one patch without EOT using the same geometry,
steps, and target. This is the control needed to quantify whether EOT actually
improves robustness.

### 3. EOT patch validation run

Run non-EOT and EOT patches on a fixed clean-correct validation subset from
both receipt datasets. If compute permits, expand to all clean-correct
validation receipts. Freeze patch size, EOT distribution, steps, and selection
rules before test evaluation.

### 4. Digital robustness evaluation

Use the real transformation suite from `transfer_robustness`. Store success per
transformation and report the fraction of conditions and replicates that
preserve the exact target.

### 5. Physical-pilot selection

Select 50 EOT patch examples before observing physical outcomes:

- 40 validation receipts: 20 SROIE and 20 CORD;
- 10 held-out test receipts: 5 SROIE and 5 CORD;
- stratify within each dataset by digital robustness and target confidence;
- keep all variants of a document in one physical-predictor split.

Within the 40 validation examples, assign 30 to predictor training and 10 to
predictor validation. The 10 held-out test examples are opened only after the
predictor is frozen. If only 25 physical examples are feasible, preserve the
same proportions and report the reduced statistical power.

Save the selected IDs and split assignment before printing:

```text
outputs/eot_patch/<run_id>/physical/physical_pilot_manifest.csv
```

### 6. Print-and-photograph protocol

Each of five teammates handles ten manifest rows. Use the assigned worker ID
and do not substitute examples.

For every example:

1. Print at 100% scale on white letter-size paper; no “fit to page.”
2. Record printer model, print setting, paper type, and date.
3. Place the page flat under ordinary indoor lighting.
4. Photograph the full page using the rear phone camera, without digital zoom.
5. Capture one near-frontal image from approximately 40-60 cm away.
6. Do not crop or enhance the photo manually.
7. Record phone model, native dimensions, and whether flash was used.
8. Save the original photo with its metadata intact.

Use filenames:

```text
physical_<record_id>__worker-<id>__capture-01.<original_extension>
```

Store files under:

```text
outputs/eot_patch/<run_id>/physical/raw/worker_00/
```

The inference pipeline may auto-orient and crop the page using one frozen
procedure, but retain the raw photograph and record every processing step.

### 7. Physical inference

Run the frozen Qwen prompt on each processed photograph. The binary physical
survival label is true only when the normalized response exactly equals the
adversarial target. Also record whether the clean answer is recovered or a
third answer is produced.

### 8. Transfer and survival predictors

Fit separate scikit-learn logistic regressions for:

1. InternVL direct-transfer label from `transfer_robustness`.
2. Physical-survival label from this notebook.

Predictors use only Qwen-derived features available before the label:

- target likelihood and clean/target margin;
- transform success fraction;
- transform likelihood mean, minimum, and variance;
- epsilon or patch type;
- LPIPS and SSIM;
- page count and dataset indicator if predeclared.

Standardize continuous features using training documents only. Use disjoint
document splits and keep all attack variants of one document together. Compare
against a constant predictor equal to the training-set positive rate.

Report AUROC and Brier score with bootstrap intervals. With very few physical
labels, treat results as a pilot and show uncertainty rather than making a
strong generalization claim.

## Five-worker and checkpoint plan

GPU workers shard documents and run all patch conditions for their IDs.
Checkpoint patch state every 10 iterations, including the patch tensor, mask,
placement, RNG states, iteration, best robust success, and config hash.

The same worker IDs are used for the physical pilot: ten manifest rows per
person. This assignment is independent of which GPU worker originally created
the patch.

Each teammate submits one fixed GPU shard:

```bash
bash scripts/submit_stage.sh eot_patch user1  # Prabudhd
bash scripts/submit_stage.sh eot_patch user2  # Gary
bash scripts/submit_stage.sh eot_patch user3  # Saaketh
bash scripts/submit_stage.sh eot_patch user4  # Khalid
bash scripts/submit_stage.sh eot_patch user5  # Shail
```

The launcher requests one A100 80 GB GPU, 8 CPUs, 64 GB system RAM, and twelve
hours for each worker. Physical capture cannot be automated; the merge command
checks that every manifest row has a photo and metadata row before physical
inference begins.

## Exact outputs

```text
outputs/eot_patch/<run_id>/merged/
├── patch_attacks.parquet
├── transform_trials.parquet
├── physical_trials.parquet
├── transfer_predictor_predictions.parquet
├── physical_predictor_predictions.parquet
├── predictor_metrics.json
├── failures.csv
└── summary.json
```

Patch images live in worker shard directories. Physical photos and their
manifest stay in the physical directory. Record SHA-256 for every file.

## Results shown in `05_attack_analysis.ipynb`

1. Non-EOT versus EOT digital success under every transformation.
2. Patch success by dataset, patch area, and transformation.
3. Physical survival count and rate with confidence intervals.
4. InternVL and physical predictor AUROC and Brier scores versus the constant
   baseline.
5. Calibration plots for both predictors.
6. Failure breakdown: clean answer, target answer, other answer, parse failure.
7. Representative digital and physical images with recorded capture metadata.

## Completion criteria

- Patch-only modification tests pass.
- Non-EOT and EOT runs use identical selection and evaluation rules.
- Physical examples are selected before outcomes are known.
- All five team assignments are explicit and non-overlapping.
- Raw physical photos, metadata, and hashes are retained.
- Predictor splits are document-disjoint.
- AUROC and Brier score are compared with a constant baseline.
- Physical claims are labeled as a pilot with sample-size limitations.
