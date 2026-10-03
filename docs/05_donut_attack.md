# 05 - Donut targeted-attack reproduction

## GPU stage and analysis notebook

- Stage command: `donut_attack`
- Implementation: `src/stages/donut_attack.py`
- Active config: `configs/donut_attack/active.json`
- CPU analysis: `notebooks/05_attack_analysis.ipynb`

## Purpose

Validate the targeted image-optimization implementation on a smaller OCR-free
document model before attacking Qwen. The notebook follows the threat model in
the proposal and the methodology of *Counterfeit Answers: Adversarial Forgery
against OCR-Free Document Visual Question Answering*.

**GPU required:** yes, on CARC. This is a white-box gradient experiment.

## Questions answered

1. Can gradients through Donut change its answer toward a predetermined target?
2. Does projected optimization respect the requested pixel bound?
3. Does the target still appear after the image is saved and reloaded?
4. Which attack settings are stable enough to carry into the Qwen notebook?

This is an implementation-validation stage, not the main project result.

## Inputs

```text
data/processed/sroie/validation/
data/processed/cord_v2/validation/
outputs/baseline/<frozen_donut_run>/merged/predictions.parquet
outputs/baseline/<frozen_run>/merged/donut_clean_correct_ids.txt
```

Select only documents satisfying all of the following:

- receipt task;
- validation split during development;
- `usable=True`;
- `attack_eligible=True`;
- Donut `clean_correct=True` under the frozen `baseline` config.

Begin with a deterministic smoke subset of ten SROIE and ten CORD documents.
After implementation checks pass, run all clean-correct validation receipts.
Do not use the test set to choose PGD steps, step size, loss, or stopping rules.

## Model

- `naver-clova-ix/donut-base-finetuned-docvqa`
- Revision `b19d2e332684b0e2d35d9144ce34047767335cf8`
- Official processor and DocVQA prompt template.
- Freeze model weights; enable gradients only for the image variable.
- Start in FP32. Use FP16 only after matching FP32 gradients and attack outcomes
  on the fixed smoke subset.
- Do not quantize the model.

## Attack definition

Implement targeted L-infinity PGD over the model input image.

### Objective

Use teacher forcing on the complete adversarial target token sequence. Minimize
target-token cross-entropy while masking prompt and padding tokens. The target
for each receipt is the precomputed `attack_target`, not a value invented in
this notebook.

Log at every checkpoint:

- total target-token loss;
- mean target-token log probability;
- greedy decoded prediction;
- whether the exact target is produced;
- current L-infinity distance in 0-255 units.

### Initial validation configuration

```text
epsilon values: 2, 4, 8, 16 on the 0-255 scale
iterations: 100
step size: 1/255 for eps 2 and 4; 2/255 for eps 8 and 16
initialization: clean image for the primary run
projection: clamp to clean +/- epsilon, then clamp to [0, 1]
checkpoint interval: every 10 iterations
early stopping: optional only after exact target survives save/reload
seed: 566
```

These are project defaults, not a claim that they are the only settings in the
reference paper. Compare them against the reference implementation or paper
before freezing the final reproduction config. Any validated change gets a new
config version and must be selected using validation data only.

### Processor boundary

Keep a clearly documented mapping between original RGB pixel space and Donut's
normalized tensor space. Projection and reported epsilon must refer to the
original 0-255 RGB image. Do not accidentally interpret epsilon in normalized
model space.

## Stage implementation procedure

### 1. Attack unit tests on CPU/GPU

Before a full run, verify:

1. Epsilon zero gives exactly the clean image and clean prediction.
2. One gradient step produces finite, nonzero image gradients.
3. Model parameters receive no gradients and are never updated.
4. Projection never exceeds epsilon beyond floating-point tolerance.
5. Pixel values remain in `[0,1]`.
6. Saving and reloading a zero-perturbation PNG preserves the prediction.
7. A resumed checkpoint produces the same result as an uninterrupted run.

### 2. Twenty-document smoke run

Run epsilon 8 on ten SROIE and ten CORD clean-correct validation examples.
Inspect loss curves and at least five clean/adversarial/difference triplets.
Confirm that difference visualizations are magnified for inspection and never
mistaken for the actual perturbation.

### 3. Validation sweep

Run all four epsilon values on all eligible validation receipts. Use the same
attack settings for SROIE and CORD unless a dataset-specific difference is
approved and recorded before test evaluation.

### 4. Saved-image verification

For every completed attack:

1. Convert the projected tensor back to an RGB image without changing size.
2. Save losslessly as PNG.
3. Record SHA-256 and file dimensions.
4. Release the in-memory tensor.
5. Reload the PNG from disk.
6. Run the frozen Donut inference pipeline.
7. Count success only from this reloaded prediction.

### 5. Perceptibility metrics

Compute L-infinity, LPIPS, and SSIM between clean and saved/reloaded attacked
images. LPIPS inputs must be scaled exactly as required by the library. Record
the SSIM channel-axis and data-range settings in the config.

## Five-worker and checkpoint plan

All five workers run the same epsilon grid on disjoint document shards. This is
better than assigning one epsilon per person because it balances easy and hard
examples and avoids one missing person leaving an entire epsilon unfinished.

For each document and epsilon, use this directory:

```text
outputs/donut_attack/<run_id>/shards/worker_00/
├── checkpoints/<experiment_key>/state.pt
├── images/<experiment_key>.png
└── records/<experiment_key>.json
```

`state.pt` must contain the current adversarial tensor, clean tensor hash,
iteration, optimizer or momentum state if used, best loss, RNG states, config
hash, and model revision. Save every 10 iterations using a temporary file and
atomic rename. On CARC timeout, restart the same worker; it skips final records
and resumes partial attack states.

Each teammate submits the same stage with their fixed identity:

```bash
bash scripts/submit_stage.sh donut_attack user1  # Prabudhd
bash scripts/submit_stage.sh donut_attack user2  # Gary
bash scripts/submit_stage.sh donut_attack user3  # Saaketh
bash scripts/submit_stage.sh donut_attack user4  # Khalid
bash scripts/submit_stage.sh donut_attack user5  # Shail
```

The launcher requests one L40S 48 GB GPU, 8 CPUs, 48 GB system RAM, and eight
hours for each worker. The same command resumes completed and partial work.

## Exact outputs

```text
outputs/donut_attack/<run_id>/merged/
├── attacks.parquet
├── attacks.csv
├── failures.csv
├── successful_attack_ids.txt
└── summary.json
```

Each attack record includes baseline provenance plus:

```text
epsilon_255, step_size, max_steps, completed_steps, clean_prediction,
target_text, final_raw_response, final_prediction_normalized,
targeted_success_memory, targeted_success_reloaded, linf_255, lpips, ssim,
clean_image_sha256, attacked_image_sha256, attack_runtime_seconds,
final_target_loss, early_stop_reason, checkpoint_resumed
```

## Results shown in `05_attack_analysis.ipynb`

1. Conditional ASR by epsilon and dataset, using reloaded images.
2. Harmful-output rate by epsilon and dataset.
3. L-infinity assertion table: requested versus observed maximum.
4. LPIPS and SSIM distributions by epsilon.
5. Median optimization steps and runtime.
6. In-memory success versus saved/reloaded success.
7. Representative successful and failed examples with clean image, attacked
   image, and magnified absolute difference.

Expected summary layout:

| Dataset | Epsilon | Attempted | Reloaded successes | ASR | Median LPIPS | Median SSIM |
|---|---:|---:|---:|---:|---:|---:|
| SROIE | 2 | computed | computed | computed | computed | computed |
| CORD | 2 | computed | computed | computed | computed | computed |
| ... | ... | ... | ... | ... | ... | ... |

## Completion criteria

- All attack-unit tests pass.
- At least one nonzero gradient is demonstrated and recorded.
- Every final image satisfies the requested bound after reload.
- Every eligible validation record has one result for each frozen epsilon.
- Interrupted-run equivalence is tested.
- In-memory and reloaded success are reported separately.
- The Donut configuration to be reused by the `qwen_attack` stage is frozen.
- Zero attack success is reported honestly if the implementation checks pass;
  never tune on test examples simply to force a positive result.
