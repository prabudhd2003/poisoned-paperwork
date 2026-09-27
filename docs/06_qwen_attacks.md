# 06 - Main Qwen full-page attacks

## GPU stage and analysis notebook

- Stage command: `qwen_attack`
- Implementation: `src/stages/qwen_attack.py`
- Active config: `configs/qwen_attack/active.json`
- CPU analysis: `notebooks/05_attack_analysis.ipynb`

## Purpose

Run the main white-box targeted-attack experiment against
Qwen2.5-VL-3B-Instruct. The primary study is receipt-total extraction on SROIE
and CORD. Resume degree extraction is the secondary case study and begins only
after the receipt pipeline is stable.

**GPU required:** yes, on CARC. This is the most memory- and time-intensive
notebook, so checkpointing and five-worker sharding are mandatory.

## Questions answered

1. How does targeted attack success change at epsilon 2, 4, 8, and 16?
2. What perceptibility cost accompanies each attack strength?
3. How often does success survive lossless save/reload?
4. Does vulnerability extend from receipt totals to resume degrees?
5. How expensive is Qwen attack optimization per document?

## Inputs

```text
data/processed/sroie/{validation,test}/
data/processed/cord_v2/{validation,test}/
data/processed/resume_parsing_vision/{validation,test}/
outputs/baseline/<frozen_qwen_run>/merged/predictions.parquet
outputs/baseline/<frozen_qwen_run>/merged/clean_correct_ids.txt
outputs/donut_attack/<frozen_validation_run>/merged/summary.json
```

Attack only records satisfying:

- `usable=True`;
- `attack_eligible=True`;
- Qwen `clean_correct=True` under the exact frozen baseline config;
- split equals the run's declared split.

Use validation for implementation and hyperparameter decisions. Test attacks
start only after the attack config is frozen and marked `frozen_for_test=true`.

## Model and prompt

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Prompt, output normalization, image limits, generation settings, and
  precision: load unchanged from the frozen `baseline` Qwen config.
- Freeze every model parameter. Only uploaded-image pixels are optimized.
- Do not use quantized weights for gradient attacks.

The validation attack is invalid if its baseline prediction differs from the
stored `baseline` clean prediction. Record and stop on such configuration
drift rather than attacking under a silently changed pipeline.

## Attack definition

### Objective

Create the exact chat template used at inference, append the complete desired
assistant answer, and compute teacher-forced target-token cross-entropy. Mask
system, user, visual-placeholder, and padding tokens so only assistant target
tokens contribute to the loss.

For every checkpoint, record:

- target-token loss and mean target-token log probability;
- greedy decoded answer under the frozen inference settings;
- normalized answer and exact-target status;
- observed pixel-space L-infinity distance;
- gradient norm and peak GPU memory.

### Pixel space and differentiable preprocessing

The threat model changes the uploaded RGB image, not an arbitrary normalized
embedding. Maintain the attack variable in original RGB pixel space scaled to
`[0,1]`. Projection uses epsilon divided by 255.

Qwen's normal image processor may pass through PIL/NumPy operations that break
gradients. If a differentiable resize/normalize path is implemented, it must:

1. reproduce the frozen official processor's output dimensions;
2. preserve aspect ratio and interpolation choices;
3. match official processed tensors within a documented numeric tolerance;
4. remain connected to the original image tensor;
5. be tested on at least 20 varied receipt dimensions;
6. use the real official pipeline again for final saved-image verification.

Do not optimize only a post-resize tensor and call it an uploaded-image attack
unless an inverse mapping to a bounded original-resolution PNG is implemented
and verified.

### Initial frozen candidate grid

```text
epsilon values: 2, 4, 8, 16 on the 0-255 scale
iterations: 100
step size: inherit the validated Donut rule initially
initialization: clean image
projection: per-channel L-infinity projection plus [0,1] clamp
checkpoint interval: 10 iterations
generation check interval: every 10 iterations and final iteration
early stopping: only after target survives an immediate save/reload check
seed: 566
```

If 100 Qwen steps are computationally infeasible, compare 25, 50, and 100 on a
fixed validation smoke subset, freeze one setting, and document the tradeoff.
Never shorten only unsuccessful test attacks after observing their outcomes.

### Receipt attack

Optimize the one receipt image. The target is the exact normalized value in
`attack_target`. Save as lossless RGB PNG at the original dimensions.

### Resume attack

Provide all retained pages in original order. Optimize all pages jointly under
the same per-pixel epsilon bound, save one PNG per page, and count document
success only from a new multi-page inference over the reloaded files. Report
L-infinity as the maximum across pages and LPIPS/SSIM both per page and averaged
per document.

If multi-page gradients exceed memory, choose and document one validation-only
fallback before test execution: gradient accumulation across pages or attack
only the page known to contain the highest degree. Do not change the policy per
test document.

## Stage implementation procedure

### 1. Reuse Donut-tested components

Reuse projection, checkpoint I/O, metrics, and saved-image verification from
the `donut_attack` stage. Add Qwen-specific tests rather than copying those
functions.

### 2. Qwen gradient smoke test

Use five SROIE and five CORD clean-correct validation receipts at epsilon 8.
Verify nonzero finite gradients, target-loss reduction, bounds, checkpoint
resume, and official-pipeline reload. Display peak memory before expanding.

### 3. Validation receipt grid

Run all clean-correct validation receipts at all four epsilons. Use these runs
to freeze step count, step size, early stopping, and any memory setting.

### 4. Resume case-study smoke and validation

After the receipt pipeline passes merge checks, run five single-page and five
multi-page resumes. Then run all eligible clean-correct validation resumes.

### 5. Freeze and run test once

Write a new immutable config with `frozen_for_test=true`, its validation run ID,
and a SHA-256 of the selected settings. Run the held-out test set without
further tuning. Any bug fix after inspecting test outputs requires a documented
new run; never replace results silently.

## Five-worker and timeout plan

All workers receive disjoint documents and run the complete epsilon grid for
each assigned document. Within a document, epsilon runs are sequential to limit
GPU memory. The assignment is based on sorted record IDs and is identical for
all epsilon settings.

Checkpoint each document/epsilon every 10 iterations:

```text
outputs/qwen_attack/<run_id>/shards/worker_00/
├── checkpoints/<experiment_key>/state.pt
├── images/<experiment_key>/page_000.png
└── records/<experiment_key>.json
```

The state contains all fields required in the shared execution contract plus
the exact processed image grid, chat template hash, target token IDs, and best
known attack tensor. A restarted job must verify all hashes before resuming.

Each teammate submits one fixed shard:

```bash
bash scripts/submit_stage.sh qwen_attack user1  # Prabudhd
bash scripts/submit_stage.sh qwen_attack user2  # Gary
bash scripts/submit_stage.sh qwen_attack user3  # Saaketh
bash scripts/submit_stage.sh qwen_attack user4  # Khalid
bash scripts/submit_stage.sh qwen_attack user5  # Shail
```

The launcher requests one A100 80 GB GPU, 8 CPUs, 64 GB system RAM, and twelve
hours for each worker. Five people can run concurrently. Nobody should edit the
stage implementation or active config during a frozen run.

## Exact outputs

```text
outputs/qwen_attack/<run_id>/merged/
├── attacks.parquet
├── attacks.csv
├── page_metrics.parquet
├── successful_attack_ids.txt
├── failures.csv
└── summary.json
```

Each attack record contains the `donut_attack` fields plus:

```text
document_page_count, attacked_page_count, target_token_ids_hash,
official_processor_shape, differentiable_processor_shape,
processor_max_abs_difference, mean_target_log_probability,
gradient_norm_final, peak_gpu_memory_mb
```

## Results shown in `05_attack_analysis.ipynb`

1. Conditional ASR versus epsilon, with separate SROIE, CORD, receipt-overall,
   and resume curves.
2. Harmful-output rate over the entire frozen split.
3. LPIPS, SSIM, and actual L-infinity versus epsilon.
4. In-memory versus saved/reloaded success.
5. Runtime, steps, and peak memory distributions.
6. Single-page versus multi-page resume results.
7. Representative successes and failures selected by a deterministic rule.

Primary result table:

| Task | Dataset | Epsilon | Clean-correct attacked | Reloaded successes | Conditional ASR | Harmful-output rate | Median LPIPS | Median SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Receipt total | SROIE | 2 | computed | computed | computed | computed | computed | computed |
| Receipt total | CORD | 2 | computed | computed | computed | computed | computed | computed |
| Resume degree | Resumes | 2 | computed | computed | computed | computed | computed | computed |

## Completion criteria

- Differentiable preprocessing is validated against the official processor.
- All attacked PNGs retain original dimensions and satisfy epsilon after reload.
- All clean-correct validation candidates have results for four epsilons.
- Test settings are frozen from validation before test inference.
- All five worker shards merge without duplicates or missing keys.
- Conditional ASR, harmful-output rate, LPIPS, SSIM, and runtime are reported.
- Receipt results are complete even if the secondary resume study is reduced.
