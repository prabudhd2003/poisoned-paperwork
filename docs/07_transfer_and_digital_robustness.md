# 07 - Transfer and digital robustness

## GPU stage and analysis notebook

- Stage command: `transfer_robustness`
- Implementation: `src/stages/transfer_robustness.py`
- Active config: `configs/transfer_robustness/active.json`
- CPU analysis: `notebooks/05_attack_analysis.ipynb`

## Purpose

Measure whether Qwen-optimized attacks transfer to InternVL and survive common
digital document-processing operations. Also create the transformation-based
features used later to predict model transfer and physical survival.

**GPU required:** yes, on CARC, for Qwen and InternVL inference. Image
transformations and final merging are CPU work.

No adversarial image is optimized in this stage. It evaluates the fixed outputs
of `qwen_attack`.

## Questions answered

1. Do Qwen attacks produce the exact target on InternVL without re-optimization?
2. Which attacks survive PNG, JPEG, resizing, blur, and brightness changes?
3. Does robustness increase with epsilon?
4. Can Qwen target likelihood and digital survival features predict transfer?
5. How much clean accuracy is lost under the same transformations?

## Inputs

```text
data/processed/<dataset>/<split>/
outputs/baseline/<frozen_qwen_run>/merged/predictions.parquet
outputs/qwen_attack/<frozen_qwen_run>/merged/attacks.parquet
outputs/qwen_attack/<frozen_qwen_run>/shards/*/images/
```

Include every completed Qwen attack, not only successful ones, so overall
transfer and survival denominators remain honest. Each attacked image hash must
match the `qwen_attack` result before evaluation.

Development and predictor selection use validation documents. Test results are
generated after all transformation and model settings are frozen.

## Models

### Source-model re-evaluation

- `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Load the identical frozen prompt, processor, generation, and normalization
  config from the `baseline` and `qwen_attack` stages.

### Transfer model

- `OpenGVLab/InternVL2_5-4B`
- Revision `2cf4a8158bbc40d35015e7c63b527890de4d27b3`
- BF16 if supported, otherwise FP16.
- Deterministic generation with `do_sample=False`, maximum 32 new tokens.
- Official dynamic image preprocessing from the pinned repository.
- Use the same natural-language task prompts as Qwen, adapted only to the
  model's required chat template.

InternVL is transfer-only. Never use its gradients, outputs, or target
likelihood to create or improve an adversarial image.

## Required transformation suite

Apply each operation to both clean and attacked images. Preserve a transform
record containing exact parameters and seed.

| Transform ID | Operation |
|---|---|
| `png_reload` | Save lossless PNG and reload; mandatory base condition |
| `jpeg_q95` | JPEG encode/decode at quality 95 |
| `jpeg_q85` | JPEG encode/decode at quality 85 |
| `jpeg_q75` | JPEG encode/decode at quality 75 |
| `resize_075` | Downscale to 75%, then return to original size |
| `resize_050` | Downscale to 50%, then return to original size |
| `blur_05` | Gaussian blur, sigma 0.5 |
| `blur_10` | Gaussian blur, sigma 1.0 |
| `brightness_090` | Multiply brightness by 0.90 with clipping |
| `brightness_110` | Multiply brightness by 1.10 with clipping |
| `common_pipeline` | JPEG q85 + resize factor 0.75 + blur 0.5 |

Use one frozen resampling method for resizing and record the Pillow version.
For multi-page resumes, apply the same condition to every page.

The validation summary may justify removing a redundant condition, but no new
condition may be added after inspecting test outcomes.

## Stage implementation procedure

### 1. Provenance and clean InternVL baseline

Before transfer evaluation, run InternVL on the unmodified validation and test
documents used in this stage. Record clean exact-match accuracy. This is needed
to distinguish failed transfer from a transfer model that cannot answer the
task at all.

### 2. Source-model consistency check

Re-run a deterministic sample of `qwen_attack` adversarial PNGs through Qwen.
The prediction must match the stored reloaded prediction. Stop on widespread
drift caused by a changed prompt, processor, or dependency.

### 3. Direct transfer

Run each original adversarial PNG through InternVL. Store raw text, normalized
prediction, exact target status, and runtime. Report:

- transfer among all attempted Qwen attacks;
- transfer among Qwen-successful attacks;
- transfer by epsilon, dataset, and task.

### 4. Transformation matrix

For each clean/adversarial pair, model, and transformation:

1. Generate the transformed image in memory from the frozen source file.
2. Save/reload when the transform represents a real file operation.
3. Run deterministic inference.
4. Store clean-target correctness and adversarial-target success.
5. Store target log probability when it can be computed consistently.
6. Delete disposable transformed images after their hash and metrics are saved,
   unless the run config requests retention for auditing.

### 5. Feature table for later predictors

Create one row per attacked document with features derived only from Qwen and
digital transforms:

- clean target log probability;
- adversarial target log probability before transformation;
- target log-probability margin over the clean answer;
- binary target success for every transform;
- fraction of transforms producing the target;
- mean, minimum, and variance of target log probability across transforms;
- epsilon, LPIPS, SSIM, dataset, and page count.

Attach labels separately:

- `internvl_transfer_label` from this notebook;
- `physical_survival_label` later from the `eot_patch` stage.

Do not fit the final transfer/physical predictor on test documents here.

## Five-worker plan

Shard by attacked `record_id`. Each worker evaluates all epsilons, transforms,
and intended models for its assigned documents. This keeps all variants of a
document in the same shard and prevents predictor leakage.

Because InternVL and Qwen cannot necessarily fit together, load one model at a
time:

1. Complete Qwen transformation inference and unload it.
2. Clear GPU cache.
3. Load InternVL and complete direct transfer plus selected transformation
   inference.

Checkpoint after every `(record_id, epsilon, transform, model)` key. Inference
records are small JSON files, so no work unit should wait until the end of a
document to save.

Each teammate submits one fixed shard:

```bash
bash scripts/submit_stage.sh transfer_robustness user1  # Prabudhd
bash scripts/submit_stage.sh transfer_robustness user2  # Gary
bash scripts/submit_stage.sh transfer_robustness user3  # Saaketh
bash scripts/submit_stage.sh transfer_robustness user4  # Khalid
bash scripts/submit_stage.sh transfer_robustness user5  # Shail
```

The launcher requests one L40S 48 GB GPU, 8 CPUs, 48 GB system RAM, and eight
hours for each worker.

## Exact outputs

```text
outputs/transfer_robustness/<run_id>/merged/
├── internvl_clean_predictions.parquet
├── direct_transfer.parquet
├── transformation_trials.parquet
├── document_features.parquet
├── failures.csv
└── summary.json
```

Required trial columns:

```text
record_id, dataset_name, task, split, epsilon_255, source_attack_success,
model_id, transform_id, transform_parameters_json, replicate, raw_response,
prediction_normalized, clean_target_correct, adversarial_target_success,
target_log_probability, runtime_seconds, input_sha256, output_sha256
```

## Results shown in `05_attack_analysis.ipynb`

1. InternVL clean exact-match accuracy by dataset and task.
2. Direct transfer rate by epsilon, dataset, and source-attack success.
3. Qwen attack survival heatmap across transformations and epsilon.
4. InternVL transfer after selected transformations.
5. Clean accuracy under every transformation.
6. Target likelihood before and after transformation.
7. Failure and parse-failure counts.

Primary robustness table:

| Transform | Model | Clean accuracy | Attack survival | N |
|---|---|---:|---:|---:|
| PNG reload | Qwen | computed | computed | computed |
| JPEG q85 | Qwen | computed | computed | computed |
| Resize 0.75 | Qwen | computed | computed | computed |
| Direct PNG | InternVL | computed | computed | computed |

## Completion criteria

- InternVL is never used to optimize attacks.
- All model and transform configs are frozen before test evaluation.
- Clean and attacked images undergo identical transformation definitions.
- Every expected trial key is present or listed as a failure.
- Transfer is reported with both required denominators.
- Clean accuracy loss is shown beside attack survival.
- `document_features.parquet` is ready for `eot_patch` predictor work.
