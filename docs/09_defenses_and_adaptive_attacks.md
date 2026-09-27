# 09 - Defenses and adaptive attacks

## Notebook

`notebooks/09_defenses_and_adaptive_attacks.ipynb`

## Purpose

Evaluate the proposal's three lightweight defense pipelines on clean and
attacked documents, then re-optimize attacks against each complete defended
pipeline. A defense is useful only if it lowers attack success without
unacceptable clean-accuracy or latency costs and remains meaningful against an
attacker who knows it.

**GPU required:** yes, on CARC, for Qwen inference, visual-feature extraction,
majority voting, and adaptive attacks. Logistic-regression training, threshold
selection, merging, and plots can run on CPU.

## Questions answered

1. How much do JPEG re-encoding and randomized resizing reduce attack success?
2. How much clean accuracy and latency do those defenses cost?
3. Can a lightweight detector route suspicious documents to deeper verification?
4. Do the defenses still work after attacks are optimized with full knowledge
   of preprocessing, randomness, detector, threshold, and routing behavior?

## Inputs

```text
data/processed/<dataset>/{train,validation,test}/
outputs/04_clean_baselines/<qwen_run>/merged/predictions.parquet
outputs/06_qwen_attacks/<run_id>/merged/attacks.parquet
outputs/07_transfer_robustness/<run_id>/merged/transformation_trials.parquet
outputs/08_eot_patch/<run_id>/merged/patch_attacks.parquet
```

Use clean images and both attack families:

- imperceptible full-page PGD;
- visible EOT patch attacks.

Keep all variants of one source document in the same detector split. Detector
training, threshold selection, and final testing must be document-disjoint.

## Model

- `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Frozen notebook-04 prompts and normalization.
- Frozen notebook-06 inference processor.
- No model fine-tuning.

## Defense D1 - JPEG re-encoding

Re-encode the incoming document as JPEG before Qwen inference.

Validation candidates are quality 95, 85, and 75. Select one quality using a
predeclared rule:

1. clean exact-match loss must be at most 2 absolute percentage points from the
   undefended baseline;
2. among settings satisfying that constraint, choose the one with the lowest
   attack success;
3. break ties in favor of higher quality.

Freeze the selected quality before test evaluation. For multi-page documents,
apply the same quality to every page.

## Defense D2 - Randomized resizing with majority vote

For each document, run five independently seeded passes:

1. sample scale uniformly from `[0.70, 1.00]`;
2. downscale using the frozen interpolation method;
3. restore the original dimensions;
4. run Qwen inference;
5. normalize each response.

The defended response is the strict majority normalized answer. If no answer
has at least three votes, return the untransformed base response and mark
`majority_tie=True`. Seeds are deterministic from `(record_id, defense_seed,
replicate)` so runs are reproducible.

Validation may compare 3, 5, and 7 votes. Freeze the smallest count within one
percentage point of the best validation robustness to control latency.

## Defense D3 - Detector-triggered verification

### Detector features

Extract frozen Qwen visual features from the same layer and pooling rule for
every image. The first implementation should mean-pool the final vision-encoder
token embeddings after the model's final visual normalization. Record layer
name, tensor shape, and pooling rule in the config.

For multi-page documents, mean-pool within each page and then mean across pages.
Do not use the model's predicted class, target label, or test outcome as a
detector feature.

### Detector

- Model: scikit-learn logistic regression.
- Inputs: standardized frozen visual features.
- Labels: clean versus adversarial.
- Training: document-disjoint train set; class-balanced weighting selected on
  validation only.
- Threshold: choose on validation to keep clean false-positive rate at or below
  5%; among eligible thresholds maximize adversarial detection rate.
- Baseline: constant score equal to training adversarial prevalence.

### Routing behavior

1. Run normal Qwen inference once.
2. Compute the detector score from frozen visual features.
3. If score is below threshold, return the normal answer.
4. If score is at or above threshold, run the selected JPEG transform plus the
   selected randomized-resize vote configuration.
5. Return the verification majority answer, using the same tie rule as D2.

Record the route, number of additional VLM passes, and total latency.

## Non-adaptive evaluation

Evaluate D1, D2, and D3 on:

- all clean validation/test documents;
- all completed full-page attacks from notebook 06;
- all completed patch attacks from notebook 08.

Report clean exact-match accuracy, conditional ASR, harmful-output rate, end-to-
end latency, and additional VLM passes for every defense. For D3 also report
detection rate, clean false-positive rate, and fraction routed.

## Adaptive attacks

The attacker knows the complete defense. Start from clean images and optimize a
new attack; do not merely re-evaluate the notebook-06 images.

### Adaptive attack on D1

Use a differentiable JPEG approximation during optimization and the exact real
JPEG re-encoding for periodic and final checks. Apply BPDA only if required and
document the forward and backward operations separately.

### Adaptive attack on D2

Use EOT over the frozen resize distribution. At every attack step, average the
target-token loss over the same number of transform samples declared in the
config. Final success requires the defended five-vote pipeline to return the
exact target.

### Adaptive attack on D3

Optimize a combined objective:

```text
target_token_loss + lambda * detector_evasion_loss
```

where detector evasion encourages the score below the frozen threshold. If the
attack still routes to verification, it must also survive the triggered JPEG
and randomized-vote pipeline. Use EOT for routing randomness. Choose `lambda`
on validation from a fixed candidate set and freeze it before test execution.

Never report a defense as robust based only on attacks created before the
defense was known.

## Notebook procedure

1. Validate feature extraction on clean and attacked copies of 20 documents.
2. Build immutable document-level detector splits.
3. Fit detector and select threshold using train/validation only.
4. Freeze D1, D2, and D3 configs.
5. Run non-adaptive clean and attack evaluation.
6. Run adaptive smoke attacks on five SROIE and five CORD validation receipts.
7. Check bounds, loss, detector score, route, and save/reload behavior.
8. Run all adaptive validation attacks at the predeclared epsilon subset. At
   minimum use epsilon 8; if compute permits, use all four epsilons.
9. Freeze adaptive settings and run held-out test once.

## Five-worker and checkpoint plan

Shard by source document, never by individual attack variant. Each worker owns
all clean, attack, defense, and adaptive variants for its documents. This keeps
detector splits and route comparisons aligned.

Non-adaptive inference checkpoints after every
`(record_id, attack_id, defense_id, replicate)`. Adaptive attacks checkpoint
every 10 iterations and additionally store detector score, current route, EOT
RNG states, and defense-config hash.

Worker cells call:

```python
run_defense_evaluation(worker_id=N, num_workers=5, config=CONFIG)
run_adaptive_attacks(worker_id=N, num_workers=5, config=CONFIG)
```

Run the first call before the second. Five teammates may run their assigned
shards concurrently on separate CARC GPU allocations.

## Exact outputs

```text
outputs/09_defenses/<run_id>/merged/
├── detector_features.parquet
├── detector_splits.csv
├── detector_predictions.parquet
├── defense_trials.parquet
├── adaptive_attacks.parquet
├── routing_and_latency.parquet
├── failures.csv
└── summary.json
```

Required defense-trial fields include:

```text
record_id, source_attack_id, attack_family, defense_id, defense_parameters,
split, clean_or_adversarial, detector_score, detector_flag,
routed_to_verification, vote_predictions, final_prediction,
clean_correct, adversarial_target_success, additional_vlm_passes,
end_to_end_latency_seconds
```

## Results shown in the notebook

1. Clean accuracy and attack success for no defense, D1, D2, and D3.
2. Non-adaptive versus adaptive ASR for each defense.
3. Detector AUROC, detection rate, clean false-positive rate, and calibration.
4. Fraction routed and average additional VLM passes.
5. Median and p95 end-to-end latency.
6. Defense results separated by dataset, epsilon, and attack family.

Primary table:

| Defense | Clean accuracy | Non-adaptive ASR | Adaptive ASR | Median latency | Routed fraction | Extra passes |
|---|---:|---:|---:|---:|---:|---:|
| None | computed | computed | computed | computed | 0 | 0 |
| JPEG | computed | computed | computed | computed | 1 | 0 |
| Resize vote | computed | computed | computed | computed | 1 | computed |
| Detector route | computed | computed | computed | computed | computed | computed |

## Completion criteria

- Three defense definitions and thresholds are frozen from validation.
- Detector splits are document-disjoint and all variants remain together.
- Every defense is evaluated on clean inputs and both attack families.
- Clean accuracy, ASR, latency, routing, and computation cost are all reported.
- Adaptive attacks know and optimize against the complete defended pipeline.
- Test evaluation does not change defense or attack hyperparameters.
- Failures are included in denominators or explicitly reported according to the
  frozen scoring policy.
