# CARC terminal workflow from 100 receipts to final results

All model inference, attacks, transformations, and defenses run from the CARC
terminal on L40S GPUs. Notebooks are read-only analysis artifacts and are never
used to submit or execute an experiment.

## 0. One-time setup

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull --ff-only origin main
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
git status --short --branch
```

The status must be clean before a job is submitted. Every command below is
restart-safe: submitting the same user/config again skips completed JSON records
and continues its assigned shard.

## 1. Freeze the 100 collected PNG receipts

The raw files belong in `collected receipts/`; labels belong in
`annotations/collected_receipt_labels.csv`. There must be exactly 100 unique
rows with `status=verified`. Inputs must be PNG.

```bash
python scripts/freeze_collected_external.py --expected 100
```

This creates the immutable processed PNGs, SHA256 manifest, decimal clean
answers, and the prospectively defined target (exactly 10× the clean total).
It does not run a model.

## 2. Clean Qwen and Donut baselines on the new receipts

```bash
python scripts/prepare_stage_config.py baseline_collected
CONFIG=outputs/runtime_configs/baseline_collected.json
for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh baseline "$USER" "$CONFIG"
done
```

Check progress without changing anything:

```bash
python scripts/stage_status.py baseline baseline_collected_external_v1
```

Do not attack baseline mistakes. The attack stages automatically select each
model's own clean-correct subset.

## 3. Full-page collected-receipt attacks

After the baseline is merged:

```bash
python scripts/prepare_stage_config.py qwen_collected
python scripts/prepare_stage_config.py donut_collected

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh qwen_attack "$USER" outputs/runtime_configs/qwen_collected.json
done

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh donut_attack "$USER" outputs/runtime_configs/donut_collected.json
done
```

Status:

```bash
python scripts/stage_status.py qwen_attack qwen_collected_external_v1
python scripts/stage_status.py donut_attack donut_collected_external_v1
```

If a shard times out or runs out of memory, rerun only that user's identical
submission command. Never delete the shard; saved records/checkpoints are the
resume mechanism.

## 4. Localized patch experiment (non-EOT control versus EOT)

This is a separate, visible threat model and must not be pooled with full-page
L-infinity ASR.

```bash
python scripts/prepare_stage_config.py eot_patch
for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh eot_patch "$USER" outputs/runtime_configs/eot_patch.json
done
python scripts/stage_status.py eot_patch eot_patch_collected_v1
```

The experiment changes only a deterministic 12.5%-of-short-edge corner patch,
compares equal-step no-EOT and EOT optimization, and evaluates real JPEG,
resize, blur, and brightness survival.

## 5. Black-box transfer and digital robustness

First ensure the existing public resume and receipt Qwen runs have strict
merged attack tables. Typical paths are:

```bash
RESUME=outputs/qwen_attack/qwen_resume_validation_v1/merged/attacks.parquet
RECEIPT=/project2/yzhao010_1531/qwen-receipt-runs/qwen_attack/qwen_receipt_validation_v1/merged/attacks.parquet
COLLECTED=outputs/qwen_attack/qwen_collected_external_v1/merged/records.parquet

python scripts/prepare_stage_config.py transfer \
  --source resume="$RESUME" \
  --source receipt="$RECEIPT" \
  --source collected="$COLLECTED"

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh transfer_robustness "$USER" outputs/runtime_configs/transfer.json
done
python scripts/stage_status.py transfer_robustness transfer_robustness_validation_v1
```

The stage deterministically samples 20 documents per dataset, evaluates every
epsilon for those documents, uses all 11 frozen processing conditions, and
loads Qwen and InternVL one at a time. InternVL is black-box evaluation only;
its output never modifies an attack.

## 6. Defenses and adaptive evaluation

### 6a. Full public-Qwen defense evaluation

This is the frozen final protocol for SROIE, CORD v2, and resumes. It uses all
clean-correct documents and every saved Qwen attack at epsilon 4, 8, and 16.
Epsilon 2 remains in the attack study as a negative control, but is not sent
through the expensive defense ensembles because it produced zero targeted
successes on every public dataset.

```bash
RESUME=outputs/qwen_attack/qwen_resume_validation_v1/merged/attacks.parquet
RECEIPT=/project2/yzhao010_1531/qwen-receipt-runs/qwen_attack/qwen_receipt_validation_v1/merged/attacks.parquet

python scripts/prepare_stage_config.py defenses_public \
  --source resume="$RESUME" \
  --source receipt="$RECEIPT" \
  --output outputs/runtime_configs/defenses_qwen_public_final_v1.json

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh defenses "$USER" \
    outputs/runtime_configs/defenses_qwen_public_final_v1.json
done

python scripts/stage_status.py defenses defenses_qwen_public_final_v1
```

The public run evaluates JPEG recompression, randomized resize voting,
selective visual multi-view consistency, and the project-specific cross-view /
cross-prompt semantic-consistency defense. The already saved official Qwen
predictions are the no-defense control; they are not recomputed as a fifth
"defense." The same frozen transformations, prompts, voting thresholds, and
metrics will later be applied to the 100 collected receipts.

Primary reporting includes post-defense ASR over every attempt and recovery
rate over attacks that originally achieved their exact target. Clean accuracy,
coverage, abstention, agreement, entropy, runtime, and added Qwen passes prevent
a defense from appearing effective merely by rejecting everything.

### 6b. Collected-receipt and adaptive evaluation

Use the same exact frozen attack tables:

```bash
python scripts/prepare_stage_config.py defenses \
  --source resume="$RESUME" \
  --source receipt="$RECEIPT" \
  --source collected="$COLLECTED"

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh defenses "$USER" outputs/runtime_configs/defenses.json
done
python scripts/stage_status.py defenses defenses_validation_v1
```

The defense conditions are:

1. JPEG recompression;
2. randomized resize majority vote;
3. selective multi-view consistency over identity/JPEG/resize/blur/brightness;
4. cross-view and cross-prompt semantic consistency.

The fourth defense is the project-specific contribution. It reports answer
agreement and entropy, abstains when fewer than three of five views agree, and
records clean accuracy, coverage, ASR, latency, and extra model passes. A
white-box EOT/BPDA attack then optimizes against all five views on a frozen
10-document collected-receipt subset at epsilon 8 and 16. This prevents a
misleading claim based only on attacks that did not know the defense.

Defended audit images are stored automatically under:

```text
outputs/defenses/defenses_validation_v1/shards/worker_XX/defended_images/
├── clean/
├── attacked/
└── adaptive/
```

Receipt and resume records remain distinguishable by `dataset_name` and
`record_id`; they are never mixed in reported denominators.

## 7. Analysis only (no GPU execution)

After each strict merge exists, use these notebooks locally or through VS Code
connected to CARC:

- `07_collected_external_analysis.ipynb`: external baseline, Qwen/Donut ASR,
  LPIPS/SSIM, and one-at-a-time high-resolution image inspection.
- `08_transfer_robustness_analysis.ipynb`: clean transformation accuracy,
  Qwen survival, and InternVL direct transfer.
- `09_eot_patch_analysis.ipynb`: no-EOT versus EOT ASR and survival.
- `10_defense_analysis.ipynb`: clean utility, coverage, abstention, cost,
  non-adaptive ASR, and adaptive ASR.
- `11_final_results.ipynb`: report-ready final tables.

If analyzing on the Mac, copy `outputs/` with `rsync` or `scp`; experiment
outputs are intentionally not pushed through Git.

## 8. Completion checklist

- Every stage has five `WORKER_COMPLETE.json` markers and a merged summary.
- Baselines report clean accuracy and parse failures for both models.
- ASR denominators contain only clean-correct examples and show exact N.
- Epsilon, LPIPS, SSIM, failures, and runtime appear beside ASR.
- Transfer results show source-success and all-attempted denominators.
- Defenses show clean accuracy, coverage/abstention, latency, and compute.
- Adaptive defense results are separate from non-adaptive results.
- External collected receipts remain a separately labeled external test.
- Visible-patch results remain separate from imperceptible full-page attacks.

## L40S budget estimate

The remaining stages are expected to consume roughly 55-85 aggregate L40S
GPU-hours, depending mainly on how many collected receipts are clean-correct
and how often attacks stop early. Five workers reduce wall-clock time but do
not reduce aggregate GPU-hours. Time limits are safety ceilings, not expected
usage: baseline 4 hours/job, collected Qwen 8, Donut 8, transfer 8, patch 8,
and defenses 8. Restarting an incomplete shard resumes saved work.
