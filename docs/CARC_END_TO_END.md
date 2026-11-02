# CARC terminal workflow from 114 receipts to final results

All model inference, attacks, transformations, and defenses run from the CARC
terminal on L40S GPUs. Notebooks are read-only analysis artifacts and are never
used to submit or execute an experiment.

This is the canonical operational contract for humans and coding agents. Read
it completely before changing a launcher, runtime config, output contract, or
worker assignment.

## Worker and hardware contract

| User ID | Team member | Worker shard |
|---|---|---:|
| `user1` | Prabudhd | 0 |
| `user2` | Gary | 1 |
| `user3` | Saaketh | 2 |
| `user4` | Khalid | 3 |
| `user5` | Shail | 4 |

Production stages always use these five deterministic shards. All GPU jobs use
one L40S; do not change the GPU type between workers or add another GPU type as
a fallback. Runtime configurations belong under `outputs/runtime_configs/` so
their frozen source hashes do not dirty the tracked worktree.

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

Verify the exact environment before a production submission:

```bash
python - <<'PY'
import sys, torch, transformers, pandas, numpy, PIL, lpips
print("python:", sys.executable, sys.version.split()[0])
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("pillow:", PIL.__version__)
print("lpips: available")
PY
```

The launcher repeats these imports before calling `sbatch`, so a missing package
fails on the login node instead of after waiting for a GPU. Runtime config
preflight also validates source hashes, exact dataset counts, prompts, model
revision, epsilon grid, and method-specific settings.

Once any worker for a frozen run has been submitted, do not pull another commit
or edit tracked files in that checkout until every worker has started or the run
has been deliberately cancelled. Each worker records the Git commit it actually
used, and strict merge rejects workers from different commits. Generating an
ignored runtime config or reading logs/results is safe while jobs run.

## Monitoring, logs, and recovery

The submission command prints each job ID and exact log path. Check active jobs:

```bash
squeue -u "$USER" -o "%.18i %.36j %.10T %.12M %.12l %.24R"
```

Check a finished or disappeared job using its ID:

```bash
sacct -j JOB_ID \
  --format=JobID,JobName,State,Elapsed,Timelimit,ExitCode,MaxRSS
```

`COMPLETED` with exit code `0:0` means Slurm finished the script. `TIMEOUT`,
`OUT_OF_MEMORY`, or `FAILED` means the shard is incomplete even if some records
were saved. Inspect both streams:

```bash
tail -n 150 logs/JOB_NAME_JOB_ID.out
tail -n 150 logs/JOB_NAME_JOB_ID.err
```

Warnings about unauthenticated Hugging Face downloads or deprecated torchvision
weights are not failures by themselves. A Python traceback, CUDA OOM, missing
module, nonzero exit code, or missing worker marker requires action.

Use the read-only status command at any time:

```bash
python scripts/stage_status.py STAGE RUN_ID
```

A shard is complete only when its `WORKER_COMPLETE.json` exists. A run is
complete only when all five markers exist and the strict merged summary has
been written. For a timeout or transient failure, resubmit the identical stage,
user, and runtime-config command. Do not delete records, images, checkpoints,
assignments, or the run contract.

For CUDA fragmentation, the launch environment may use
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; this does not change the
scientific configuration. Persistent OOM on the same record requires a reviewed
code/config fix rather than silently dropping the example.

Shared directories must remain group-searchable and group-writable. Diagnose a
permission failure with `namei -l /absolute/path/to/file`. Do not recursively
change another team member's private directory. Copy only readable artifacts
into the main repository output tree, verify record/image counts and hashes,
and freeze the copied source path in the runtime config.

## 1. Freeze the 114 collected PNG receipts

The raw files belong in `collected receipts/`; labels belong in
`annotations/collected_receipt_labels.csv`. There must be exactly 114 unique
rows with `status=verified`. Inputs must be PNG.

```bash
python scripts/freeze_collected_external.py --expected 114
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
metrics will later be applied to the 114 collected receipts.

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

### 6c. Full defense-aware EOT/BPDA attack

This is the adaptive follow-up to the completed non-adaptive defense study. It
uses every clean-correct public document (96 CORD, 123 SROIE, and 74 resumes)
at epsilon 4, 8, and 16: 879 attacks total. Epsilon 2 is retained as the
already completed negative control because the original attack achieved 0/293
targeted successes at that budget.

The output remains a bounded PNG. During optimization, exact Pillow JPEG,
resize, blur, and brightness transforms run in the forward pass; BPDA provides
the backward approximation. EOT also samples the five frozen semantic prompts.
The final PNG is evaluated independently against the exact JPEG, resize-vote,
selective multi-view, and cross-view/cross-prompt defenses.

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
source scripts/activate_paperwork.sh

RESUME="$PWD/outputs/qwen_attack/qwen_resume_validation_v1/merged/attacks.parquet"
RECEIPT="$PWD/outputs/imported/qwen_receipt_validation_v1/merged/attacks_for_defenses.parquet"

python scripts/prepare_stage_config.py jpeg_eot \
  --source resume="$RESUME" \
  --source receipt="$RECEIPT" \
  --output outputs/runtime_configs/jpeg_eot_public_full_v1.json

python scripts/preflight_stage.py \
  --stage jpeg_eot_attack \
  --config outputs/runtime_configs/jpeg_eot_public_full_v1.json

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh jpeg_eot_attack "$USER" \
    outputs/runtime_configs/jpeg_eot_public_full_v1.json
done
```

The jobs are checkpointed every ten optimization steps. If a shard times out,
resubmit only the same user's identical command; it resumes instead of
discarding completed work. Read-only progress and final results are available
with:

```bash
python scripts/stage_status.py \
  jpeg_eot_attack qwen_jpeg_eot_public_full_v1

python -m json.tool \
  outputs/jpeg_eot_attack/qwen_jpeg_eot_public_full_v1/merged/jpeg_eot_summary.json
```

After the strict merge exists, run `midterm/10_adaptive_defense_attack.ipynb`
for the report table, adaptive-ASR figure, and native-resolution qualitative
example. This experiment tests whether the defenses remain useful against an
attacker who knows their transformations; it does not assume or guarantee that
every document will be successfully attacked.

### 6d. Full frequency-aware hard-view attack

This second adaptive run uses the identical 293 documents, targets, epsilon
values, defenses, and metrics. Its optimizer reweights sampling toward the
highest-loss views and mixes a smooth FFT low-frequency component into the
gradient and perturbation before projection. Its output directory and
checkpoint contracts are independent from the uniform-EOT run, so both can run
concurrently.

Use the same frozen `RESUME` and `RECEIPT` paths defined above:

```bash
python scripts/prepare_stage_config.py jpeg_eot_hardview \
  --source resume="$RESUME" \
  --source receipt="$RECEIPT" \
  --output outputs/runtime_configs/jpeg_eot_hardview_public_full_v1.json

python scripts/preflight_stage.py \
  --stage jpeg_eot_attack \
  --config outputs/runtime_configs/jpeg_eot_hardview_public_full_v1.json

for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh jpeg_eot_attack "$USER" \
    outputs/runtime_configs/jpeg_eot_hardview_public_full_v1.json
done
```

To submit both full methods together after both preflights pass:

```bash
for CONFIG in \
  outputs/runtime_configs/jpeg_eot_public_full_v1.json \
  outputs/runtime_configs/jpeg_eot_hardview_public_full_v1.json
do
  for USER in user1 user2 user3 user4 user5; do
    bash scripts/submit_stage.sh jpeg_eot_attack "$USER" "$CONFIG"
  done
done
```

CARC may queue rather than simultaneously run all ten jobs. That does not
affect the frozen experiment. Check each method independently:

```bash
python scripts/stage_status.py \
  jpeg_eot_attack qwen_jpeg_eot_public_full_v1

python scripts/stage_status.py \
  jpeg_eot_attack qwen_frequency_hardview_public_full_v1
```

After both strict merges exist, run
`midterm/10_adaptive_defense_attack.ipynb` for the matched comparison.

### 6e. Matched Donut defenses and adaptive attacks

Do not pull the Donut implementation commit into a checkout that still has
pending Qwen workers. Wait until those workers finish, then pull once and keep
the checkout fixed for all Donut workers.

The primary Donut standard-attack table contains epsilon 2, 4, 8, and 16.
Defenses and adaptive attacks use epsilon 4, 8, and 16, exactly matching Qwen.
The completed epsilon-32 Donut run is an archived high-budget stress test and
must not enter these runtime configurations or primary denominators.

Build the hash-verified Donut source table from Khalid's completed full runs:

```bash
python scripts/import_donut_public_results.py \
  --source-run /project2/yzhao010_1531/csci_699_new_arch/kbali-donut-eps2-4-worktree/outputs/donut_attack/donut_paper_full_eps2_4_validation_v1 \
  --source-run /project2/yzhao010_1531/csci_699_new_arch/kbali-donut-worktree/outputs/donut_attack/donut_paper_full_eps8_16_validation_v1 \
  --output outputs/imported/donut_public_validation/merged/attacks_for_defenses.parquet
```

The importer requires 484 records over the identical 121 documents, copies all
primary PNGs into `outputs/imported/donut_public_validation/images`, verifies
every copied file against its recorded SHA256, freezes its main-project path,
and records the epsilon-32 exclusion policy.

Prepare the matched defense and adaptive runtime configs:

```bash
DONUT_SOURCE="$PWD/outputs/imported/donut_public_validation/merged/attacks_for_defenses.parquet"

python scripts/prepare_stage_config.py defenses_donut_public \
  --source donut_public="$DONUT_SOURCE" \
  --output outputs/runtime_configs/defenses_donut_public_final_v1.json

python scripts/prepare_stage_config.py jpeg_eot_donut \
  --source donut_public="$DONUT_SOURCE" \
  --output outputs/runtime_configs/donut_jpeg_eot_public_full_v1.json

python scripts/prepare_stage_config.py jpeg_eot_donut_hardview \
  --source donut_public="$DONUT_SOURCE" \
  --output outputs/runtime_configs/donut_hardview_public_full_v1.json
```

Run all three CPU preflights. Before full submission, create reviewed smoke
copies with a small document limit and confirm model loading, official-forward
preprocessing parity, PNG bounds, defense evaluation, checkpoint restart, and
strict merge. Do not bypass the smoke because the generic stages support both
models but their processors and differentiable objectives are different.

After the smoke passes, submit the full five-worker defense run:

```bash
for USER in user1 user2 user3 user4 user5; do
  bash scripts/submit_stage.sh defenses "$USER" \
    outputs/runtime_configs/defenses_donut_public_final_v1.json
done
```

Then submit both adaptive methods:

```bash
for CONFIG in \
  outputs/runtime_configs/donut_jpeg_eot_public_full_v1.json \
  outputs/runtime_configs/donut_hardview_public_full_v1.json
do
  for USER in user1 user2 user3 user4 user5; do
    bash scripts/submit_stage.sh jpeg_eot_attack "$USER" "$CONFIG"
  done
done
```

Status commands:

```bash
python scripts/stage_status.py defenses defenses_donut_public_final_v1
python scripts/stage_status.py jpeg_eot_attack donut_jpeg_eot_public_full_v1
python scripts/stage_status.py jpeg_eot_attack donut_frequency_hardview_public_full_v1
```

Use `midterm/08_defense_results.ipynb` for matched Qwen/Donut non-adaptive
defenses and `midterm/10_adaptive_defense_attack.ipynb` for all four adaptive
method/model combinations.

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
- `midterm/10_adaptive_defense_attack.ipynb`: full adaptive EOT/BPDA results
  against the four completed Qwen defenses.

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

The two full adaptive studies dominate the remaining budget. Each contains 879
attacks, three gradient-bearing EOT views per step, full-view checkpoint loss
audits, and exact four-defense evaluations. A reasonable planning range is
roughly 200-350 aggregate L40S hours per adaptive method, with early stopping
and multi-page resumes determining the actual value. Together with collected
receipts, Donut, and transfer work, the remaining project may require roughly
450-750 aggregate L40S hours.

Five workers reduce wall-clock time but not aggregate GPU-hours. The eight-hour
request is a safety ceiling per allocation, not an expectation that a full
shard finishes once. If a worker times out, resubmit its identical command; it
reuses completed records and resumes the current attack from its ten-step
checkpoint. Never delete or rename the partial run directory.
