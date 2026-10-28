# 06 - Main Qwen full-page attacks

## Two tracks, one stage

The Qwen attack is split into two **tracks** that are developed and run
independently, so two owners can work in parallel without editing each other's
code:

| Track | Datasets | Owner | Stage module | Active config |
|---|---|---|---|---|
| Receipt (primary study) | SROIE, CORD | Saaketh | `src/stages/qwen_attack_receipt.py` | `configs/qwen_attack/receipt_active.json` |
| Resume (secondary case study) | resume_parsing_vision | Shail | `src/stages/qwen_attack_resume.py` | `configs/qwen_attack/resume_active.json` |

The resume implementation, receipt implementation, and shared attack core are
merged. Each track keeps its own frozen inputs, run ID, output location, and
acceptance evidence; changes to the shared core require review because they can
affect both result sets.

Receipt execution now follows a team-approved speed-first exception: start the
five-worker validation directly on L40S GPUs without first running a separate
receipt smoke, saved-image review, or intentional checkpoint-resume
demonstration. The tracked receipt production gate is enabled. This exception
does not remove the stage's built-in checks for frozen inputs, exact Git/config
identity, processor parity, finite losses and gradients, epsilon bounds,
save/reload verification, or checkpoint integrity. Both Qwen launchers use
L40S GPUs in the current CARC configuration.

Both tracks share:

- **one stage name**, `qwen_attack`, and one run layout,
  `qwen_attack/<run_id>/`; receipt and resume resolve that layout beneath
  different output bases as documented below;
- **one attack core**, `src/attacks/qwen.py` (objective, differentiable
  preprocessing, PGD loop, save/reload verification, checkpoint state);
- the frozen `baseline` Qwen config, prompt, processor settings, and
  clean-correct IDs;
- the five-worker sharding contract in `00_execution_contract.md`.

`src/stages/qwen_attack.py` is a small router that reads `config["track"]` and
calls the matching track module. Nobody edits the router to add track logic.

Each track has its own config, run ID, smoke test, validation run, frozen test
run, and merge. Receipt results are complete and reportable without the resume
track; resume work never blocks receipts.

- CPU analysis: `notebooks/05_attack_analysis.ipynb` reads both tracks' merged
  outputs.

## Purpose

Run the main white-box targeted-attack experiment against
Qwen2.5-VL-3B-Instruct. The primary study is receipt-total extraction on SROIE
and CORD. Resume degree extraction is the secondary case study.

**GPU required:** yes, on CARC. This is the most memory- and time-intensive
stage, so checkpointing and five-worker sharding are mandatory for validation
and test runs of both tracks.

## Questions answered

1. How does targeted attack success change at epsilon 2, 4, 8, and 16?
2. What perceptibility cost accompanies each attack strength?
3. How often does success survive lossless save/reload?
4. Does vulnerability extend from receipt totals to resume degrees?
5. How expensive is Qwen attack optimization per document?

## Inputs

Both tracks read the **frozen** baseline run. Do not re-run or edit the
baseline to obtain these files; they are in the shared CARC project space (or
are copied unchanged from it, verified by SHA-256).

```text
data/processed/master_manifest.parquet
data/processed/sroie/{validation,test}/                  # receipt track
data/processed/cord_v2/{validation,test}/                # receipt track
data/processed/resume_parsing_vision/{validation,test}/  # resume track
outputs/baseline/<frozen_qwen_run>/merged/predictions.parquet
outputs/baseline/<frozen_qwen_run>/merged/qwen_clean_correct_ids.txt
outputs/baseline/<frozen_qwen_run>/merged/merge_manifest.json
outputs/donut_attack/<frozen_validation_run>/merged/summary.json  # receipt step-size rule
```

The validation baseline run is `baseline_validation_v1`. Each track config
records the SHA-256 of the baseline config and `predictions.parquet` and
refuses to start if either differs.

Attack only records satisfying:

- `usable=True`;
- `attack_eligible=True`;
- Qwen `clean_correct=True` under the exact frozen baseline config;
- dataset is in the track's `datasets` list;
- split equals the run's declared split.

Use validation for implementation and hyperparameter decisions. Test attacks
start only after that track's attack config is frozen and marked
`frozen_for_test=true`.

## Model and prompt

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Prompt, output normalization, image limits, generation settings, and
  precision: load unchanged from the frozen `baseline` Qwen config.
- Freeze every model parameter. Only uploaded-image pixels are optimized.
- Do not use quantized weights for gradient attacks.

A run is invalid if any document's clean prediction differs from the stored
`baseline` clean prediction. Record and stop on such configuration drift rather
than attacking under a silently changed pipeline.

## Shared attack core: `src/attacks/qwen.py`

The core is written once and used unchanged by both tracks, so receipt and
resume numbers are produced by the same attack. It takes a list of page images
(one page for receipts, all retained pages for resumes), the frozen prompt, the
target string, and a track-supplied normalization function.

**Ownership.** Shail creates the core in the first PR by moving the
track-independent parts of his resume implementation (objective, differentiable
preprocessing, PGD loop, save/reload, checkpoint contract) out of resume-only
code. The receipt owner reviews that PR before merge. After merge, any change
to `src/attacks/qwen.py` needs approval from both track owners, and a change
made after a frozen run requires a new run ID for every affected track.

### Objective

Create the exact chat template used at inference, append the complete desired
assistant answer followed by `<|im_end|>`, and compute teacher-forced
target-token cross-entropy. Mask system, user, visual-placeholder, and padding
tokens so only assistant target tokens contribute to the loss.

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

Qwen's normal image processor uses PIL bicubic resizing, which breaks gradients
and is not pixel-identical to PyTorch bicubic resizing. Use BPDA (a
straight-through estimator): every model forward receives the **exact official
PIL/NumPy processor values**, while the backward pass uses the differentiable
PyTorch resize/normalize path. The implementation must:

1. reproduce the frozen official processor's output dimensions;
2. preserve aspect ratio and interpolation choices;
3. match official processed tensors within the strict exact-forward tolerance;
4. remain connected to the original image tensor through the BPDA surrogate;
5. be tested on at least 20 varied page dimensions, including at least 10
   receipts and at least 5 resume pages;
6. use the real official pipeline again for final saved-image verification.

Report both exact-forward parity and raw surrogate error in **0-255 pixel
units** as well as normalized units. The raw surrogate mean error must be well
below the smallest epsilon (2/255); document its maximum error, but do not fail
on isolated maximum differences caused by PIL versus PyTorch interpolation.
Do not loosen the exact-forward or surrogate-mean tolerances just to pass.

Do not optimize only a post-resize tensor and call it an uploaded-image attack
unless an inverse mapping to a bounded original-resolution PNG is implemented
and verified.

### Initial candidate grid

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

The epsilon grid, seed, and core are identical across tracks. Step count and
step size are frozen per track from that track's validation run and recorded in
its config.

If 100 Qwen steps are computationally infeasible, compare 25, 50, and 100 on a
fixed validation smoke subset, freeze one setting, and document the tradeoff.
Never shorten only unsuccessful test attacks after observing their outcomes.

## Receipt track

Optimize the one receipt image. The target is the exact normalized value in
`attack_target`. Save as lossless RGB PNG at the original dimensions.

Smoke subset: five SROIE and five CORD clean-correct validation receipts at
epsilon 8.

The receipt tasks 1-6 CPU handoff passed on CARC at implementation commit
`fa5934fd6d3a1345da1fd0130998c297ace2e61a`: 24 focused tests passed and all
219 sources (123 SROIE and 96 CORD v2) produced canonical source-record digest
`77480634c85f37271c1927798bd975858877da6c1ebb06442c549730d58da277`.
The Task 7 closure pinned that evidence and, at that time, unlocked only the
owner smoke. It did not claim that Qwen, CUDA, gradients, or an attack had run,
and `receipt_active.json` was still blocked then. The later speed-first
activation described above supersedes that execution gate without changing
the accepted source evidence. The complete evidence record and the distinction
between the stable source digest and historical report hash are in
`qwen_receipt_contract.md`.

Task 8 then verified the exact closure commit
`0cc7c13fd640545966c98bafc4aa58fd919ec6c3` on CARC: 31 of 31 focused tests
passed, all 219 sources matched the accepted digest before smoke selection, the
same deterministic ten-image CPU preflight passed, and the worktree stayed
clean. The additional Task 8 report SHA-256 is
`ad27ba4b401a1f8b4d457f847ea9043c3d93d3414c107100a89795949e3a42ee`;
it supplements rather than replaces the original accepted report hash.

## Resume track

Provide all retained pages in original order. Optimize all pages jointly under
the same per-pixel epsilon bound, save one PNG per page, and count document
success only from a new multi-page inference over the reloaded files. Report
L-infinity as the maximum across pages and LPIPS/SSIM both per page and averaged
per document.

If multi-page gradients exceed memory, choose and document one fallback during
the resume smoke test, before the validation run: gradient accumulation across
pages, or attack only the page known to contain the highest degree. Do not
change the policy per document.

Smoke subset: five single-page and five multi-page clean-correct validation
resumes at epsilon 8.

## Gates

Each track moves through its own gates independently. Only gate 1 is shared.

1. **Shared core merged (both owners).** `src/attacks/qwen.py` merged with CPU
   tests; processor parity passed on the 20+ varied page dimensions above;
   nonzero finite gradients and target-loss reduction shown on one receipt and
   one multi-page resume.
2. **Track smoke (track owner alone).** The owner runs the smoke subset with
   the one-worker smoke runner (see below). Verify bounds, checkpoint resume,
   official-pipeline reload, and peak memory before expanding.
3. **Track validation (all five teammates).** All clean-correct validation
   documents of that track at all four epsilons, five shards. Use these runs to
   freeze step count, step size, early stopping, and memory settings.
4. **Track test (all five teammates).** Write a new immutable config with
   `frozen_for_test=true`, its validation run ID, and a SHA-256 of the selected
   settings. Run the held-out test set once without further tuning. Any bug fix
   after inspecting test outputs requires a documented new run; never replace
   results silently.

For `qwen_receipt_validation_v1`, the team explicitly waived gate 2 and moved
the receipt track directly from the completed CPU/source checks to gate 3.
This is a receipt-only execution decision; it does not change the resume-track
gates or remove any checks executed inside a production receipt job.

If GPU queue time is scarce, receipt validation and test jobs go first.

## Configs and run IDs

Every config is tracked in Git under `configs/qwen_attack/` and contains at
least:

```json
{
  "stage": "qwen_attack",
  "track": "receipt",
  "run_id": "qwen_receipt_validation_v1",
  "mode": "validation",
  "ready_for_production": true,
  "split": "validation",
  "datasets": ["sroie", "cord_v2"],
  "frozen_for_test": false,
  "seed": 566,
  "baseline_run_id": "baseline_validation_v1",
  "baseline_config_sha256": "...",
  "baseline_predictions_sha256": "...",
  "attack": {"epsilons_255": [2, 4, 8, 16], "max_steps": 100,
             "step_size_255": {"2": 0.5, "4": 1.0, "8": 2.0, "16": 4.0},
             "checkpoint_interval": 10}
}
```

The resume config is the same with `"track": "resume"`,
`"datasets": ["resume_parsing_vision"]`, and its own run ID. Receipt step
sizes are the team-approved candidate settings for the direct validation run;
the validation results, not the held-out test, determine any later frozen test
settings. Every non-smoke config must contain `ready_for_production=true`;
both the launcher and stage reject a gated config before production work
begins.

| Purpose | Receipt run ID | Resume run ID |
|---|---|---|
| Smoke (owner only) | `qwen_receipt_smoke_saaketh_v1` | `qwen_resume_smoke_shail_v2` |
| Resume proof (owner only) | n/a | `qwen_resume_smoke_shail_v3_resumeproof` |
| Best-iterate acceptance (owner only) | n/a | `qwen_resume_smoke_shail_v4_bestiterate` |
| Validation (5 shards) | `qwen_receipt_validation_v1` | `qwen_resume_validation_v1` |
| Test (5 shards) | `qwen_receipt_test_v1` | `qwen_resume_test_v1` |

`receipt_active.json` and `resume_active.json` always point at the run the team
should submit next. Owners change them only between runs.

## How to run

### Receipt CARC roots and interpreter

Receipt commands require these values to be set explicitly in every submitting
shell. Do not infer the interpreter from a Conda prompt and do not use a bare
`python`:

```bash
export PP_PYTHON_BIN=/absolute/path/to/the/qualified/full-runtime/bin/python
export PP_SHARED_ARTIFACT_ROOT=/absolute/path/to/read-only/poisoned-paperwork
export PP_RECEIPT_OUTPUT_ROOT=/absolute/path/to/shared-writable/receipt-outputs
```

`PP_RECEIPT_OUTPUT_ROOT` is a base directory, not a run-specific directory.
The code deterministically resolves a run to
`$PP_RECEIPT_OUTPUT_ROOT/qwen_attack/<run_id>`. It must exist before submission,
must be group-writable and group-searchable with setgid enabled, must support
file creation, locking, and atomic rename for every participating account, and
must be a directory tree separate from both the personal checkout and
`PP_SHARED_ARTIFACT_ROOT`. Both the launcher and the receipt core reject an
output base that lacks those mode bits.

For the current direct validation, use the production launcher's dry-run mode.
It validates the explicit interpreter, roots, clean exact commit, committed
config, run and worker contract, then prints the L40S Slurm command without
submitting it:

```bash
bash scripts/submit_stage.sh qwen_attack user3 \
  configs/qwen_attack/receipt_active.json --dry-run
```

The conservative Task 9 cross-account storage preflight remains available as
a diagnostic, but the team waived it as a prerequisite for this speed-first
run. If the team elects to run it later, each account uses the same commit and
filesystem-safe session ID with its own user ID:

```bash
"$PP_PYTHON_BIN" -E -s scripts/preflight_qwen_receipt_storage.py \
  --config configs/qwen_attack/receipt_smoke_saaketh.json \
  --user-id user3 \
  --session-id task9-<UTC-timestamp> \
  --expected-git-commit <exact-Task-9-commit>
```

After user1 through user5 complete that optional diagnostic, Saaketh can repeat
it with the identical session and commit plus `--require-all`. That final pass
observes five distinct account markers and verifies cross-account reads,
advisory locking, and atomic rename in the shared tree. It is storage evidence
only and does not run Qwen or request a GPU.

Receipt production is now enabled in `receipt_active.json` for the approved
direct validation. The config still fails closed if that Boolean is false.

### Smoke (optional receipt diagnostic; required by the resume workflow)

If used, smoke tests follow the isolated one-worker pattern of
`scripts/smoke_donut_attack.py`: a dedicated runner calls the track's
`run(..., worker_id=0, num_workers=1)` with a smoke config whose
`mode` is `"smoke"`. Track modules accept `num_workers=1` **only** when
`mode == "smoke"`; validation and test runs require five workers.

```bash
python scripts/check_qwen_processor_parity.py \
  --config configs/qwen_attack/resume_smoke_shail.json
python scripts/smoke_qwen_attack.py \
  --config configs/qwen_attack/resume_smoke_shail.json --preflight
mkdir -p logs
bash scripts/submit_qwen_smoke.sh \
  configs/qwen_attack/resume_bestiterate_shail.json
```

The submit wrapper verifies the interpreter and core package imports, processor
parity, frozen artifacts, and candidate selection before requesting a GPU. It
also strips inherited Conda activation markers before calling Slurm. The batch
job activates the environment afresh and repeats processor parity before
allocating the model. A passing
result confirms a pixel-exact official forward, a bounded-mean differentiable
surrogate, and the BPDA construction used by the attack. The smoke
runner then requires exactly ten documents, no logged failure, finite metrics
and traces, valid PNG hashes, and no epsilon violation. To prove resume, cancel
one smoke attempt only after a `state.pt` exists, resubmit the exact command,
and confirm at least one final record reports `checkpoint_resumed=true`.
Visually inspect every smoke page with
`notebooks/06_qwen_smoke_review.ipynb` before unlocking production.
The smoke launcher requests one hour; the 100-step ten-document L40S pilot
completed in about 25 minutes, leaving more than a twofold wall-time margin.
The separate resume-proof smoke uses 100 steps at epsilon 16, matching the
strongest frozen production setting. It also serves as an efficacy pilot: zero
decoded successes remain a valid result if gradients are finite and the best
target loss improves, but should be reviewed before spending the full run.

After review, change only `ready_for_production` in
`configs/qwen_attack/resume_active.json` to `true`, commit and push it, then
have all five teammates pull the same commit and submit the commands below.

### Validation and test (everyone, five shards per track)

Each teammate runs one command per track, with only their own user ID changed:

```bash
# Receipt track
bash scripts/submit_stage.sh qwen_attack user1 configs/qwen_attack/receipt_active.json  # Prabudhd
bash scripts/submit_stage.sh qwen_attack user2 configs/qwen_attack/receipt_active.json  # Gary
bash scripts/submit_stage.sh qwen_attack user3 configs/qwen_attack/receipt_active.json  # Saaketh
bash scripts/submit_stage.sh qwen_attack user4 configs/qwen_attack/receipt_active.json  # Khalid
bash scripts/submit_stage.sh qwen_attack user5 configs/qwen_attack/receipt_active.json  # Shail

# Resume track
bash scripts/submit_stage.sh qwen_attack user1 configs/qwen_attack/resume_active.json   # Prabudhd
bash scripts/submit_stage.sh qwen_attack user2 configs/qwen_attack/resume_active.json   # Gary
bash scripts/submit_stage.sh qwen_attack user3 configs/qwen_attack/resume_active.json   # Saaketh
bash scripts/submit_stage.sh qwen_attack user4 configs/qwen_attack/resume_active.json   # Khalid
bash scripts/submit_stage.sh qwen_attack user5 configs/qwen_attack/resume_active.json   # Shail
```

The two tracks have different run IDs, so their jobs can run at the same time
without touching each other's checkpoints. The launcher requests one L40S
48 GB GPU, 8 CPUs, 64 GB system RAM, and twelve hours for each worker. The
resume smoke peaked at about 30.9 GiB on an L40S, leaving about 14 GiB free.
After a
timeout, resubmit the identical command; completed experiments are skipped and
unfinished ones resume from their last checkpoint. Nobody edits
`src/attacks/qwen.py`, a track module, or an active config during a run.

The final worker merges automatically. A strict CPU-only retry is available:

```bash
"$PP_PYTHON_BIN" -E -s scripts/merge_qwen_receipt.py \
  --config configs/qwen_attack/receipt_active.json
"$PP_PYTHON_BIN" -E -s scripts/merge_qwen_resume.py \
  --config configs/qwen_attack/resume_active.json
```

## Sharding and checkpoints

Within a track, all workers receive disjoint documents
(`src/sharding.py: assigned_record_ids`, sorted record IDs modulo 5) and run
the complete epsilon grid for each assigned document. Within a document,
epsilon runs are sequential to limit GPU memory. Track modules use the shared
helper; they do not invent their own assignment.

Checkpoint each document/epsilon every 10 iterations:

```text
<track-output-base>/qwen_attack/<run_id>/
├── assignments/worker_00.txt ... worker_04.txt
├── shards/worker_00/
│   ├── checkpoints/<experiment_key>/state.pt
│   ├── images/<experiment_key>/page_000.png
│   ├── records/<experiment_key>.json
│   ├── progress.json
│   └── WORKER_COMPLETE.json
└── merged/
```

For receipts, `<track-output-base>` is exactly
`$PP_RECEIPT_OUTPUT_ROOT`; it is not the personal checkout. The receipt stage,
smoke summary, manual merge, and all five production workers resolve that same
base. The resume track retains its existing repository-relative `outputs/`
layout until its owner adopts an equivalent independently reviewed storage
contract.

The state contains all fields required in the shared execution contract plus
the exact processed image grid, chat template hash, target token IDs, and best
known attack tensor. A restarted job must verify all hashes before resuming.

## Exact outputs

Each track run produces, after its own merge:

```text
<track-output-base>/qwen_attack/<run_id>/merged/
├── attacks.parquet
├── attacks.csv
├── page_metrics.parquet
├── successful_attack_ids.txt
├── failures.csv
└── summary.json
```

Both tracks use the **same record schema** so downstream stages can
concatenate them. Each attack record contains the `donut_attack` fields plus:

```text
track, document_page_count, attacked_page_count, target_token_ids_hash,
official_processor_shape, differentiable_processor_shape,
processor_max_abs_difference, mean_target_log_probability,
gradient_norm_final, peak_gpu_memory_mb
```

Before downstream stages (07-10) consume receipt results, their configs or
launchers must be updated and reviewed to point explicitly at
`$PP_RECEIPT_OUTPUT_ROOT/qwen_attack/<run>/merged/`; they must not silently
fall back to a personal checkout. Task 9 changes receipt attack execution and
merge paths only—it does not claim that those later stages are implemented.

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

- `src/attacks/qwen.py` is shared by both tracks; exact-forward BPDA
  preprocessing is validated on receipts and resume pages, and its raw
  differentiable surrogate error is recorded separately.
- All attacked PNGs retain original dimensions and satisfy epsilon after reload.
- All clean-correct validation candidates of each track have results for four
  epsilons.
- Each track's test settings are frozen from its own validation run before
  test inference.
- All five worker shards of each run merge without duplicates or missing keys.
- Conditional ASR, harmful-output rate, LPIPS, SSIM, and runtime are reported.
- Receipt results are complete even if the secondary resume study is reduced.
