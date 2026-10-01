# Immediate next task: integrate Donut and complete the clean baseline

This document is the complete handoff for the next implementation task. A
Donut receipt baseline and Donut targeted-attack scaffold already exist in Git,
but the complete project baseline still requires integration, correctness
fixes, Qwen inference, and résumé support. The assigned owner should finish and
verify one unified clean baseline stage. The other four teammates should not
submit production jobs until the owner announces that the smoke test passed
and provides the exact Git commit and run ID.

## Outcome

The baseline stage must run unmodified documents through the pinned Qwen and
Donut models, save one resumable prediction record per experiment, and produce
the clean-correct document lists used by every later attack.

This stage does not train a model and does not create adversarial images.

- Stage command: `baseline`
- Main implementation: `src/stages/baseline.py`
- Production config: `configs/baseline/active.json`
- GPU: one L40S 48 GB per worker
- Team structure: five deterministic document shards
- Analysis notebook: `notebooks/04_baseline_analysis.ipynb`, created only after
  merged baseline results exist

## Current implementation state

The existing Donut implementation is available in Git commit:

```text
ba0032fbbce420864aed6421ff5e908dc1e344e0
```

If those files are not already present in the working branch, inspect them
without checking out or overwriting a dirty worktree:

```bash
git show ba0032f:src/models/donut.py
git show ba0032f:src/attacks/donut.py
git show ba0032f:src/stages/baseline.py
git show ba0032f:src/stages/donut_attack.py
git show ba0032f:src/normalization.py
git show ba0032f:tests/test_donut_attack.py
git show ba0032f:tests/test_normalization.py
git show ba0032f:configs/baseline/active.json
```

Do not blindly copy the entire commit over current files. Read the current
working tree and integrate only after comparing both versions. Preserve
unrelated notebook or user changes.

### What is already implemented

| Existing file | Implemented behavior |
|---|---|
| `src/models/donut.py` | Pinned Donut DocVQA loading, prompt template, deterministic clean inference, RGB tensor conversion, and differentiable processor approximation |
| `src/attacks/donut.py` | Target-token cross-entropy and margin losses, L-infinity projection, Adam/sign optimization, and attack checkpoint save/load |
| `src/normalization.py` | Conservative SROIE and CORD receipt-total normalization |
| `src/stages/baseline.py` | Donut-only SROIE/CORD selection, five-way sharding, atomic prediction records, provenance, progress, completion markers, and automatic merge |
| `src/stages/donut_attack.py` | Clean-correct selection, targeted image attack, PNG save/reload verification, L-infinity/LPIPS/SSIM metrics, checkpoint resume, and merge |
| `tests/test_donut_attack.py` | CPU tests for projection, target loss positions, frozen model weights, optimization, checkpoint resume, and differentiable preprocessing |
| `tests/test_normalization.py` | Receipt normalization tests |
| `configs/baseline/active.json` | A 20-receipt Donut smoke configuration, not a complete baseline config |
| `configs/donut_attack/example_validation.json` | An inactive attack template that requires a completed Donut baseline path |

The existing code is a useful starting point. It is intentionally Donut-only,
has not completed real CARC GPU validation, and must not be described as the
complete project baseline.

### What remains for a complete baseline

1. Add the pinned Qwen adapter and deterministic inference.
2. Run Qwen on SROIE, CORD, and all retained résumé pages.
3. Add exact seven-label résumé normalization.
4. Replace the Donut-only config schema with one unified Qwen-plus-Donut
   baseline config.
5. Change output filenames and merge uniqueness from `record_id` to a complete
   experiment key so Qwen and Donut can both produce a row for one receipt.
6. Validate existing record contents before skipping them on resume.
7. Validate exact dataset and expected experiment counts before declaring a
   worker or merged run complete.
8. Write explicit failed records so one permanently failing document does not
   make a run impossible to close and audit.
9. Validate config hash, Git commit, model revision, prompt, schema, and status
   during merge.
10. Produce separate Qwen and Donut clean-correct ID files.
11. Record missing provenance fields, including precision, input page count,
    prompt text hash, and GPU driver.
12. Add baseline, checkpoint, merge, corrupted-record, and test-freeze tests.
13. Add a clear owner-only smoke command and a CPU merge command.
14. Run actual Qwen and Donut CARC smoke tests before team submissions.

### Existing-code corrections required before production

The following behaviors in the starting implementation must be changed rather
than carried forward:

- Do not skip a record only because `<key>.json` exists. Parse it and verify all
  required fields, `status`, config hash, model revision, prompt ID, and
  experiment key. A corrupt, partial, or stale record must be rerun.
- Do not use `record_id` alone as the filename or merge key. Receipts have both
  Qwen and Donut baseline experiments.
- Do not count arbitrary `*.json` files as proof of completion. Compare the
  exact observed experiment-key set with the exact expected set.
- Do not define the expected run from whatever subset happens to load. Assert
  the locked validation/test counts in this document.
- Do not leave failures only in an append-only log. Write an auditable final
  failed record with the same schema and `status="failed"`.
- Do not use generic `clean_correct_ids.txt` after multiple models are present.
- Do not treat the current 20-receipt `active.json` as a full validation run.
- Do not merge reported results until both real model adapters pass CARC smoke
  tests.

The targeted-attack code is outside the baseline's immediate execution path.
Preserve it and its tests, but do not expand or run the attack until the full
Donut clean baseline has completed and been reviewed.

## Ordered integration plan for an LLM

An LLM completing this task should work in this order:

1. Read all required files below and inspect `git status --short`.
2. Compare the current branch with commit `ba0032f`; do not checkout over local
   changes.
3. Integrate and retain `src/models/donut.py`, `src/attacks/donut.py`, the
   Donut tests, and receipt normalization.
4. Extract generic data/checkpoint/merge behavior from the Donut-only baseline
   instead of duplicating it inside two model paths.
5. Implement strict record validation and exact experiment-key helpers first.
6. Add `src/models/qwen.py` and unit-test the prompt/message construction with
   mocked model objects.
7. Extend normalization with résumé labels and tests.
8. Rewrite `src/stages/baseline.py` as the unified Qwen-plus-Donut orchestrator.
9. Create the unified config and exact expected experiment set.
10. Add the owner smoke script and standalone CPU merge script.
11. Run all CPU tests without downloading models.
12. Run a Donut CARC smoke test, then a Qwen CARC smoke test, using the exact
    production adapters.
13. Run the complete five-worker validation round and merge it.
14. Freeze prompts/settings, create a new test run ID, and run the five-worker
    held-out test round.
15. Create the analysis notebook only after merged outputs pass all checks.

## Required reading before an LLM generates code

An LLM or developer must inspect these files in this order before editing. Do
not generate a replacement architecture without understanding the existing
launcher and data contract.

1. `README.md`
   - Current project status, data counts, CARC workflow, team identities, GPU
     policy, and stage order.
2. `docs/00_execution_contract.md`
   - Required `run(...)` signature, sharding, checkpoints, completion markers,
     environment metadata, and merge rules.
3. `docs/04_clean_baselines.md`
   - Exact baseline models, revisions, prompts, metrics, outputs, and scientific
     completion criteria.
4. Existing Donut implementation at commit `ba0032f`:
   - `src/models/donut.py`
   - `src/attacks/donut.py`
   - `src/stages/baseline.py`
   - `src/stages/donut_attack.py`
   - `src/normalization.py`
   - `tests/test_donut_attack.py`
   - `tests/test_normalization.py`
   - `configs/baseline/active.json`
   - Understand what can be reused and the corrections listed above. Do not
     regenerate working Donut math from scratch without a concrete reason.
5. `src/team.py`
   - Fixed mapping from `user1` through `user5` to worker IDs.
6. `src/sharding.py`
   - Existing deterministic document assignment. Reuse it; do not invent a
     second sharding rule.
7. `scripts/run_gpu_stage.py`
   - Imports `stages.baseline` and calls its `run(...)` function.
8. `scripts/submit_stage.sh`
   - Already requests the L40S, CPU, memory, time limit, log files, config, and
     worker ID. Teammates must not allocate a production GPU manually.
9. `scripts/slurm/gpu_stage.sbatch`
   - Activates the CARC environment and starts the Python dispatcher.
10. `scripts/setup_carc.sh` and `scripts/activate_paperwork.sh`
   - The CARC runtime is the `paperwork` conda environment using Python 3.11.
11. `env/requirements.txt` and `env/requirements.lock.txt`
    - Installed model and data-library versions. Add a dependency only if it is
      genuinely missing, and update both through the setup process.
12. `data/processed/preprocessing_summary.csv`
    - Expected document counts by dataset and split.
13. `data/processed/master_manifest.parquet`
    - Inspect its schema and a few rows, not the entire table in a prompt.
14. The three `data/processed/<dataset>/dataset_dict.json` files and split
    `dataset_info.json` files.
    - SROIE and CORD contain one `image` per document. Résumés contain an
      ordered `images` list; all retained pages form one document.
15. `notebooks/03_data_preprocessing.ipynb`
    - Consult only when the processed schema or split construction is unclear.
      Do not duplicate preprocessing inside the baseline stage.
16. `tests/test_team_and_sharding.py`
    - Preserve the existing five-worker behavior.
17. `.gitignore`
    - Data, model caches, logs, outputs, checkpoints, and model weights must
      remain outside Git.

Before editing, also run `git status --short` and preserve unrelated work. The
LLM should read small metadata files, not load all images merely to understand
the repository.

## Locked data contract

Load the processed Hugging Face datasets with `datasets.load_from_disk`:

```text
data/processed/sroie/
data/processed/cord_v2/
data/processed/resume_parsing_vision/
```

The manifest includes:

```text
record_id, dataset_name, document_id, source_split, source_index,
experiment_split, task, language, num_pages, target_raw, clean_target,
attack_target, target_source, usable, attack_eligible, exclusion_reason
```

Expected usable counts:

| Dataset | Validation | Test |
|---|---:|---:|
| SROIE | 126 | 361 |
| CORD v2 | 100 | 100 |
| English résumés | 75 | 75 |
| **Total documents** | **301** | **536** |

Run Qwen on all usable validation/test documents. Run Donut only on SROIE and
CORD receipts.

Expected final prediction rows:

| Phase | Qwen rows | Donut rows | Total rows |
|---|---:|---:|---:|
| Validation | 301 | 226 | 527 |
| Frozen test | 536 | 461 | 997 |

Shard by `record_id`, not by model invocation. All Qwen and Donut work for one
document must stay with the same worker.

## Locked models and prompts

### Qwen

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Precision: BF16 on L40S; fail clearly if the selected precision is unsupported
- Batch size: 1 initially
- Generation: greedy, `do_sample=False`, no temperature, `max_new_tokens=32`
- Processor: official processor, RGB input, aspect ratio preserved
- Initial pixel bounds: `min_pixels=256*28*28`, `max_pixels=1280*28*28`
- Résumés: pass every retained page in original order in one multi-image
  conversation; never score individual pages as independent résumés

### Donut

- Model: `naver-clova-ix/donut-base-finetuned-docvqa`
- Revision: `b19d2e332684b0e2d35d9144ce34047767335cf8`
- Scope: SROIE and CORD receipts only
- Batch size: 1
- Use the official processor settings and DocVQA question template
- Start smoke testing in FP32; use FP16 only if a fixed 20-document comparison
  produces the same normalized outputs
- Do not run this checkpoint on résumés

### Frozen prompt candidates

`receipt_total_v1`:

```text
What is the final total amount shown on this receipt? Return only the amount, with no currency symbol, label, or explanation.
```

`resume_degree_v1`:

```text
What is the highest degree level stated in this resume? Return exactly one label from: secondary, certificate_or_diploma, associate, bachelor, postgraduate, master, doctorate.
```

Prompt changes are allowed only during validation. Create a new prompt ID for
every change. Never silently edit the text attached to an existing prompt ID.

## Files to integrate, add, or modify

Preserve the existing model-specific Donut modules. Use `src/models/` for model
adapters and `src/attacks/` for attack logic. Keep generic loading,
normalization, checkpointing, and experiment-key helpers directly under `src/`.
The existing `src/stages/` directory remains the launcher entry point.

| File | Action | Required responsibility |
|---|---|---|
| `src/models/donut.py` | Integrate and retain | Pinned Donut loading, official clean inference, and differentiable preprocessing. Add configurable precision only after the required FP32/FP16 comparison. |
| `src/attacks/donut.py` | Integrate and retain | Keep attack primitives isolated from the clean baseline. Apply checkpoint provenance fixes without coupling attacks to Qwen. |
| `src/models/qwen.py` | Add | Pinned Qwen loading, official processor/chat template, deterministic receipt and multi-page résumé inference, and explicit unloading. |
| `src/data.py` | Add | Locate the repo, load the manifest and processed datasets, assert counts/fields, and return one document with either `image` or ordered `images`. |
| `src/normalization.py` | Extend | Retain tested receipt normalization and add strict seven-label résumé normalization. |
| `src/checkpoints.py` | Add | Atomic JSON writes, config/environment snapshots, required-field validation, exact experiment keys, resumable skipping, and completion markers. |
| `src/stages/baseline.py` | Refactor | Replace Donut-only orchestration with one Qwen-plus-Donut stage using the required `run(...)` signature. |
| `src/stages/donut_attack.py` | Preserve for later | Do not run until the Donut baseline is complete. Fix stale-record validation and exact merge-key validation before production attacks. |
| `scripts/smoke_baseline.py` | Add | Owner-only interactive smoke test on two SROIE, two CORD, and two résumé documents. |
| `scripts/merge_baseline.py` | Add | CPU merge requiring five completion markers and exact expected experiment keys. It may call shared merge functions used by the automatic merge. |
| `configs/baseline/active.json` | Replace after smoke review | Unified frozen Qwen-plus-Donut configuration. The existing 20-receipt Donut config remains a smoke reference, not production. |
| `tests/test_normalization.py` | Extend | Retain receipt cases and add résumé successes, explanations, invalid labels, empty output, and ambiguity. |
| `tests/test_donut_attack.py` | Integrate and retain | Preserve the attack-math regression tests even though attacks run later. |
| `tests/test_baseline.py` | Add | Filtering, expected counts, experiment keys, model scope, output schema, multi-page résumés, failures, and test protection using mocked adapters. |
| `tests/test_checkpoints.py` | Add | Atomic writes, valid completed records, corrupt/stale records, config mismatches, and restart skipping. |
| `tests/test_merge_baseline.py` | Add | Assignment gaps/overlaps, missing markers, missing/duplicate keys, provenance mismatches, row counts, and model-specific ID outputs. |

Modify `env/requirements.txt` only if an import required for this design is not
already present. Do not commit downloaded model weights or runtime outputs.

Do not create `notebooks/04_baseline_analysis.ipynb` until the first real merged
results exist. The notebook must visualize saved results, not rerun models.

## Configuration contract

The starting Donut config stores one top-level `model_id` and filters to two
receipt datasets. Keep it only as a smoke reference. The complete baseline
must use the unified schema below so one frozen run describes both model paths.

`configs/baseline/active.json` must include, at minimum:

```json
{
  "run_id": "baseline_validation_v1",
  "stage": "baseline",
  "split": "validation",
  "frozen_for_test": false,
  "seed": 566,
  "output_root": "outputs/baseline",
  "datasets": ["sroie", "cord_v2", "resume_parsing_vision"],
  "models": {
    "qwen": {
      "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
      "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
      "precision": "bfloat16",
      "max_new_tokens": 32,
      "min_pixels": 200704,
      "max_pixels": 1003520
    },
    "donut": {
      "model_id": "naver-clova-ix/donut-base-finetuned-docvqa",
      "revision": "b19d2e332684b0e2d35d9144ce34047767335cf8",
      "precision": "float16"
    }
  },
  "prompts": {
    "receipt_total": {
      "id": "receipt_total_v1",
      "text": "What is the final total amount shown on this receipt? Return only the amount, with no currency symbol, label, or explanation."
    },
    "resume_degree": {
      "id": "resume_degree_v1",
      "text": "What is the highest degree level stated in this resume? Return exactly one label from: secondary, certificate_or_diploma, associate, bachelor, postgraduate, master, doctorate."
    }
  }
}
```

The implementation may add fields, but it must not hard-code experimental
choices that belong in this config. On test, use a new `run_id`, set
`split="test"` and `frozen_for_test=true`, and include the selected validation
run ID. The stage must refuse a test run when `frozen_for_test` is false.

Do not modify `active.json` while any worker from that run is queued or
running. Its exact contents must be copied into the run output directory.

## Normalization rules

Always save `raw_response` before normalization.

### SROIE

- Strip surrounding whitespace.
- Permit an optional currency symbol and a short optional total label.
- Parse exactly one unambiguous amount.
- Format it with two decimal places.

Examples:

```text
$8.20      -> 8.20
Total: 8.2 -> 8.20
8.20 USD   -> 8.20
```

### CORD

- Strip whitespace, an optional currency prefix, and supported thousands
  separators.
- Return one non-negative integer rupiah amount as digits only.
- Test punctuation behavior explicitly; do not conflate decimal and thousands
  separators without a dataset-specific rule.

Examples:

```text
289,000   -> 289000
Rp 289000 -> 289000
```

### Résumés

- Lowercase and trim the response.
- Accept only one exact label from the seven-label vocabulary.
- Do not infer a label from a sentence or degree name during normalization.

Examples:

```text
Bachelor          -> bachelor
master            -> master
Master of Science -> parse failure
```

An empty, ambiguous, multiple-value, or explanatory response must produce:

```text
prediction_normalized = null
parse_status = "failed"
```

## Baseline stage implementation contract

`src/stages/baseline.py` must define:

```python
def run(config, config_path, user_id, worker_id, num_workers):
    ...
```

The function must:

1. Resolve every path from the repository root, not the current notebook or
   shell directory.
2. Validate the config, model revisions, split, expected data counts, and
   required fields before loading a model.
3. Select only usable records for the requested experiment split.
4. Call `assigned_record_ids(...)` from `src/sharding.py`.
5. Save the full expected assignment for the worker before model loading.
6. Create the complete expected experiment-key set for the assigned documents.
   The validation run has 527 keys and the frozen test run has 997 keys.
7. Reuse the integrated Donut adapter to run Donut only on assigned SROIE and
   CORD receipts.
8. Run Qwen on every assigned receipt and résumé, with all résumé pages passed
   together in original order.
9. Save one JSON record immediately after every model/document inference.
10. Use a temporary file plus atomic rename; never write a final record
    directly.
11. On restart, skip only records that exist, parse as JSON, match the expected
    experiment key, current config hash, model revision and prompt, and contain
    every required field with `status` equal to `complete` or an explicitly
    allowed final `failed` state.
12. Record inference exceptions as explicit failed records and continue when
    safe. Never silently drop a document.
13. Load one large model at a time. Unload it and clear GPU memory before
    loading the second model.
14. Record model revision, prompt ID, Git commit, GPU name, precision, runtime,
    and peak GPU memory.
15. Write `WORKER_COMPLETE.json` only after comparing the exact observed key
    set with the exact expected key set and verifying every key has a complete
    or explicit failed record.

The unique experiment key must include at least:

```text
record_id + model_id + task + prompt_id + experiment_split
```

## Prediction record contract

Every JSON prediction record must include:

```text
run_id
record_id
dataset_name
experiment_split
model_id
model_revision
task
prompt_id
clean_target
attack_target
attack_eligible
raw_response
prediction_normalized
parse_status
clean_correct
runtime_seconds
peak_gpu_memory_mb
worker_id
seed
git_commit
config_sha256
input_page_count
status
error
```

`clean_correct` is true only when parsing succeeded and the normalized
prediction exactly matches `clean_target`.

## Output and checkpoint layout

```text
outputs/baseline/<run_id>/
├── config.json
├── environment.json
├── assignments/
│   ├── worker_00.txt
│   ├── worker_01.txt
│   ├── worker_02.txt
│   ├── worker_03.txt
│   └── worker_04.txt
├── shards/
│   ├── worker_00/
│   │   ├── records/*.json
│   │   └── WORKER_COMPLETE.json
│   ├── worker_01/
│   ├── worker_02/
│   ├── worker_03/
│   └── worker_04/
└── merged/
    ├── predictions.parquet
    ├── predictions.csv
    ├── qwen_clean_correct_ids.txt
    ├── donut_clean_correct_ids.txt
    ├── qwen_clean_incorrect_ids.txt
    ├── donut_clean_incorrect_ids.txt
    ├── failures.csv
    └── summary.json
```

The model-specific clean-correct files are required because Donut and Qwen may
answer different documents correctly. Later attack stages must not use a
generic list that mixes models.

## Merge requirements

`scripts/merge_baseline.py` runs on CPU and must:

1. Load the same frozen config and expected input set.
2. Require all five assignment files and completion markers.
3. Assert assignments are disjoint and cover the complete expected document
   set.
4. Validate every JSON record and reject inconsistent config hashes, model
   revisions, prompt IDs, or Git commits.
5. Reject duplicate or missing experiment keys.
6. Assert 527 rows for validation or 997 rows for frozen test.
7. Write deterministic CSV and Parquet files.
8. Write model-specific clean-correct and clean-incorrect ID files.
9. Include parse failures and inference failures in `failures.csv` and summary
   counts.
10. Never overwrite an existing merged result unless the inputs are identical.

## Tests required before CARC smoke testing

Run:

```bash
pytest -q
```

At minimum, tests must cover:

- SROIE, CORD, and résumé normalization successes and failures;
- required manifest fields and expected counts;
- Qwen runs on all three datasets while Donut runs only on receipts;
- related model invocations remain on the document's assigned worker;
- experiment keys are stable and unique;
- incomplete/corrupt records are rerun;
- valid completed records are skipped;
- atomic temporary-file replacement;
- test execution is rejected unless `frozen_for_test=true`;
- merge rejection for missing worker markers, overlaps, gaps, duplicates, wrong
  config hashes, and wrong row counts;
- model adapters are mocked in CPU tests so unit tests do not download models.

## Owner-only interactive GPU workflow

The baseline owner uses an interactive L40S only for development and smoke
testing. From a CARC login node:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork

salloc --account=yzhao010_1531 --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=02:00:00 \
  --gpus-per-task=l40s:1

srun --pty bash -l
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
source scripts/activate_paperwork.sh
nvidia-smi
python scripts/smoke_baseline.py --config configs/baseline/active.json
```

The smoke script must run:

- two SROIE receipts through Qwen and Donut;
- two CORD receipts through Qwen and Donut;
- two résumés through Qwen using all retained pages;
- every example twice to confirm identical normalized predictions.

It must display record ID, image/page dimensions, processed tensor shape, raw
answer, normalized answer, target, correctness, runtime, and peak GPU memory.
Stop if a model returns only empty or malformed answers.

Also interrupt one smoke run and restart it to prove completed records are
skipped. The smoke test should load both models once so their pinned weights are
present in the shared project cache before production begins.

Leave the interactive allocation with `exit`, then release the allocation with
a second `exit` if necessary.

## Production workflow for all five teammates

Production GPU jobs are submitted from the normal CARC login node. Do not run
`salloc` first. `scripts/submit_stage.sh` already asks Slurm for one L40S, eight
CPUs, 32 GB system RAM, and a four-hour allocation.

Before announcing the run, the owner must:

1. Pass all CPU tests.
2. Pass the interactive GPU smoke test.
3. Freeze `configs/baseline/active.json`.
4. Commit and push code, tests, config, and documentation.
5. Send the team the exact Git commit, run ID, phase, and command.
6. Avoid changing code or config until all five workers finish.

Each teammate then runs:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull
git rev-parse HEAD
```

The printed commit must match the owner's announced commit. Submit exactly one
assigned shard:

```bash
# Worker 0
bash scripts/submit_stage.sh baseline user1

# Worker 1
bash scripts/submit_stage.sh baseline user2

# Worker 2
bash scripts/submit_stage.sh baseline user3

# Worker 3
bash scripts/submit_stage.sh baseline user4

# Worker 4
bash scripts/submit_stage.sh baseline user5
```

The command returns a Slurm job ID. Monitor it with:

```bash
squeue -u "$USER"
```

The important states are:

- `PD`: queued and waiting for a suitable GPU;
- `R`: running on an allocated GPU;
- `CG`: completing;
- absent from `squeue`: completed or failed, so inspect the log.

Inspect the job's log path printed by `submit_stage.sh`. The browser and login
terminal may be closed after submission; Slurm continues the job.

If a job times out or fails recoverably, rerun the exact same command. The
worker must resume its same shard and skip valid completed records.

## Validation and frozen-test rounds

The team runs two separate five-worker production rounds.

### Round 1: validation

```text
run_id = baseline_validation_v1
split = validation
frozen_for_test = false
```

After all workers finish, the owner merges the results on CPU:

```bash
source scripts/activate_paperwork.sh
python scripts/merge_baseline.py --config configs/baseline/active.json
```

Use only validation results to select prompts, processor choices, and supported
precision. Record every prompt candidate and its validation performance.

### Round 2: frozen test

After validation decisions are signed off, update and commit the config:

```text
run_id = baseline_test_v1
split = test
frozen_for_test = true
validation_run_id = baseline_validation_v1
```

Keep the selected prompts, model revisions, normalization, and processor
settings frozen. All five teammates pull the new commit and submit their
baseline command again. Do not tune settings after inspecting test results.

## What teammates do not need to do

Teammates do not:

- request an interactive GPU for production;
- choose a node or physical GPU;
- choose document IDs;
- change worker IDs;
- run another teammate's shard;
- edit `active.json`;
- download data manually;
- copy output files into Git;
- run models from a notebook.

The launcher handles the GPU request. Slurm may start the five workers at
different times; that is acceptable as long as they use the same commit,
config, run ID, model revisions, and L40S GPU type.

## Definition of complete

The baseline stage is complete only when:

- every required code and test file exists;
- all CPU tests pass;
- Qwen and Donut smoke tests produce deterministic normalized answers;
- the exact model revisions and prompt texts are recorded;
- checkpoint interruption and resume behavior is demonstrated;
- all five validation workers have completion markers;
- validation merges to exactly 527 expected prediction rows;
- validation prompt and processor decisions are frozen;
- all five test workers have completion markers;
- test merges to exactly 997 expected prediction rows;
- no duplicate or missing experiment keys remain;
- failures and parse failures are reported rather than discarded;
- Qwen and Donut clean-correct ID files exist separately;
- the output config, environment, Git revision, GPU model, and runtime metadata
  are saved;
- `git diff --check` and the test suite pass;
- no datasets, model weights, logs, checkpoints, or outputs are staged in Git.

Only after these checks should the team create
`notebooks/04_baseline_analysis.ipynb` and begin the attack implementations.

## Message the owner sends before production

The stage owner should send one unambiguous message in this form:

```text
Baseline validation is ready.

Git commit: <full commit SHA>
Run ID: baseline_validation_v1
Config: configs/baseline/active.json
GPU: L40S 48 GB

Pull the repository, verify the commit, and run only your assigned command:
user1 / worker 0 -> bash scripts/submit_stage.sh baseline user1
user2 / worker 1 -> bash scripts/submit_stage.sh baseline user2
user3 / worker 2 -> bash scripts/submit_stage.sh baseline user3
user4 / worker 3 -> bash scripts/submit_stage.sh baseline user4
user5 / worker 4 -> bash scripts/submit_stage.sh baseline user5

Send back the Slurm job ID. If the job times out, rerun the same command.
Do not edit or pull a different commit until this run is complete.
```
