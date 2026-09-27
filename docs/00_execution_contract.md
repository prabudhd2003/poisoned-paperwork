# 00 - Shared execution contract

This contract applies to notebooks 04-10 and to any helper code created for
them. An LLM or teammate implementing a notebook should treat every item marked
**required** as an acceptance criterion.

## Fixed project choices

### Data

Use only the processed datasets created by `03_data_preprocessing.ipynb`:

```text
data/processed/sroie/
data/processed/cord_v2/
data/processed/resume_parsing_vision/
data/processed/master_manifest.parquet
```

Never modify `data/sroie`, `data/cord_v2`, or
`data/resume_parsing_vision`. Never derive experimental splits again in a
later notebook.

Use validation splits to select prompts, thresholds, attack hyperparameters,
and defense settings. Use test splits only after those choices are frozen.

### Models

Use these exact Hugging Face repositories and pin the listed revision in every
`from_pretrained` call:

| Role | Model ID | Frozen revision |
|---|---|---|
| Primary white-box model | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Reproduction/debug model | `naver-clova-ix/donut-base-finetuned-docvqa` | `b19d2e332684b0e2d35d9144ce34047767335cf8` |
| Transfer-only model | `OpenGVLab/InternVL2_5-4B` | `2cf4a8158bbc40d35015e7c63b527890de4d27b3` |

If a pinned revision cannot load, stop and document the incompatibility. Do not
silently switch models or use `revision="main"`.

### Tasks and targets

- Receipt clean target: normalized final total from `clean_target`.
- Receipt adversarial target: ten times the true total from `attack_target`.
- Resume clean target: one of the seven canonical degree levels.
- Resume adversarial target: the next higher level from `attack_target`.
- Documents with `usable=False` are never modeled.
- Documents with `attack_eligible=False` may be used for clean accuracy but not
  targeted attack metrics.

### Evaluation definitions

- **Clean exact-match accuracy:** clean normalized prediction equals
  `clean_target`.
- **Conditional attack success rate (ASR):** exact adversarial target among
  documents that are usable, attack-eligible, and clean-correct for that same
  model and configuration.
- **Harmful-output rate:** exact adversarial target divided by all usable test
  documents for the task, including clean-incorrect documents.
- **Transfer rate:** exact target on InternVL among Qwen-generated adversarial
  examples, reported both over all attempted attacks and over attacks that
  succeeded on Qwen.
- **Saved-image success:** success must be recomputed after saving the attacked
  image as lossless PNG, closing it, and loading it again.
- **Perceptibility:** report L-infinity in 0-255 pixel units, LPIPS, and SSIM.

Always store the raw model text and the conservatively normalized prediction.
Never change a metric by manually correcting a model output.

## Image-processing rules

Original images remain unchanged on disk. Runtime processing is model-specific:

- Qwen uses its official processor with aspect ratio preserved. Begin with
  `min_pixels = 256 * 28 * 28` and `max_pixels = 1280 * 28 * 28`, batch size 1,
  and deterministic generation. If memory requires a smaller limit, choose it
  on validation and freeze it in the run configuration.
- Donut uses its official processor and checkpoint input size. Do not manually
  stretch images outside the processor.
- InternVL uses its official dynamic image preprocessing from the pinned model
  repository.
- Convert inputs to RGB and apply EXIF orientation consistently.
- Random transforms are forbidden in clean baselines. They are used only in
  experiments that explicitly test randomness or EOT.

The exact same deterministic clean pipeline must be used before and after an
attack. Keep transformations differentiable during optimization when required,
but always verify the final saved file through the real inference pipeline.

## Notebook structure

Every GPU notebook must follow this section order:

1. **Purpose and success criteria** - markdown only.
2. **Configuration** - run ID, split, task, model revision, seed, and worker
   settings in one visible cell.
3. **CPU preflight** - paths, input counts, schema, Git commit, and disk space.
4. **GPU preflight** - allocated GPU, CUDA, model load, and one-example smoke
   test. Fail before the long run if any check is wrong.
5. **Shared functions** - import tested helpers; do not copy five versions of
   model or metric code.
6. **Worker 0** - calls the common runner with `worker_id=0`.
7. **Worker 1** - calls the common runner with `worker_id=1`.
8. **Worker 2** - calls the common runner with `worker_id=2`.
9. **Worker 3** - calls the common runner with `worker_id=3`.
10. **Worker 4** - calls the common runner with `worker_id=4`.
11. **CPU merge and validation** - combines shards only after completion.
12. **Summary** - tables, small plots, failures, and the next-stage path.

The five worker cells call the same function. They differ only in worker ID.
Do not maintain five separate copies of attack logic.

## Five-person CARC execution

One person is the **stage owner**. That person implements and tests the notebook
on a tiny smoke subset, freezes the config, commits it, and tells the other four
people the exact Git commit and `RUN_ID`.

Each teammate then:

1. Pulls the same Git commit on CARC.
2. Requests their own GPU allocation.
3. Uses one unique worker ID from 0 through 4.
4. Runs only the common setup and their assigned worker cell.
5. Does not edit or save executed output into the source notebook.
6. Reports the worker completion file to the stage owner.

Environment variables should control worker selection:

```bash
export PP_RUN_ID=qwen_receipt_pgd_v1
export PP_NUM_WORKERS=5
export PP_WORKER_ID=0        # unique value 0, 1, 2, 3, or 4
```

The implementation must also support a future Slurm array using
`SLURM_ARRAY_TASK_ID`, but the notebook variable takes precedence when set.

### Deterministic sharding

Sort eligible `record_id` values. Assign the item at position `i` to
`i % PP_NUM_WORKERS`. Do not use Python's built-in `hash()`, because it changes
between processes. Save every worker's assigned record IDs before model loading:

```text
outputs/<stage>/<run_id>/assignments/worker_00.txt
...
outputs/<stage>/<run_id>/assignments/worker_04.txt
```

The merge step must assert that the five assignments are disjoint and their
union equals the expected input set.

## Checkpoint and timeout requirements

CARC jobs may end without warning. GPU work must therefore be resumable.

### Directory contract

```text
outputs/<stage>/<run_id>/
├── config.json
├── environment.json
├── assignments/
│   └── worker_00.txt
├── shards/
│   └── worker_00/
│       ├── records/          # one final JSON per experiment key
│       ├── images/           # adversarial images when applicable
│       ├── checkpoints/      # in-progress attack state
│       ├── failures.jsonl
│       ├── progress.json
│       └── WORKER_COMPLETE.json
└── merged/                   # written only by the merge step
```

`outputs/`, `results/`, and `checkpoints/` are gitignored and live in the shared
CARC project space. Model downloads belong in `.cache/`, which is also
gitignored. Never store model weights in the home directory or in Git.

### Atomic records

- The unit of completion is one unique experiment key, not a whole notebook.
- Write `<key>.json.tmp`, flush and close it, then rename it to `<key>.json`.
- A final record is never overwritten unless `FORCE=True` is explicitly set.
- On restart, skip valid final records and continue missing work.
- Attack notebooks save the current image tensor, iteration, optimizer state,
  and random generator state every 10 optimization steps.
- A worker writes `WORKER_COMPLETE.json` only after validating all assigned
  experiment keys.
- The merge step refuses to run when a worker completion marker is missing,
  unless the operator passes a clearly recorded partial-run override.

### Experiment key

Use a filesystem-safe key containing all factors that distinguish outputs:

```text
<record_id>__<model>__<task>__<attack>__eps-<value>__seed-<value>
```

For clean baselines, omit attack and epsilon. For transformation or defense
runs, add the transformation/defense name and replicate number.

## Configuration and provenance

Before a full run, create a tracked JSON config under `configs/<stage>/`. Copy
that config unchanged into the run output. Every output record must contain:

- `run_id` and stage;
- repository Git commit;
- model ID and revision;
- dataset, split, `record_id`, `source_split`, and `source_index`;
- task, clean target, attack target, and eligibility flags;
- prompt ID and prompt text hash;
- processor settings;
- attack, transformation, or defense settings when applicable;
- global seed, worker ID, start/end timestamps, and runtime seconds;
- raw response, normalized prediction, and metric values;
- status and any error message.

Save `environment.json` with Python, PyTorch, CUDA, Transformers, Datasets,
Pillow, GPU model, and GPU driver versions.

## Shared code rule

Notebooks should explain and orchestrate the experiment. Reusable logic belongs
under `src/poisoned_paperwork/`, with small tests under `tests/`. At minimum,
later implementation should provide shared helpers for:

- dataset loading and schema checks;
- prompts and output normalization;
- model loading and inference;
- deterministic sharding and atomic checkpoint I/O;
- attacks and transformations;
- metrics and result merging.

An LLM implementing a notebook must inspect existing helpers before creating a
new one. Do not duplicate subtly different normalization or metric functions.

## GPU and safety preflight

All model inference, feature extraction, and gradient attacks run on CARC GPU
nodes, never on the login node. Before a long job, print and save:

- hostname and Slurm job ID;
- GPU name and free memory;
- worker ID and assigned count;
- run ID, Git commit, and config SHA-256;
- model revision actually loaded;
- first and last assigned record IDs;
- estimated output disk usage.

Use mixed precision only when the method supports correct gradients. Never use
4-bit or 8-bit quantization for white-box image-gradient attacks unless a
separate validation proves the attack objective remains differentiable.

## Merge validation

Every stage merge must check:

1. No duplicate experiment keys.
2. No missing expected keys.
3. All record IDs belong to the frozen split.
4. No train/validation/test leakage.
5. Every record uses the expected model revision and config hash.
6. Numeric metrics are finite and within valid ranges.
7. Saved image paths exist and their SHA-256 hashes match the record.
8. Failures are counted and displayed rather than silently removed.

Write merged machine-readable results as both Parquet and CSV, plus a compact
`summary.json`. Small final tables and figures selected by notebook 10 will be
copied to tracked `reports/final/`; raw experiment outputs stay gitignored.
