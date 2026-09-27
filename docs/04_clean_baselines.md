# 04 - Clean model baselines

## GPU stage and analysis notebook

- Stage command: `baseline`
- Implementation: `src/stages/baseline.py`
- Active config: `configs/baseline/active.json`
- CPU analysis: `notebooks/04_baseline_analysis.ipynb`

## Purpose

Establish how accurately each attack model answers unmodified documents and
create the exact clean-correct sets used by later attacks. This notebook also
freezes prompts, output normalization, runtime image processing, and model
revisions. No adversarial optimization occurs here.

**GPU required:** yes, on CARC. Merging and plotting can run on CPU.

## Questions answered

1. How accurately does Qwen answer receipt totals on SROIE and CORD?
2. How accurately does Qwen classify the highest resume degree?
3. How accurately does Donut answer receipt totals before attack reproduction?
4. Which validation and test documents are clean-correct for each model?
5. What are the runtime and failure rates for each model/task combination?

## Inputs

```text
data/processed/sroie/{train,validation,test}/
data/processed/cord_v2/{train,validation,test}/
data/processed/resume_parsing_vision/{train,validation,test}/
data/processed/master_manifest.parquet
```

Required fields are `record_id`, image or images, `task`, `clean_target`,
`attack_target`, `usable`, `attack_eligible`, `source_split`, and
`source_index`. Assert the processed counts from the README before loading a
model.

The first development pass uses validation only:

- 126 SROIE receipts;
- 100 CORD receipts;
- 75 resumes.

After prompts and processor settings are frozen, run the same configuration on
the 536 held-out test documents.

## Models

### Qwen baseline

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Precision: BF16 when the allocated GPU supports it; otherwise FP16.
- Batch size: 1 initially.
- Generation: greedy, `do_sample=False`, temperature omitted,
  `max_new_tokens=32`.
- Processor: official processor, RGB inputs, aspect ratio preserved,
  `min_pixels=256*28*28`, `max_pixels=1280*28*28` initially.
- Resumes: provide all retained pages for one resume in their original order in
  a single multi-image conversation.

### Donut baseline

- Model: `naver-clova-ix/donut-base-finetuned-docvqa`
- Revision: `b19d2e332684b0e2d35d9144ce34047767335cf8`
- Scope: SROIE and CORD receipt totals only.
- Precision: FP32 for the first correctness smoke test, then FP16 if outputs
  match on a fixed 20-document comparison set.
- Batch size: 1.
- Processor and input size: use the checkpoint's official processor settings.
- Generation: deterministic and long enough to include the short numeric
  answer; retain the complete decoded response.

Do not run Donut on resumes unless the project explicitly adds and validates a
resume-specific Donut checkpoint.

## Frozen prompts

Start with these prompt IDs and exact strings:

```text
receipt_total_v1:
What is the final total amount shown on this receipt? Return only the amount,
with no currency symbol, label, or explanation.

resume_degree_v1:
What is the highest degree level stated in this resume? Return exactly one
label from: secondary, certificate_or_diploma, associate, bachelor,
postgraduate, master, doctorate.
```

The Donut task string uses the receipt question inside the checkpoint's
required DocVQA special-token template.

Prompt changes are allowed only while using validation data. If a prompt is
changed, create a new prompt ID rather than silently editing `v1`. Save all
prompt candidates and validation scores. Freeze exactly one prompt per task
before test inference.

## Output normalization

Store `raw_response` before normalization.

- SROIE: remove surrounding whitespace, an optional currency symbol, and an
  optional leading label; parse one amount and format with two decimals.
- CORD: remove surrounding whitespace, an optional currency symbol, and
  thousands separators; parse one non-negative integer rupiah amount.
- Resumes: lowercase, trim whitespace, and accept only an exact member of the
  seven-label vocabulary. Do not infer a label from a long explanation.
- If there is no single unambiguous value, set `prediction_normalized=null` and
  `parse_status="failed"`.

Normalization code must be shared by every later notebook and unit-tested with
representative strings.

## Stage implementation procedure

### 1. CPU preflight

1. Find the repository root portably.
2. Load only metadata first and assert counts and fields.
3. Confirm there is no overlap between experiment splits.
4. Create tracked configs:
   - `configs/04_clean_baselines/qwen_v1.json`
   - `configs/04_clean_baselines/donut_v1.json`
5. Create deterministic worker assignments.
6. Refuse test execution unless the config has `frozen_for_test=true`.

### 2. GPU smoke tests

For each model:

1. Load the exact revision.
2. Run two SROIE, two CORD, and, for Qwen, two resume examples.
3. Display input dimensions, processed tensor shape, raw answer, normalized
   answer, target, runtime, and peak GPU memory.
4. Confirm two consecutive runs produce identical normalized outputs.
5. Stop if the model produces only empty or malformed responses.

### 3. Validation prompt selection

Run Qwen prompt candidates over all validation documents. Run Donut's fixed
DocVQA question over validation receipts. Compare exact-match accuracy and
parse failure rate. Prefer the shortest prompt among statistically similar
choices. Do not tune prompts per individual dataset example.

### 4. Frozen baseline run

For each usable document and model/task configuration:

1. Run deterministic clean inference.
2. Save the raw and normalized prediction immediately.
3. Record exact-match correctness, runtime, GPU memory, and any exception.
4. Resume by skipping valid completed record files.
5. Repeat the frozen run on test only after the stage owner signs off on the
   validation summary.

## Five-worker CARC plan

Workers shard sorted document IDs according to the shared execution contract.
Each worker runs these jobs sequentially for its assigned documents:

1. Qwen receipt baseline.
2. Qwen resume baseline.
3. Donut receipt baseline.

If startup cost is too high, use three independent run IDs and let workers
complete the Qwen-receipt, Qwen-resume, and Donut-receipt runs separately. A
run is complete only when all five shards for that run are merged.

After the stage owner freezes and pushes the implementation and config, each
teammate submits exactly one command:

```bash
bash scripts/submit_stage.sh baseline user1  # Prabudhd
bash scripts/submit_stage.sh baseline user2  # Gary
bash scripts/submit_stage.sh baseline user3  # Saaketh
bash scripts/submit_stage.sh baseline user4  # Khalid
bash scripts/submit_stage.sh baseline user5  # Shail
```

The launcher requests one L40S 48 GB GPU, 8 CPUs, 32 GB system RAM, and four
hours for each worker. The baseline analysis notebook is opened only after all
five worker outputs merge.

## Exact outputs

Raw resumable outputs:

```text
outputs/baseline/<run_id>/
├── config.json
├── environment.json
├── assignments/
├── shards/worker_00/records/*.json
└── merged/
    ├── predictions.parquet
    ├── predictions.csv
    ├── clean_correct_ids.txt
    ├── clean_incorrect_ids.txt
    ├── failures.csv
    └── summary.json
```

Each prediction record must contain:

```text
run_id, record_id, dataset_name, experiment_split, model_id, model_revision,
task, prompt_id, clean_target, attack_target, attack_eligible, raw_response,
prediction_normalized, parse_status, clean_correct, runtime_seconds,
peak_gpu_memory_mb, worker_id, seed, git_commit, status, error
```

## Results shown in `04_baseline_analysis.ipynb`

1. Exact-match accuracy and 95% bootstrap confidence interval by model,
   dataset, task, and split.
2. Number and percentage of clean-correct, clean-incorrect, parse-failed, and
   inference-failed examples.
3. Median and p95 runtime plus peak GPU memory.
4. A small error table with up to 20 examples, not thousands of raw rows.
5. Counts of attack-eligible and clean-correct documents passed to notebooks
   05 and 06.

Example summary shape:

| Model | Dataset | Split | N | Exact match | Parse failures | Attack candidates |
|---|---|---|---:|---:|---:|---:|
| Qwen | SROIE | validation | 126 | computed | computed | computed |
| Qwen | CORD | validation | 100 | computed | computed | computed |
| Qwen | Resumes | validation | 75 | computed | computed | computed |
| Donut | SROIE | validation | 126 | computed | computed | computed |
| Donut | CORD | validation | 100 | computed | computed | computed |

## Completion criteria

- Exact model revisions and prompts are recorded.
- Every usable validation document has one final prediction per intended model.
- No duplicate or missing experiment keys remain after merge.
- Output normalization tests pass.
- Qwen and Donut produce deterministic answers on the smoke set.
- Test data is untouched until prompt and processor configs are frozen.
- Clean-correct ID files exist for the `donut_attack` and `qwen_attack` stages.
- Failures and parse failures are reported, never silently discarded.
