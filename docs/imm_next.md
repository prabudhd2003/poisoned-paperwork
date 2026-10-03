# Immediate next task: accept the clean baseline on CARC

The Qwen and Donut implementations are integrated into one checkpointed stage.
No teammate should submit the 527-row production validation run until the
owner-only GPU smoke test below passes and its saved predictions are reviewed.

## What is implemented

- Qwen2.5-VL-3B-Instruct, pinned to revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`, runs on SROIE, CORD v2, and all
  retained pages of each resume.
- Donut DocVQA, pinned to revision
  `b19d2e332684b0e2d35d9144ce34047767335cf8`, runs on SROIE and CORD v2.
- The real processed resume task name, `resume_highest_degree`, is used.
- Documents are sorted and divided among workers. Both model experiments for a
  receipt stay on the same worker.
- One model is loaded at a time. It is unloaded before the second model loads.
- Every inference becomes an atomic JSON checkpoint, including explicit failed
  records. A restart skips only records matching the config, Git commit, model
  revision, task, prompt, split, and exact experiment key.
- The merge requires disjoint complete assignments, identical config hashes
  and Git commits, and the exact expected experiment-key set.
- Validation requires 527 records: 301 Qwen and 226 Donut.
- Frozen test requires 997 records: 536 Qwen and 461 Donut.
- Correct/incorrect ID lists are model-specific, so later attack stages cannot
  accidentally mix Qwen and Donut eligibility.

## Files to review before using GPU time

1. `configs/baseline/smoke.json`
2. `configs/baseline/active.json`
3. `src/stages/baseline.py`
4. `src/models/qwen.py`
5. `src/models/donut.py`
6. `src/data.py`
7. `src/checkpoints.py`
8. `src/normalization.py`
9. `scripts/smoke_baseline.py`
10. `scripts/merge_baseline.py`
11. `scripts/submit_stage.sh`

## Step 1: update and run CPU checks

From the shared CARC repository, verify that Git is clean and that the processed
manifest exists:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull origin main
git status --short
test -f data/processed/master_manifest.parquet && echo "processed data found"
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
bash scripts/submit_stage.sh baseline user1 --dry-run
```

The dry run must show one `l40s` GPU, 8 CPUs, 32 GB RAM, a four-hour limit, and
worker 0 of 5. It does not submit a job.

## Step 2: owner-only interactive smoke test

Run from a CARC login node:

```bash
salloc --account=yzhao010_1531 --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=32G --time=02:00:00 --gpus-per-task=l40s:1
srun --pty bash -l
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
nvidia-smi
python scripts/smoke_baseline.py --config configs/baseline/smoke.json
```

This performs 10 experiments on one GPU:

| Dataset | Qwen | Donut | Total |
|---|---:|---:|---:|
| SROIE | 2 | 2 | 4 |
| CORD v2 | 2 | 2 | 4 |
| Resumes | 2 | 0 | 2 |
| Total | 6 | 4 | 10 |

Every experiment is inferred twice. The script stops on nondeterminism and
fails acceptance when an inference error is recorded. It automatically merges
the one-worker smoke output.

## Step 3: inspect smoke artifacts

```bash
column -s, -t < outputs/baseline/baseline_smoke_v1/merged/predictions.csv | less -S
cat outputs/baseline/baseline_smoke_v1/merged/summary.json
cat outputs/baseline/baseline_smoke_v1/merged/failures.csv
```

Confirm all of the following:

- 10 total rows, split as 6 Qwen and 4 Donut;
- no inference failures;
- every `deterministic` value is true;
- SROIE outputs normalize to two decimals;
- CORD outputs normalize to integer rupiah strings;
- resume outputs are exactly one allowed degree label or an explicit parse
  failure, never silently guessed;
- resume `input_page_count` matches the saved pages;
- raw answers, normalized answers, targets, runtime, and peak memory look sane;
- the L40S did not approach its 48 GB limit.

Parse failures or wrong predictions are model results and may remain, but they
must be understandable from the raw responses. Inference failures, empty-output
streaks, nondeterminism, missing pages, and out-of-memory errors block production.

## Step 4: prove restart behavior

Run the same smoke command again. It should reuse all 10 matching checkpoints,
avoid loading either model, and finish quickly. If the current Git commit or
config differs, it must rerun rather than trust stale records.

## Step 5: submit the five production workers

Exit the interactive node. From a normal CARC login shell, all teammates may
submit concurrently:

```bash
bash scripts/submit_stage.sh baseline user1
bash scripts/submit_stage.sh baseline user2
bash scripts/submit_stage.sh baseline user3
bash scripts/submit_stage.sh baseline user4
bash scripts/submit_stage.sh baseline user5
```

No one runs `salloc` before these commands. `submit_stage.sh` requests the GPU
and other resources from Slurm. Monitor with `squeue -u "$USER"` and inspect
`logs/pp_baseline_w<worker>_<job>.out` and `.err`.

If a job times out, that same teammate runs the same command again. Completed
records are retained; only missing or stale records run. Do not change the
config or `run_id` between retries.

## Step 6: merge and accept validation

The final worker tries the merge automatically. It can also be checked on a CPU
login node:

```bash
python scripts/merge_baseline.py --config configs/baseline/active.json
cat outputs/baseline/baseline_validation_v1/merged/summary.json
```

Acceptance requires exactly 527 records, five correct completion markers, one
Git commit, one config hash, no missing/duplicate keys, and the four files:

```text
qwen_clean_correct_ids.txt
qwen_clean_incorrect_ids.txt
donut_clean_correct_ids.txt
donut_clean_incorrect_ids.txt
```

Only after acceptance should the team create `04_baseline_analysis.ipynb` and
activate the attack stages. Test remains untouched until validation decisions
are frozen.
