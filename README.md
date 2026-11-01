# Poisoned Paperwork

Targeted adversarial attacks and defenses for document vision-language models.
Receipt-total extraction is the primary task; resume highest-degree extraction
is the secondary case study.

For the terminal-only CARC workflow from the 100 collected receipts through
transfer, EOT patches, defenses, adaptive evaluation, and final analysis, see
[`docs/CARC_END_TO_END.md`](docs/CARC_END_TO_END.md). All GPU stages use L40S;
notebooks only read and visualize merged results.

## Status

| Stage | Status |
|---|---|
| Download, audit, and preprocess SROIE, CORD v2, and English resumes | Complete |
| Manual receipt-label repair | Complete |
| Qwen clean baselines and full-page attacks on public datasets | Complete |
| Four Qwen defenses on all public clean-correct documents | Complete: 4,688 records, zero failures |
| Donut clean baselines and attacks | In progress in the Donut worktree |
| Full defense-aware Qwen EOT/BPDA attack | Implemented for all 293 documents at ε=4,8,16; CARC run pending |
| Team-collected 100-receipt external test | Collection/annotation in progress |
| Transfer and localized-patch studies | Implemented; CARC runs pending |

## Current results

Qwen's clean accuracy is high on the three public validation sets: 96/100 on
CORD v2, 123/126 on SROIE, and 74/75 on resumes. All attack rates below are
therefore conditional on documents the model originally answered correctly.

| Dataset | ε=2 | ε=4 | ε=8 | ε=16 |
|---|---:|---:|---:|---:|
| CORD v2 | 0.0% | 62.5% | 87.5% | 89.6% |
| SROIE | 0.0% | 37.4% | 67.5% | 91.1% |
| Resumes | 0.0% | 1.4% | 5.4% | 23.0% |

This establishes a real, dataset-dependent robustness curve rather than merely
showing that perturbations can change an answer. Receipt totals are much easier
to target than resume degree classification, and ε=2 is a useful negative
control: it produced 0/293 targeted successes.

The completed defense study contains all 293 clean-correct documents at ε=4,
8, and 16. It evaluates exact JPEG recompression, randomized resize voting,
selective multi-view consistency, and cross-view/cross-prompt semantic
consistency. Aggregate post-defense targeted ASR is:

| Condition | ε=4 | ε=8 | ε=16 |
|---|---:|---:|---:|
| No defense (saved attack outputs) | 36.5% | 58.4% | 73.4% |
| JPEG Q85 | **4.4%** | **15.7%** | 41.0% |
| Resize vote | 11.6% | 22.5% | **39.9%** |
| Selective multi-view | 23.9% | 38.2% | 54.6% |
| Semantic consistency | 15.4% | 31.1% | 44.4% |

JPEG is the strongest tested defense at ε=4 and ε=8, while resize voting is
slightly stronger at ε=16. These are non-adaptive results: the original attack
did not know which defense would be applied.

## Direction

The next experiment is a full adaptive test, not a small demonstration. For
every one of the 293 public clean-correct documents and ε in {4, 8, 16}, the
attacker optimizes a bounded PNG using EOT/BPDA over the exact JPEG, resize,
blur, brightness, and semantic-prompt views used by the defenses. The final PNG
is then evaluated independently against all four defenses, including a strict
"beats all four" outcome. This determines how much of the apparent protection
survives when the attacker knows the defense.

After that, the same frozen methodology is applied to the independently
collected 100-receipt external set to address memorization concerns. Transfer
tests provide a separate black-box check, and Donut supplies a second model
family. The final study therefore reports clean utility, targeted ASR,
perceptibility, adaptive robustness, cross-model transfer, runtime, and exact
denominators—not only whether a perturbation changes one prediction.

The processed manifest contains 301 validation documents and 536 test
documents. The unified baseline produces 527 validation predictions and 997
test predictions because Qwen runs on every document while Donut runs only on
the 226 validation or 461 test receipts.

## Repository layout

```text
data/          local/CARC data; ignored by Git
notebooks/     data loading, exploration, preprocessing, and later analysis
src/           reusable model, attack, checkpoint, and GPU-stage code
configs/       tracked frozen experiment definitions
scripts/       CARC setup, Slurm launchers, smoke tests, and CPU merge tools
outputs/       checkpoints and results; ignored by Git
docs/          execution contract and detailed stage plans
```

Start with [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). The exact next action
and acceptance checklist are in [`docs/imm_next.md`](docs/imm_next.md).

## CARC setup

Clone or pull the repository in the shared project directory. Download and
preprocess data only if `data/processed/master_manifest.parquet` is absent.

```bash
bash scripts/setup_carc.sh
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

The baseline is designed for one L40S 48 GB GPU, 8 CPU cores, 32 GB system RAM,
and up to 4 hours per production worker. All project GPU jobs use L40S so model
settings and hardware remain consistent between workers.

## Baseline smoke test

Only `user1` runs the first smoke test. Request an interactive L40S from a CARC
login node:

For an unattended run that survives closing the laptop, submit the smoke test
as an L40S Slurm batch job:

```bash
bash scripts/submit_smoke_baseline.sh 02:00:00
```

The command prints a job ID and log paths, then it is safe to disconnect. Check
the result later with `squeue`, `sacct`, and the saved logs. Do not keep an
interactive `salloc` request queued at the same time.

For an attended interactive run instead:

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

The smoke test runs two documents from each dataset: 6 Qwen predictions and 4
Donut predictions. Every prediction is repeated for determinism. Inspect:

```text
outputs/baseline/baseline_smoke_v1/merged/predictions.csv
outputs/baseline/baseline_smoke_v1/merged/failures.csv
outputs/baseline/baseline_smoke_v1/merged/summary.json
```

Do not start production if there is an inference failure, a nondeterministic
response, repeated empty output, an unexpected page count, or a GPU memory
error. Exit twice when finished to release the interactive allocation.

## Five-person production baseline

After the smoke output is reviewed, each teammate submits exactly one worker
from the normal CARC login node. Do **not** request an interactive GPU first;
the launcher submits the resource request itself.

```bash
bash scripts/submit_stage.sh baseline user1
bash scripts/submit_stage.sh baseline user2
bash scripts/submit_stage.sh baseline user3
bash scripts/submit_stage.sh baseline user4
bash scripts/submit_stage.sh baseline user5
```

The commands may run concurrently. Each worker receives a disjoint document
shard, writes an atomic record after each inference, and safely resumes after a
timeout. The fifth completed worker triggers the strict merge automatically.
The merge can also be checked from a CPU login node:

```bash
python scripts/merge_baseline.py --config configs/baseline/active.json
```

Production is accepted only when the merged summary reports exactly 527 rows
and all five workers used the same config hash and Git commit.

## Rules that protect the experiment

- Do not edit `configs/baseline/active.json` while jobs are queued or running.
- Use a new `run_id` whenever code-independent experiment settings change.
- Do not inspect the test split until the validation prompt and settings are frozen.
- Do not commit `data/`, `.cache/`, `outputs/`, logs, or model weights.
- Analysis notebooks read saved outputs; they do not rerun GPU inference.
