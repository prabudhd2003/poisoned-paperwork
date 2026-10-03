# Poisoned Paperwork

Targeted adversarial attacks and defenses for document vision-language models.
Receipt-total extraction is the primary task; resume highest-degree extraction
is the secondary case study.

## Status

| Stage | Status |
|---|---|
| Download, audit, and preprocess SROIE, CORD v2, and English resumes | Complete |
| Manual receipt-label repair | Complete |
| Unified Qwen + Donut clean baseline | Implemented; CARC GPU smoke test required |
| Donut targeted attack | Implemented scaffold; do not run until the baseline is accepted |
| Qwen attacks, transfer tests, EOT, and defenses | Planned in `docs/` |

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
and up to 4 hours per production worker. An A40 48 GB is a reasonable fallback,
but do not mix model settings or configs between workers.

## Baseline smoke test

Only `user1` runs the first smoke test. Request an interactive L40S from a CARC
login node:

For an unattended run that survives closing the laptop, submit the smoke test
as a Slurm batch job (A100, A40, and L40S are supported):

```bash
bash scripts/submit_smoke_baseline.sh a100 02:00:00
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
