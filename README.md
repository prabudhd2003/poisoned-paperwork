# Poisoned Paperwork

Targeted adversarial attacks and defenses for document vision-language models.
Receipt-total extraction is the primary task; resume highest-degree extraction
is the secondary case study.

## Status

| Stage | Status |
|---|---|
| Download, audit, and preprocess SROIE, CORD v2, and English resumes | Complete |
| Manual receipt-label repair | Complete |
| Unified Qwen + Donut clean validation baseline | **Accepted:** 527 predictions, five workers, zero inference failures |
| Baseline analysis | Complete in `notebooks/04_baseline_analysis.ipynb` |
| Donut targeted attack | Smoke-ready; production config is gated pending smoke review |
| Qwen attacks | Router/launchers present; shared core and track modules are owned by the assigned Qwen developers |
| Transfer, EOT, and defenses | Planned in `docs/`; not implemented |
| Prospective external receipts | 11 of 100 collected and annotated; preprocessing notebook complete |

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

Start with [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). The proposal feedback
and response are in [`docs/01_proposal_feedback_response.md`](docs/01_proposal_feedback_response.md).
The exact next action and Donut acceptance checklist are in
[`docs/imm_next.md`](docs/imm_next.md).

## Accepted validation baseline

The authoritative CARC run is `baseline_validation_v1`, produced from Git
commit `5e387062d29b2a9b2b79de03c305c52d7b97f01f`. Its merged predictions have
SHA-256 `483e12b88167555d94e943d0179314de25016333e409417b34a6f922c46caeb0`.
The analysis verified 527 records on 301 documents from all five workers.
Qwen was correct on 96/100 CORD receipts, 123/126 SROIE receipts, and 74/75
resumes; Donut was correct on 75/100 CORD and 46/126 SROIE receipts. Later
attacks consume these saved predictions by hash and never silently rerun or
replace the baseline.

## CARC setup

Clone or pull the repository in the shared project directory. Download and
preprocess data only if `data/processed/master_manifest.parquet` is absent.

```bash
bash scripts/setup_carc.sh
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
```

The accepted baseline should not be rerun. Donut attacks use one L40S 48 GB GPU,
8 CPU cores, and 48 GB RAM per worker.

## Donut attack: next CARC action

Khalid (`user4`) owns the isolated Donut smoke. From the repository root, run
the CPU preflight and then submit the 20-receipt smoke job:

```bash
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python scripts/smoke_donut_attack.py \
  --config configs/donut_attack/smoke_validation.json --preflight
sbatch scripts/slurm/donut_smoke.sbatch \
  configs/donut_attack/smoke_validation.json
```

After the smoke completes, inspect processor parity, optimization traces,
gradient norms, bounds, saved/reloaded predictions, perceptibility metrics, and
checkpoint resume. Only then change `ready_for_production` in
`configs/donut_attack/active.json` from `false` to `true`, commit that reviewed
change, and ask all five teammates to submit:

```bash
bash scripts/submit_stage.sh donut_attack user1
bash scripts/submit_stage.sh donut_attack user2
bash scripts/submit_stage.sh donut_attack user3
bash scripts/submit_stage.sh donut_attack user4
bash scripts/submit_stage.sh donut_attack user5
```

The same command safely resumes a timed-out worker. Check the strict CPU merge
manually with:

```bash
python scripts/merge_donut_attack.py --config configs/donut_attack/active.json
```

## Rules that protect the experiment

- Do not edit an active config while jobs are queued or running.
- Use a new `run_id` whenever code-independent experiment settings change.
- Do not inspect the test split until the validation prompt and settings are frozen.
- Do not commit `data/`, `.cache/`, `outputs/`, logs, or model weights.
- Store collected raw receipts on CARC rather than public Git; they may contain
  names, transaction identifiers, and other personal information.
- Analysis notebooks read saved outputs; they do not rerun GPU inference.
