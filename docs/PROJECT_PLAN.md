# Project implementation plan

This is the index for the remaining project work. GPU computation runs as
checkpointed Python programs submitted to USC CARC. Notebooks are reserved for
data preparation, inspection, analysis, and figures.

Read [the execution contract](00_execution_contract.md) before implementing or
running a GPU stage.

## Final architecture

```text
configs/                         # tracked, frozen JSON experiment configs
src/                             # reusable project code
├── models/                      # Qwen, Donut, and InternVL adapters
├── attacks/                     # PGD, EOT, projection, and transforms
├── stages/                      # one run(config, worker_id, ...) per GPU stage
├── checkpointing.py
├── metrics.py
├── sharding.py
└── team.py
scripts/
├── submit_stage.sh              # teammate-facing CARC command
├── run_gpu_stage.py             # dispatches the requested stage
└── slurm/gpu_stage.sbatch       # shared Slurm job body
notebooks/
├── 01_data_loading.ipynb
├── 02_data_exploration.ipynb
├── 03_data_preprocessing.ipynb
├── 04_baseline_analysis.ipynb
├── 05_attack_analysis.ipynb
└── 06_defense_and_final_analysis.ipynb
```

The first three notebooks already prepare the data. The final three notebooks
are CPU analysis notebooks and never contain a multi-hour model loop.

## Fixed team identities

The launcher converts each easy-to-remember user ID into one deterministic
worker shard:

| User ID | Team member | Worker ID |
|---|---|---:|
| `user1` | Prabudhd | 0 |
| `user2` | Gary | 1 |
| `user3` | Saaketh | 2 |
| `user4` | Khalid | 3 |
| `user5` | Shail | 4 |

For every large GPU stage, all five people run the same command with only their
own user ID changed. Nobody chooses record IDs or edits code while a run is in
progress.

Example after the stage owner has created the active config and implementation:

```bash
# Prabudhd
bash scripts/submit_stage.sh qwen_attack user1

# Gary
bash scripts/submit_stage.sh qwen_attack user2
```

The remaining three teammates use `user3`, `user4`, and `user5`. The launcher
requests the correct GPU profile, assigns the shard, writes separate logs, and
passes the same frozen run ID to every worker.

## GPU stage roadmap

| Order | Stage command | Detailed plan | Recommended GPU | Analysis notebook |
|---:|---|---|---|---|
| 1 | `baseline` | [Clean baselines](04_clean_baselines.md) | L40S 48 GB | 04 |
| 2 | `donut_attack` | [Donut targeted attack](05_donut_attack.md) | L40S 48 GB | 05 |
| 3 | `qwen_attack` | [Qwen full-page attacks](06_qwen_attacks.md) | A100 80 GB | 05 |
| 4 | `transfer_robustness` | [Transfer and digital robustness](07_transfer_and_digital_robustness.md) | L40S 48 GB | 05 |
| 5 | `eot_patch` | [EOT patch and physical pilot](08_eot_patch_and_physical_pilot.md) | A100 80 GB | 05 |
| 6 | `defenses` | [Defenses and adaptive attacks](09_defenses_and_adaptive_attacks.md) | A100 80 GB | 06 |
| 7 | CPU analysis | [Final results and figures](10_results_and_figures.md) | No GPU | 06 |

## Notebook responsibilities

### 04 - Baseline analysis

Inputs: merged `baseline` outputs.

Displays clean accuracy, parsing failures, latency, model examples, and the
clean-correct document counts that become attack candidates. It must not load a
VLM or generate missing predictions.

### 05 - Attack analysis

Inputs: merged outputs from `donut_attack`, `qwen_attack`,
`transfer_robustness`, and `eot_patch`.

Displays attack success versus epsilon, LPIPS/SSIM, save/reload survival,
InternVL transfer, transformation robustness, EOT versus non-EOT patches, and
the physical pilot. It may display images already produced by the scripts but
must not optimize images itself.

### 06 - Defense and final analysis

Inputs: all frozen merged results, especially `defenses`.

Displays detector performance, clean accuracy versus defended attack success,
adaptive attacks, routing/latency costs, final tables, final figures, and the
proposal-question matrix. It is the only notebook that writes tracked final
artifacts under `reports/final/`.

## Stage-owner workflow

Before teammates submit a stage, one assigned stage owner must:

1. Implement `src/stages/<stage>.py` with a callable
   `run(config, config_path, user_id, worker_id, num_workers)`.
2. Reuse common model, attack, metric, sharding, and checkpoint helpers.
3. Create and review `configs/<stage>/active.json`.
4. Run a two-document smoke test in an interactive CARC GPU allocation.
5. Verify resume after interruption.
6. Commit and push the exact code/config revision.
7. Tell all five users the stage name and Git commit.

Only then do all five teammates pull and submit their one worker command.

## What “done” means

A GPU stage is complete only when:

1. All five worker completion markers exist.
2. An interrupted worker can restart using the same command.
3. Every expected experiment key is present once.
4. Model, config, prompt, data, and Git revisions match across workers.
5. Failures are explicitly recorded.
6. The CPU merge produces Parquet, CSV, and `summary.json` outputs.
7. The appropriate analysis notebook can render its section without loading a
   model.

## Scope priority if CARC time is limited

1. Qwen clean receipt baseline.
2. Donut attack validation.
3. Qwen receipt PGD at epsilon 2, 4, 8, and 16.
4. Save/reload, InternVL transfer, and digital robustness.
5. Three defenses and adaptive attacks.
6. Resume case study.
7. EOT patch and physical pilot extensions.

Any reduced scope must be recorded in the final report rather than silently
removed from the result tables.
