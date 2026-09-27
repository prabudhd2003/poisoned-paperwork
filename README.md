# Poisoned Paperwork

Targeted adversarial attacks and defenses for document vision-language models.
The primary task is receipt-total extraction; resume highest-degree extraction
is a secondary case study.

| Role | Model |
|---|---|
| Primary white-box model | Qwen2.5-VL-3B-Instruct |
| Attack reproduction/debugging | Donut DocVQA |
| Transfer-only model | InternVL2.5-4B |

## Current status

The data pipeline is complete and ready for modeling.

| Stage | File | Status |
|---|---|---|
| Download pinned datasets | `notebooks/01_data_loading.ipynb` | Done |
| Explore and audit all documents | `notebooks/02_data_exploration.ipynb` | Done |
| Review receipt labels | `annotations/receipt_total_labels.csv` | Done |
| Create modeling-ready datasets | `notebooks/03_data_preprocessing.ipynb` | Done |
| Implement and run clean baselines | `baseline` GPU stage | **Next** |
| Donut, Qwen, transfer, EOT, and defense experiments | GPU stage scripts | Planned |
| Analyze results | notebooks 04-06 | Planned |

### Modeling-ready data

| Dataset | Train | Validation | Test | Usable documents |
|---|---:|---:|---:|---:|
| SROIE | 500 | 126 | 361 | 987 |
| CORD v2 | 798 | 100 | 100 | 998 |
| English resumes | 850 | 75 | 75 | 1,000 |
| **Total** | **2,148** | **301** | **536** | **2,985** |

All images decode. Two CORD receipts were excluded because their answers require
arithmetic rather than extraction, seven blank resume pages were removed, and
83 raw degree strings were mapped to seven canonical levels. Raw datasets are
never modified.

## How the remaining project is organized

Long GPU computation runs as Python scripts submitted to Slurm. Notebooks are
used only to inspect completed outputs, analyze metrics, and build figures.

```text
configs/                         # frozen JSON experiment settings
src/
├── models/                      # model-specific adapters (planned)
├── attacks/                     # PGD/EOT/transform helpers (planned)
├── stages/                      # one module per GPU stage (implemented in order)
├── sharding.py                  # deterministic five-way document split
└── team.py                      # fixed user-to-worker mapping
scripts/
├── submit_stage.sh              # command teammates run on CARC
├── run_gpu_stage.py             # dispatches a stage worker
└── slurm/gpu_stage.sbatch       # shared Slurm job body
notebooks/
├── 01_data_loading.ipynb
├── 02_data_exploration.ipynb
├── 03_data_preprocessing.ipynb
├── 04_baseline_analysis.ipynb               # planned, CPU only
├── 05_attack_analysis.ipynb                 # planned, CPU only
└── 06_defense_and_final_analysis.ipynb      # planned, CPU only
```

The full implementation plan is [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).
Exact models, inputs, outputs, metrics, checkpoints, and completion checks are
in the stage documents under [`docs/`](docs/PROJECT_PLAN.md).

## Team worker identities

These identities are fixed for every large GPU stage:

| User ID | Team member | Worker shard |
|---|---|---:|
| `user1` | Prabudhd | 0 of 5 |
| `user2` | Gary | 1 of 5 |
| `user3` | Saaketh | 2 of 5 |
| `user4` | Khalid | 3 of 5 |
| `user5` | Shail | 4 of 5 |

Each teammate runs the same stage command with only their user ID changed. The
launcher chooses the worker shard, GPU, CPUs, system RAM, time limit, config,
output directory, and log names. Teammates never choose document IDs manually.

## USC CARC workflow

The shared repository is:

```text
/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
```

### 1. Open the terminal and update the repository

The OnDemand shell initially opens on a login node such as `discovery1`. It is
safe to pull code, submit jobs, and inspect logs there. Never load a VLM or run
an attack directly on that prompt.

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull
myaccount
```

Confirm that `yzhao010_1531` appears in the account output.

### 2. Build the environment once

Each teammate creates the environment once from a short L40S compute
allocation so PyTorch can be verified against a real GPU:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
salloc --account=yzhao010_1531 --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
  --gpus-per-task=l40s:1
srun --pty bash -l
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
bash scripts/setup_carc.sh
exit
exit
```

For ordinary shell use:

```bash
source scripts/activate_paperwork.sh
```

Model caches go to the gitignored `.cache/` directory rather than the CARC home
directory.

### 3. Interactive GPU for stage-owner smoke tests only

The stage owner uses an interactive allocation to test one or two documents
before allowing the full team to submit jobs.

From the login terminal, check currently free GPUs with:

```bash
noderes -f -g
```

See all configured GPU resources, including busy ones, with:

```bash
noderes -c -g
```

For a node-state summary and the current GPU queue:

```bash
sinfo -p gpu -N
squeue -p gpu
```

This is a live snapshot, not a reservation. A GPU shown as free may be assigned
before your request starts, and fair-share priority can still leave a request
pending.

Recommended Qwen gradient-debug allocation:

```bash
salloc --account=yzhao010_1531 --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --gpus-per-task=a100:1 --constraint=a100-80gb
srun --pty bash -l
hostname
nvidia-smi
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
source scripts/activate_paperwork.sh
```

Recommended inference or Donut-debug allocation:

```bash
salloc --account=yzhao010_1531 --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G --time=02:00:00 \
  --gpus-per-task=l40s:1
srun --pty bash -l
```

Use `exit` to leave the compute shell, then `exit` again if needed to release
the allocation.

### 4. Full GPU work: submit scripts, do not reserve manually

For production runs, `submit_stage.sh` requests the GPU through `sbatch`. Do not
run `salloc` first.

After the assigned stage owner has implemented the stage and committed
`configs/<stage>/active.json`, all five teammates pull that exact commit and run
one command each. For example:

```bash
# Prabudhd
bash scripts/submit_stage.sh qwen_attack user1

# Gary
bash scripts/submit_stage.sh qwen_attack user2

# Saaketh
bash scripts/submit_stage.sh qwen_attack user3

# Khalid
bash scripts/submit_stage.sh qwen_attack user4

# Shail
bash scripts/submit_stage.sh qwen_attack user5
```

The browser can be closed after submission. Slurm continues the job. If a job
times out, run the exact same command again; completed records are skipped and
partial attacks resume from checkpoints.

### 5. Monitor jobs and logs

```bash
squeue -u "$USER"
jobhist
tail -f logs/<log-file>.out
seff <job-id>
```

Cancel the wrong job with:

```bash
scancel <job-id>
```

## GPU recommendations, people, and time estimates

USC Discovery currently provides L40S 48 GB, A40 48 GB, A100 40/80 GB, V100
32 GB, and P100 16 GB GPUs. The times below are **initial planning ranges**, not
guarantees. They exclude time waiting in the Slurm queue and must be replaced
with measured estimates after each stage's smoke test. A range covers the
stage's planned validation and frozen-test computation; those phases use
separate configs and may require several checkpointed submissions.

| Stage | Command | Production GPU | CPU / system RAM | Who submits? | Estimated compute per worker | Team wall time | Slurm cutoff |
|---|---|---|---|---|---:|---:|---:|
| Clean baselines | `baseline` | L40S 48 GB | 8 / 32 GB | All five | 1-4 h | 1-4 h | 4 h |
| Donut attack | `donut_attack` | L40S 48 GB | 8 / 48 GB | All five | 4-12 h | 4-12 h | 8 h |
| Qwen full-page attack | `qwen_attack` | A100 80 GB | 8 / 64 GB | All five | 12-48 h | 12-48 h | 12 h |
| Transfer and transformations | `transfer_robustness` | L40S 48 GB | 8 / 48 GB | All five | 4-16 h | 4-16 h | 8 h |
| EOT patch | `eot_patch` | A100 80 GB | 8 / 64 GB | All five | 12-48 h | 12-48 h | 12 h |
| Defenses and adaptive attacks | `defenses` | A100 80 GB | 8 / 64 GB | All five | 12-48 h | 12-48 h | 12 h |

If every worker starts at approximately the same time, a stage takes about as
long as its slowest worker; five workers do not multiply the wall time. The
current rough total for all six stages is therefore **45-176 hours of active
compute wall time (about 2-7.5 days)**, but **225-880 total GPU-hours** across
the five workers. Queue delays, failed jobs, validation changes, and the
physical pilot are additional. The lower end assumes early stopping and fast
inference; the upper end assumes most attacks run their full step count.

The cutoff is how long one Slurm job may run, not the expected completion time.
For example, a Qwen worker estimated to need 30 hours will require roughly
three checkpointed 12-hour submissions. Re-running the same command resumes
that worker's unfinished shard.

### When one person is enough and when all five are needed

| Work | People needed |
|---|---|
| Implement the stage and run a 1-10 document smoke test | One assigned stage owner |
| Run the complete validation split | All five; one fixed shard per person |
| Run the frozen held-out test split | All five again, using the frozen test config |
| Merge shards and verify completion | One stage owner after all five finish |
| Run analysis notebooks and make figures | One assigned person; CPU only |
| Capture the 50-example physical pilot | All five; ten manifest rows per person |

One person's production command completes only one fifth of the documents. It
does **not** complete a five-worker stage. The other four commands are still
required before merging. Workers may enter the queue and finish at different
times; simultaneous starts are helpful but not required.

### Same-GPU rule

For one frozen stage and run ID, all five production workers must use the same
GPU model and VRAM size. Do not mix L40S with A40/V100, or A100 40 GB with A100
80 GB, inside one run. This keeps precision support, memory behavior, numerical
behavior, and runtime comparisons consistent. Different stages may use
different GPUs exactly as listed in the table.

If the recommended GPU has an unacceptable queue:

1. The stage owner chooses one fallback for the entire team before submissions.
2. The owner smoke-tests that exact GPU and records it in the frozen config.
3. All five workers use that same fallback for the complete run.
4. A worker resuming a checkpoint continues on the same GPU type.

The A100 80 GB is the safest choice for Qwen image gradients, multi-page
resumes, EOT, and adaptive defenses. L40S is the better balance for inference
and Donut work. A40 is a reasonable 48 GB fallback if the L40S queue is long.
Avoid P100 for this project; 16 GB is too restrictive. Use V100 only for small
smoke tests after confirming the code path works without BF16.

### Replace estimates with smoke-test measurements

Every stage records completed examples and elapsed seconds during its smoke
test. Estimate the remaining time before the team submits:

```text
seconds_per_work_unit = smoke_elapsed_seconds / completed_work_units
estimated_worker_hours =
    seconds_per_work_unit * largest_worker_work_units / 3600
estimated_number_of_submissions =
    ceil(estimated_worker_hours / Slurm_cutoff_hours)
```

A work unit is one final model inference for baselines and transfer stages, and
one complete `(document, epsilon)` or `(document, patch condition)` optimization
for attack stages. Add a 20% buffer. Publish the measured estimate with the
frozen config before all five teammates submit.

CARC resource references:

- [GPU models and constraints](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming.html)
- [Discovery GPU specifications](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
- [Running and monitoring Slurm jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)

## GPU stage order

| Order | Stage command | Detailed plan | Analyzed in |
|---:|---|---|---|
| 1 | `baseline` | [Clean baselines](docs/04_clean_baselines.md) | Notebook 04 |
| 2 | `donut_attack` | [Donut attack](docs/05_donut_attack.md) | Notebook 05 |
| 3 | `qwen_attack` | [Qwen attacks](docs/06_qwen_attacks.md) | Notebook 05 |
| 4 | `transfer_robustness` | [Transfer and robustness](docs/07_transfer_and_digital_robustness.md) | Notebook 05 |
| 5 | `eot_patch` | [EOT patch and physical pilot](docs/08_eot_patch_and_physical_pilot.md) | Notebook 05 |
| 6 | `defenses` | [Defenses and adaptive attacks](docs/09_defenses_and_adaptive_attacks.md) | Notebook 06 |
| 7 | CPU only | [Final results and figures](docs/10_results_and_figures.md) | Notebook 06 |

The immediate task is to implement and smoke-test the `baseline` stage, then
create `04_baseline_analysis.ipynb` after its five worker shards finish.

## Data sources

Dataset revisions are pinned in notebook 01.

| Dataset | Hugging Face ID | Revision |
|---|---|---|
| SROIE | `jsdnrs/ICDAR2019-SROIE` | `bffe40c2` |
| CORD v2 | `naver-clova-ix/cord-v2` | `7f0115a4` |
| English resumes | `sukhrobnurali/resume-parsing-vision` | `3c254be3` |

The submitted proposal named `SyntheticResumeData`; the implemented project
uses `sukhrobnurali/resume-parsing-vision`. This change must be stated in the
midterm and final reports.

## Experiment rules

- Never modify raw datasets.
- Never run GPU work directly on a CARC login node.
- Keep original image files unchanged; model processors resize at runtime.
- Use validation data to choose prompts, attacks, and defenses.
- Freeze configurations before held-out test evaluation.
- Report attack success only among documents the same model answered correctly
  before attack.
- Store one atomic result per experiment key and checkpoint attacks every 10
  iterations.
- Never change `active.json` or stage code while team workers are running.
