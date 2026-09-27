# Poisoned Paperwork

Targeted adversarial attacks and defenses for document vision-language models.
We test whether small pixel perturbations can make a model return a chosen
wrong answer while the document still looks unchanged to a person.

| Role | Choice |
|---|---|
| Primary model | Qwen2.5-VL-3B-Instruct |
| Attack reproduction / debugging | Donut |
| Transfer-only model | InternVL2.5-4B |
| Primary task | Receipt total extraction |
| Secondary task | Resume highest-degree classification |

## Status

The data pipeline is complete and ready for modeling.

| Stage | File | Status |
|---|---|---|
| Download pinned datasets | `notebooks/01_data_loading.ipynb` | Done |
| Explore and audit all documents | `notebooks/02_data_exploration.ipynb` | Done |
| Review missing or incorrect receipt totals | `annotations/receipt_total_labels.csv` | Done |
| Normalize labels, remove unusable pages, and create splits | `notebooks/03_data_preprocessing.ipynb` | Done |
| Establish clean model baselines | `notebooks/04_clean_baselines.ipynb` | **Next** |
| Reproduce the targeted attack on Donut | `notebooks/05_donut_attack.ipynb` | Planned |
| Run full-page targeted attacks on Qwen | `notebooks/06_qwen_attacks.ipynb` | Planned |
| Test transfer and digital robustness | `notebooks/07_transfer_and_digital_robustness.ipynb` | Planned |
| Run EOT patches and a physical pilot | `notebooks/08_eot_patch_and_physical_pilot.ipynb` | Planned |
| Test defenses and adaptive attacks | `notebooks/09_defenses_and_adaptive_attacks.ipynb` | Planned |
| Produce final tables and figures | `notebooks/10_results_and_figures.ipynb` | Planned |

### Completed data checks

- All 2,987 raw documents and 3,785 image pages were audited.
- Every image decodes and has valid dimensions.
- The manual annotation file contains 29 verified receipt totals or corrections.
- Two CORD receipts were excluded because their answer requires arithmetic
  rather than direct extraction.
- Seven completely blank resume pages were removed.
- All 83 raw resume degree strings were mapped to seven canonical levels.
- Raw datasets remain unchanged; processed copies are written to
  `data/processed/`.

### Modeling-ready data

| Dataset | Train | Validation | Test | Usable documents |
|---|---:|---:|---:|---:|
| SROIE | 500 | 126 | 361 | 987 |
| CORD v2 | 798 | 100 | 100 | 998 |
| English resumes | 850 | 75 | 75 | 1,000 |
| **Total** | **2,148** | **301** | **536** | **2,985** |

The resume dataset has 992 documents with a valid next-degree adversarial
target. The master manifest contains all 2,987 raw documents, including the two
excluded CORD receipts, so every decision remains traceable.

Generated files are stored under `data/processed/` and are intentionally not
committed to GitHub. Running notebooks 01 and 03 recreates them on any machine.

## Next steps

The complete implementation contract is in [`docs/`](docs/README.md). It gives
each remaining notebook exact inputs, models and revisions, outputs, metrics,
five-person CARC execution, checkpoint behavior, and completion checks.

| Order | Notebook | Outcome | GPU? |
|---:|---|---|---:|
| 1 | [04 clean baselines](docs/04_clean_baselines.md) | Frozen prompts, clean accuracy, and attack-eligible IDs | CARC GPU |
| 2 | [05 Donut attack](docs/05_donut_attack.md) | Validated targeted-PGD implementation | CARC GPU |
| 3 | [06 Qwen attacks](docs/06_qwen_attacks.md) | Main epsilon sweep and resume case study | CARC GPU |
| 4 | [07 transfer and robustness](docs/07_transfer_and_digital_robustness.md) | InternVL transfer and digital-survival features | CARC GPU |
| 5 | [08 EOT patch and physical pilot](docs/08_eot_patch_and_physical_pilot.md) | Robust patches, physical labels, and predictors | CARC GPU + manual capture |
| 6 | [09 defenses and adaptive attacks](docs/09_defenses_and_adaptive_attacks.md) | Three defenses tested against informed attackers | CARC GPU; detector fitting is CPU |
| 7 | [10 results and figures](docs/10_results_and_figures.md) | Final verified tables, figures, and report summary | CPU only |

The immediate task is notebook 04. Use validation data to select prompts and
settings; keep test splits untouched until configurations are frozen.

All GPU notebooks follow the [shared execution contract](docs/00_execution_contract.md):

- CARC only; never load a model on the login node.
- Five deterministic document shards, one per teammate and GPU allocation.
- One shared implementation with worker IDs 0-4, not five code copies.
- Atomic per-document results and attack checkpoints every 10 steps.
- Completed work is skipped automatically after a CARC timeout.
- Results merge only after all expected worker shards pass validation.

## Run on USC CARC

The shared working copy is:

```text
/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
```

Pull the latest code:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull
```

Create the environment once:

```bash
bash scripts/setup_carc.sh
```

Select **paperwork (Python 3.11)** as the notebook kernel. For terminal work:

```bash
source scripts/activate_paperwork.sh
```

On a fresh CARC copy, run `01_data_loading.ipynb` and then
`03_data_preprocessing.ipynb`. Notebook 02 is for inspection and auditing and
does not need to run every time.

Request a GPU before loading a model:

```bash
salloc --account=yzhao010_1531 --partition=gpu --gres=gpu:1 \
       --cpus-per-task=8 --mem=32G --time=2:00:00
```

Model caches are redirected to the gitignored `.cache/` directory rather than
the CARC home directory. Exact package versions are in
`env/requirements.lock.txt`.

## Data sources

Dataset revisions are pinned in `01_data_loading.ipynb`.

| Dataset | Hugging Face ID | Revision |
|---|---|---|
| SROIE | `jsdnrs/ICDAR2019-SROIE` | `bffe40c2` |
| CORD v2 | `naver-clova-ix/cord-v2` | `7f0115a4` |
| English resumes | `sukhrobnurali/resume-parsing-vision` | `3c254be3` |

The proposal named Jijun Hao's `SyntheticResumeData`; the implemented project
uses `sukhrobnurali/resume-parsing-vision`. This change should be stated in the
midterm and final reports.

## Repository layout

```text
poisoned-paperwork/
├── annotations/   # reviewed labels committed to Git
├── data/          # raw and processed data; gitignored
├── docs/          # detailed implementation plans for notebooks 04-10
├── env/           # dependencies and reproducible lock file
├── notebooks/     # numbered project workflow
├── scripts/       # CARC environment setup and activation
└── README.md
```

## Experiment rules

- Never modify the downloaded raw datasets.
- Never stretch images to a fixed shape. Preserve aspect ratio and use the
  model processor for runtime resizing and padding.
- Apply the same deterministic input pipeline to clean and attacked images.
- Tune prompts and attacks on validation data, then evaluate the frozen setup
  on test data.
- Report attack success only among examples the same model answered correctly
  before the attack.
