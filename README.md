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
| Run targeted attacks on Qwen | `notebooks/06_qwen_attacks.ipynb` | Planned |
| Test transfer, robustness, and defenses | `notebooks/07_transfer_and_defenses.ipynb` | Planned |

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

### 1. Clean baselines — immediate next task

Create `04_clean_baselines.ipynb` and:

1. Load the processed SROIE, CORD, and resume splits.
2. Build separate Qwen and Donut input pipelines using each model's official
   processor. Keep the saved images at their original resolution; resize, pad,
   and normalize only when a batch is sent to a model.
3. Freeze one prompt per task and one deterministic image-processing setup.
4. Measure exact-match accuracy on the validation split first.
5. Save one prediction row per document: `record_id`, target, prediction,
   correctness, model, prompt version, and processing settings.
6. Mark the clean-correct documents. Attack success rate will be measured only
   on this subset.

Use validation data to choose prompts and settings. Keep the test splits
untouched until the experiment design is fixed.

### 2. Validate the attack implementation

In `05_donut_attack.ipynb`, reproduce the targeted Donut attack on a small
clean-correct validation subset. Verify that saved-and-reloaded images still
work and record attack success, L-infinity distance, LPIPS, and SSIM.

### 3. Run the main Qwen experiments

In `06_qwen_attacks.ipynb`, run targeted PGD at epsilon values
`{2, 4, 8, 16}` on clean-correct validation examples. Finalize the attack
settings before running once on the held-out test set.

### 4. Test real-world limits

In `07_transfer_and_defenses.ipynb`, evaluate transfer to InternVL2.5-4B,
save/reload robustness, EOT robustness, patch attacks, and the planned
lightweight defenses. Compare both attack success and image perceptibility.

### 5. Report results

Produce per-dataset and overall tables for clean accuracy, conditional attack
success rate, transfer rate, robustness, defense effectiveness, and image
quality. Record the exact model revisions, prompts, seeds, and preprocessing
settings needed to reproduce every table.

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
