# Reproducing the experiments

The work runs in two places, in this order:

1. **CARC pipeline** (`src/`, `scripts/`, `configs/`): prepares the data, measures clean
   accuracy, and generates the saved attacks (standard PGD, uniform EOT, hard-view EOT).
   It runs as checkpointed Slurm jobs on L40S GPUs. Exact commands are in
   [`CARC_END_TO_END.md`](CARC_END_TO_END.md); each stage is described in `00_…` to `09_…`.
2. **Colab studies** (`colab/`): CBVR and the other defenses on those saved attacks, the
   attacks aimed at CBVR, resumes and Donut. Each notebook runs on one A100 (an L4 runs
   out of memory on two-page resumes) and writes resumable results to Google Drive.

Everything is frozen by hash: each Colab runner records a `contract.json` (code, inputs,
policy) and refuses to resume into a folder whose contract differs. To change anything,
start a new study ID.

## 0. Environment

```bash
git clone https://github.com/prabudhd2003/poisoned-paperwork && cd poisoned-paperwork
bash scripts/setup_carc.sh        # creates the "paperwork" env; pins in env/requirements.lock.txt
python -m pytest tests -q          # CPU tests, no GPU or model weights needed
```

Models: `Qwen/Qwen2.5-VL-3B-Instruct` and `naver-clova-ix/donut-base-finetuned-docvqa`
(revisions are pinned in the configs). The Colab notebooks build their own Python 3.11
environment with the same package versions.

## 1. Data

| Dataset | Source | Used for |
|---|---|---|
| CORD, SROIE | public receipt sets, downloaded by `notebooks/01_data_loading.ipynb` | development, attacks |
| Resumes | `resume_parsing_vision` (75 validation resumes) | degree task |
| Self-collected receipts | 114 phone photos in `collected receipts/`, labels in `annotations/` | held-out evaluation |

`notebooks/01`–`05` load, inspect and preprocess the data into `data/processed/`
(`master_manifest.parquet`); `scripts/freeze_collected_external.py` freezes the
collected set. The collected receipts were labeled by hand (`annotations/collected_receipt_labels.csv`).

## 2. CARC: baselines and saved attacks

Follow `CARC_END_TO_END.md` sections 0–6. The stages used by the final results are:

| Stage | Config folder | Output |
|---|---|---|
| Clean baselines | `configs/baseline/` | clean accuracy (RESULTS §1) |
| Standard Qwen / Donut attacks | `configs/qwen_attack/`, `configs/donut_attack/` | saved attack PNGs (RESULTS §2) |
| Image-cleaning defenses | `configs/defenses/` | RESULTS §3 |
| Uniform and hard-view EOT | `configs/jpeg_eot_attack/` | saved EOT PNGs |

The Colab receipt studies read the saved attack PNGs listed in
`colab/receipts/inputs/remote_files.txt`; their hashes are in `file_hashes.json`.

## 3. Colab: CBVR on receipts (main result)

```bash
CARC_HOST=user@discovery.usc.edu bash colab/receipts/fetch_full_inputs.sh
```

This copies the 396 + 135 saved attacks and the clean images from CARC, rebuilds
`source.tar` (the code snapshot the runner checks: commit `64aa87b`), verifies every
hash and writes `colab/receipts/csev_full_inputs.zip`. Upload it to Drive, then:

| Step | File | What it does |
|---|---|---|
| 1 | `colab/receipts/1_cbvr_study.ipynb` | preflight → development (210 cases) → **stop and review** → freeze → reserved evaluation (441 cases) → summaries |
| 2a | `2a_direct_attack_part1_cell.py` | paste as a last cell of notebook 1; attacks CBVR on the first 5 frozen receipts (uses `continue_recovery.py`) |
| 2b | `2b_direct_attack_part2_cell.py` | one standalone cell; receipts 6–10 |
| 2c | `2c_direct_attack_part3.ipynb` | ten more receipts, chosen in the same frozen order |

The 80 attack runs took 11.7 A100 hours in total; each Colab session stops after 4 hours and resumes on the next run. Set `DRIVE_FOLDER` in each notebook to your Drive folder. Every stage checkpoints after
each case; if Colab disconnects, run the same cells again and it resumes.
Output: `full_study_results/receipt_bands_full_v1/` (summaries in `results/receipts_qwen/`).

The runner is `colab/receipts/full_study.py` (also embedded in the notebook). Methods,
band geometry and attack budgets are fixed in its `POLICY` dictionary.

## 4. Colab: resumes

On CARC, after the standard resume attack has merged:

```bash
python colab/resumes/export_resume_development.py --root . --out resume_development_inputs_v1.zip
python colab/resumes/export_resume_complete.py   --root . --out resume_complete_inputs_v1.zip
```

Upload both zips (and `source.tar` in the parent folder) to Drive, then run in order:

| Step | File | What it does |
|---|---|---|
| 1 | `colab/resumes/1_development.ipynb` | design the education-band rule on 15 clean development resumes (frozen v4) |
| 2 | `colab/resumes/2_saved_attacks.ipynb` | 15 resumes + 60 saved attacks, development only |
| 3 | `colab/resumes/3_reserved_evaluation_cell.py` | paste below notebook 2; 60 reserved resumes + 236 attacks |
| 4 | `colab/resumes/4_direct_attack.ipynb` | 40 attacks aimed at recovery (10 resumes × ε 8, 16 × 2 tries) |

## 5. Colab: Donut as the reader

`colab/donut/1_donut_cbvr_study.ipynb` uses the same `csev_full_inputs.zip`; Qwen locates
the band and Donut reads it. `2_geometry_diagnostic.ipynb` is a development-only check
of other band sizes (none was adopted).

## 6. Figures and poster

`poster/main.tex` compiles with LuaLaTeX (or pdfLaTeX); `poster/make_results12.py` redraws
the defense chart from the numbers in `results/receipts_qwen/evaluation_summary.csv`.
`midterm/` and `notebooks/` hold the CPU-only analysis notebooks used for earlier reports.
