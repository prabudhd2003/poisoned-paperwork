# Reproducing the experiments

There are two parts, run in order:

1. **Cluster stages** (`src/`, `scripts/`, `configs/`) prepare the data, measure clean
   accuracy and generate the saved attacks. They run as checkpointed Slurm jobs on
   L40S GPUs (USC CARC); any Slurm cluster with similar GPUs works.
2. **Colab notebooks** (`notebooks/04`–`09`) evaluate CBVR and the other defenses on those
   saved attacks, attack CBVR directly, and run the resume and Donut studies. Each runs on
   one A100 and writes resumable results to Google Drive.

Every run is frozen by hash: a runner records its code, inputs and settings in
`contract.json` and refuses to resume into a folder whose contract differs. To change
anything, start a new study ID.

## 1. Setup

```bash
git clone -b final https://github.com/prabudhd2003/poisoned-paperwork
cd poisoned-paperwork
bash scripts/setup_carc.sh               # conda env "paperwork"; exact versions in env/requirements.lock.txt
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python -m pytest tests -q                # CPU checks
```

Models: `Qwen/Qwen2.5-VL-3B-Instruct` and `naver-clova-ix/donut-base-finetuned-docvqa`,
with revisions pinned in `configs/`.

## 2. Data

| Notebook / script | What it does |
|---|---|
| `notebooks/01_data_loading.ipynb` | downloads SROIE, CORD v2 and the resume dataset into `data/` |
| `notebooks/02_data_preprocessing.ipynb` | writes `data/processed/` and `master_manifest.parquet` |
| `notebooks/03_collected_receipt_preprocessing.ipynb` | prepares our 114 receipt photos (`data/collected_receipts/`, labels in `data/annotations/`) |
| `python scripts/freeze_collected_external.py --expected 114` | freezes the collected set by hash |

## 3. Cluster: baselines and saved attacks

Each stage is: write a runtime config, submit it, check it.
`submit_stage.sh STAGE all CONFIG` submits all five workers as one array;
`stage_status.py` reports progress and merges finished shards. Rerunning the same
command resumes a stage.

```bash
# clean accuracy
python scripts/prepare_stage_config.py baseline_collected
bash scripts/submit_stage.sh baseline all outputs/runtime_configs/baseline_collected.json

# standard targeted attacks (eps 2/4/8/16)
python scripts/prepare_stage_config.py qwen_collected
bash scripts/submit_stage.sh qwen_attack all outputs/runtime_configs/qwen_collected.json
python scripts/stage_status.py qwen_attack qwen_collected_external_v1

# EOT attacks that optimize through JPEG / resize / voting (uniform and hard-view)
Q=outputs/qwen_attack/qwen_collected_external_v1/merged/records.parquet
python scripts/prepare_stage_config.py jpeg_eot_qwen_collected --source qwen_collected="$Q" \
  --output outputs/runtime_configs/qwen_collected_jpeg_eot_full_v1.json
python scripts/prepare_stage_config.py jpeg_eot_qwen_collected_hardview --source qwen_collected="$Q" \
  --output outputs/runtime_configs/qwen_collected_hardview_full_v1.json
for C in qwen_collected_jpeg_eot_full_v1 qwen_collected_hardview_full_v1; do
  bash scripts/submit_stage.sh jpeg_eot_attack all outputs/runtime_configs/$C.json
done
```

The public-dataset runs (CORD, SROIE, resumes) and the image-cleaning defenses follow the
same pattern with the configs in `configs/`. `python scripts/prepare_stage_config.py -h`
lists every stage name.

## 4. Colab: CBVR on receipts (main result)

Build the input bundle from the saved attacks:

```bash
CARC_HOST=user@discovery.usc.edu bash scripts/build_receipt_study_inputs.sh
```

It copies the images listed in `data/receipt_study/remote_files.txt`, rebuilds
`source.tar` (the code snapshot the runner checks: commit `64aa87b`), verifies every
hash and writes `data/receipt_study/csev_full_inputs.zip`. Upload it to Drive, set
`DRIVE_FOLDER` at the top of each notebook, and run:

| Order | File | What it does |
|---|---|---|
| 1 | `notebooks/04_receipt_cbvr_study.ipynb` | development (210 cases) → **stop and review** → freeze → reserved evaluation (441 cases) |
| 2 | `scripts/colab_receipt_direct_attack_part1_cell.py` | paste as the last cell of notebook 04: attacks aimed at CBVR, receipts 1–5 |
| 3 | `scripts/colab_receipt_direct_attack_part2_cell.py` | one standalone cell: receipts 6–10 |
| 4 | `notebooks/05_receipt_direct_attack_part3.ipynb` | ten more receipts in the same frozen order |

The runner is `scripts/colab_receipt_study.py`; its `POLICY` dictionary fixes the methods,
band geometry and attack budgets. Each session stops after 4 hours; run the cell again to
continue. The 80 direct-attack runs took 11.7 A100 hours in total.

## 5. Colab: resumes

On the cluster, after the standard resume attack has merged:

```bash
python scripts/export_resume_development.py --root . --out resume_development_inputs_v1.zip
python scripts/export_resume_complete.py   --root . --out resume_complete_inputs_v1.zip
```

Upload both zips (and `source.tar` one folder up) to Drive, then run
`notebooks/06_resume_development.ipynb` → `07_resume_saved_attacks.ipynb` (and paste
`scripts/colab_resume_reserved_evaluation_cell.py` below it for the 60 reserved resumes) →
`08_resume_direct_attack.ipynb`. Use an A100; an L4 runs out of memory on two-page resumes.

## 6. Colab: Donut as the reader

`notebooks/09_donut_cbvr_study.ipynb` uses the same receipt bundle. Qwen locates the band
and Donut reads it.

## Outputs

Each notebook ends by exporting a zip of its summaries. The committed copies are in
`results/`; [`RESULTS.md`](RESULTS.md) lists every number with its denominator.
