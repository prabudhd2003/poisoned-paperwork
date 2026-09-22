# Adversarial Attacks on Document Vision-Language Models

Adversarial attacks and defenses for document vision-language models.

CSCI 566, Fall 2026 (Yue Zhao) · Prabudhd Krishna Kandpal, Gary Fenton,
Saaketh Kanduri, Khalid Ali, Shail Shah.

We test whether subtle, optimized pixel changes to an uploaded document image
can make an open-weight VLM extract a chosen wrong answer, and we measure how
far that attack actually goes: how it scales with the perturbation budget,
whether it transfers to a different model, whether it survives ordinary
document processing, and whether three lightweight defenses stop it. The point
is to find where the attack **stops** working rather than to assume it survives
a real pipeline.

- **Primary attack model:** Qwen2.5-VL-3B-Instruct
- **Reproduction / debug model:** Donut
- **Transfer-only target:** InternVL2.5-4B
- **Main task:** receipt total extraction · **Secondary:** résumé highest-degree extraction

## Where we work

All work happens on **USC CARC**, in the shared project space:

```text
/project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
```

Nobody needs a local clone. GitHub (`prabudhd2003/poisoned-paperwork`) is the
sync point and history; CARC is the working copy.

Pull before you start, every time:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull
```

Running a notebook rewrites its outputs and execution counts, which makes the
file dirty and blocks the next pull. Discard that churn (`git checkout -- <notebook>`)
or commit it deliberately.

Set your git identity once, so commits are attributed to you rather than to a
hostname. The contribution statement is checked against commit history.

```bash
git config --global user.name  "Your Name"
git config --global user.email "your@email"
git config --global pull.rebase true
```

## Environment

The conda env is called **paperwork** (Python 3.11). Build it once per person:

```bash
bash scripts/setup_carc.sh
```

Then reload the code-server window and pick **paperwork (Python 3.11)** in the
notebook kernel selector. For a shell instead of a notebook:

```bash
source scripts/activate_paperwork.sh
```

Notes worth knowing before you debug something for an hour:

- `PYTHONNOUSERSITE=1` is essential. Without it, `~/.local/lib/python3.11/site-packages`
  shadows the env, pip reports everything as "already satisfied", installs
  nothing — and the kernel never appears in the picker.
- Model weights go to `.cache/hf` inside this repo (gitignored), **not** `$HOME`.
  Home quota is 100 GB and the three models total roughly 16 GB.
- `conda activate` needs `source $(conda info --base)/etc/profile.d/conda.sh`
  first in a fresh shell.
- `cuda` loads only after `gcc/13.3.0`; it sits under that branch of the module
  hierarchy. It is optional anyway, since the cu124 torch wheels bundle their
  own runtime.
- `env/requirements.txt` holds floors; `env/requirements.lock.txt` holds the
  exact resolved versions and is what reproduces results.

Run anything that loads a model on a GPU node, not the login node:

```bash
salloc --account=yzhao010_1531 --partition=gpu --gres=gpu:1 \
       --cpus-per-task=8 --mem=32G --time=2:00:00
```

## Data

Three datasets, pinned by revision in `notebooks/01_data_loading.ipynb` so
everyone gets byte-identical copies:

| Role | Hugging Face id | Revision |
|---|---|---|
| English receipts | `jsdnrs/ICDAR2019-SROIE` | `bffe40c2` |
| Indonesian receipts | `naver-clova-ix/cord-v2` | `7f0115a4` |
| English synthetic résumés | `sukhrobnurali/resume-parsing-vision` | `3c254be3` |

**2,987 documents / 3,785 image pages**, about 6.1 GB. `data/` is gitignored;
run notebook 01 to populate it.

| Dataset | Train | Val | Test | Documents | Image pages |
|---|---:|---:|---:|---:|---:|
| SROIE | 626 | 0 | 361 | 987 | 987 |
| CORD v2 | 800 | 100 | 100 | 1,000 | 1,000 |
| Résumés | 850 | 75 | 75 | 1,000 | 1,798 |
| **Total** | **2,276** | **175** | **536** | **2,987** | **3,785** |

> The project proposal cited Jijun Hao's *SyntheticResumeData*. We replaced it
> with `sukhrobnurali/resume-parsing-vision`. Correct this in the midterm report.

## Repository structure

```text
poisoned-paperwork/
├── annotations/
│   └── receipt_total_labels.csv   # manually reviewed totals for missing labels
├── data/                          # gitignored; created by notebook 01
├── env/
│   ├── requirements.txt           # dependency floors
│   └── requirements.lock.txt      # exact resolved versions
├── notebooks/
│   ├── 01_data_loading.ipynb      # pinned downloads + count assertions
│   └── 02_data_exploration.ipynb  # samples, statistics, readiness audit, label review
├── scripts/
│   ├── setup_carc.sh              # build the paperwork env + Jupyter kernel
│   └── activate_paperwork.sh      # source this for a shell
├── .cache/                        # gitignored; HF and torch model caches
└── README.md
```

## Progress

### Done

**Data acquisition (`01_data_loading.ipynb`).** All three datasets download to
pinned revisions and save to disk. A count assertion (987 / 1,000 / 1,000)
fails loudly on an incomplete download.

**Readiness audit (`02_data_exploration.ipynb`).** Every document was scanned
without modifying the saved datasets — images decode, dimensions are valid,
pages are not blank, task labels are present, annotation structure is as
expected, identifiers are unique. Results:

- All **3,785 image pages** decoded. No corrupt images, invalid dimensions,
  duplicate identifiers, or annotation-structure mismatches.
- **SROIE:** 1 of 987 receipts was missing its `total` label (train index 33).
- **CORD v2:** 28 of 1,000 receipts lacked `total.total_price` — 21 train,
  2 validation, 5 test.
- **Résumés:** all 1,000 have at least one degree label. Seven contain a fully
  blank extra page, but each still has a usable page; the blank pages should be
  dropped during processing.
- Degree labels are messy: **2,027 annotations across 83 distinct raw strings**,
  48 of which occur five times or fewer. `Bachelor of Science` alone is 679
  (33.5%). These need a reviewed mapping to ordered canonical levels.

**Manual label recovery (`annotations/receipt_total_labels.csv`).** Every
receipt with a missing total was displayed and reviewed by hand. The raw
datasets were left untouched; corrections live in this CSV and are applied
during processing.

| | Reviewed | Verified | Excluded | Usable receipts |
|---|---:|---:|---:|---:|
| SROIE | 1 | 1 | 0 | 987 of 987 |
| CORD v2 | 28 | 26 | 2 | 998 of 1,000 |

`status` is `verified` only when the value on the receipt is unambiguous;
`ambiguous` or `exclude` otherwise. Totals are recorded exactly as printed,
including decimal marks and thousands separators — normalization happens later.

That leaves **1,985 labeled receipts** for the total-extraction task and
**1,000 résumés** for the degree task.

**Environment.** The `paperwork` conda env is built and registered as a Jupyter
kernel on CARC, with model caches redirected into the repo.

### Next

1. **`03_data_processing.ipynb`** — apply the manual labels, normalize totals to
   a canonical numeric form, drop the excluded receipts and the seven blank
   résumé pages, map the 83 raw degree strings to ordered canonical levels, and
   build the disjoint attack / detector-training / held-out splits. All variants
   of one document must stay in the same split.
2. **Clean accuracy baselines** — Qwen2.5-VL-3B on receipt totals and résumé
   degrees, with a fixed prompt and exact-match scoring. Attacks are only ever
   evaluated on documents the model already answers correctly, so this defines
   the evaluation set.
3. **Evaluation harness** — save/reload-through-PNG scoring, exact-match on the
   target, and the perceptibility metrics (L∞, LPIPS, SSIM).
4. **Donut reproduction** — reproduce the Pintore et al. attack to validate the
   implementation before pointing it at Qwen.
5. **Qwen PGD** at ε ∈ {2, 4, 8, 16}, then transfer to InternVL2.5-4B, EOT
   robustness, patches, and the three defenses.

## Key dates

| | |
|---|---|
| Midterm report | Mon Nov 9 |
| Poster + demo session | Mon Nov 30 (in person, individual Q&A) |
| Final report | Mon Dec 7, 11:59pm Pacific — no late days |
