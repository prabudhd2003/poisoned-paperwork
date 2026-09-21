# Poisoned Paperwork

Adversarial attacks and defenses for document vision-language models.

## Repository structure

```text
poisoned-paperwork/
├── data/                         # downloaded locally; not tracked by Git
├── notebooks/
│   ├── 01_data_loading.ipynb     # download the three datasets
│   └── 02_data_exploration.ipynb # inspect samples and simple statistics
├── .gitignore
└── README.md
```

## Getting started

Clone the repository, open `notebooks/01_data_loading.ipynb`, and run all
cells. The notebook finds the repository root automatically and downloads
everything into these folders:

```text
data/sroie/
data/cord_v2/
data/synthetic_resume/
```

Then run `notebooks/02_data_exploration.ipynb` to view example documents and
basic dataset plots. The contents of `data/` are excluded from Git, so running
the notebooks will not add the datasets to a commit.

### CARC note

CARC compute nodes do not have public internet access. Run the download
notebook on a CARC transfer node (`hpc-transfer1` or `hpc-transfer2`), then run
the exploration and model notebooks through the normal CARC compute/Jupyter
environment. All nodes will see the same repository and `data/` directory when
the repository is stored under the shared project filesystem.
