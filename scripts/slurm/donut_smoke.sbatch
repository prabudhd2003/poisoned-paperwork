#!/usr/bin/env bash
# Submit from the repository root with: sbatch scripts/slurm/donut_smoke.sbatch
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-task=l40s:1
#SBATCH --job-name=pp_donut_smoke_khalid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
if [[ "$(id -un)" != kbali ]]; then
    echo "This smoke launcher is configured for Khalid's environment." >&2
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit reviewed files before starting the smoke run." >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1 PYTHONNOUSERSITE=1
export HF_HOME="$PWD/.cache/hf" TORCH_HOME="$PWD/.cache/torch"
export PYTHONPATH="$PWD/src" HF_HUB_DISABLE_TELEMETRY=1
PYTHON_BIN=/home1/kbali/.conda/envs/paperwork/bin/python
echo "Job: ${SLURM_JOB_ID}; commit: $(git rev-parse HEAD)"
"$PYTHON_BIN" -c 'import sys, torch; print(sys.executable, torch.__version__); assert torch.cuda.is_available()'
nvidia-smi
"$PYTHON_BIN" -u scripts/smoke_donut_attack.py
