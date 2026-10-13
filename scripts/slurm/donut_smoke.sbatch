#!/usr/bin/env bash
# From the repository root:
#   sbatch scripts/slurm/donut_smoke.sbatch configs/donut_attack/smoke_validation.json
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
CONFIG_PATH="${1:-configs/donut_attack/smoke_validation.json}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit reviewed files before starting the smoke run." >&2
    exit 1
fi
source scripts/activate_paperwork.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Job: ${SLURM_JOB_ID}; commit: $(git rev-parse HEAD); config: $CONFIG_PATH"
python -c 'import sys, torch; print(sys.executable, torch.__version__); assert torch.cuda.is_available()'
nvidia-smi
python -u scripts/smoke_donut_attack.py --config "$CONFIG_PATH"
