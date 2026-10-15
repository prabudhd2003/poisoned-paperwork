#!/usr/bin/env bash
# Submit from the repository root. PP_AUDIT_ID must be unused.
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gpus-per-task=l40s:1
#SBATCH --job-name=pp_donut_restart_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "Job: $SLURM_JOB_ID; host: $(hostname); commit: $(git rev-parse HEAD)"
nvidia-smi
python -u scripts/verify_donut_restart.py --audit-id "${PP_AUDIT_ID:-restart_audit_paper_v2}"
