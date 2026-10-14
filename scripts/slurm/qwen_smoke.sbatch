#!/usr/bin/env bash
# Owner-only Qwen smoke run (either track). From the repository root:
#   sbatch scripts/slurm/qwen_smoke.sbatch configs/qwen_attack/<smoke>.json
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --job-name=pp_qwen_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
CONFIG_PATH="${1:?Usage: sbatch scripts/slurm/qwen_smoke.sbatch configs/qwen_attack/<smoke>.json}"
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit reviewed files before starting the smoke run." >&2
    exit 1
fi
python3 - "$CONFIG_PATH" <<'PY'
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
if config.get("track") == "receipt" and config.get("ready_for_gpu_smoke") is not True:
    raise SystemExit(
        "Receipt GPU smoke is gated until source-pixel evidence is reviewed and pinned."
    )
PY
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Job: ${SLURM_JOB_ID}; commit: $(git rev-parse HEAD); config: $CONFIG_PATH"
nvidia-smi
python -u scripts/check_qwen_processor_parity.py --config "$CONFIG_PATH"
python -u scripts/smoke_qwen_attack.py --config "$CONFIG_PATH"
