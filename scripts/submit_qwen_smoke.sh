#!/usr/bin/env bash
# Validate the complete CPU-side Qwen smoke contract before entering the GPU queue.
# Usage:
#   bash scripts/submit_qwen_smoke.sh configs/qwen_attack/<smoke>.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:?Usage: bash scripts/submit_qwen_smoke.sh configs/qwen_attack/<smoke>.json}"
cd "$REPO_ROOT"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Commit reviewed files before submitting the Qwen smoke." >&2
  exit 1
fi

source scripts/activate_paperwork.sh
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import sys
import numpy
import pandas
import torch
import transformers

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Qwen smoke requires Python 3.11; found {sys.version}")
if not sys.executable.endswith("/.conda/envs/paperwork/bin/python"):
    raise SystemExit(f"Qwen smoke is using the wrong interpreter: {sys.executable}")
print(
    "Qwen environment passed:",
    f"python={sys.executable}",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"pandas={pandas.__version__}",
    f"numpy={numpy.__version__}",
)
PY

python scripts/check_qwen_processor_parity.py --config "$CONFIG_PATH"
python scripts/smoke_qwen_attack.py --config "$CONFIG_PATH" --preflight

# Do not export submit-shell Conda bookkeeping into Slurm. The batch script
# activates and verifies paperwork afresh on the allocated host.
for conda_prefix_variable in $(compgen -A variable CONDA_PREFIX); do
  unset "$conda_prefix_variable"
done
unset CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL

mkdir -p logs
JOB_ID="$(sbatch --parsable scripts/slurm/qwen_smoke.sbatch "$CONFIG_PATH")"
echo "Submitted validated Qwen L40S smoke job ${JOB_ID%%;*}"
echo "Monitor: squeue -j ${JOB_ID%%;*}"
echo "Output:  logs/pp_qwen_smoke_${JOB_ID%%;*}.out"
echo "Errors:  logs/pp_qwen_smoke_${JOB_ID%%;*}.err"
