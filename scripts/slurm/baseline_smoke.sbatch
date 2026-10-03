#!/usr/bin/env bash

set -euo pipefail

: "${PP_REPO_ROOT:?PP_REPO_ROOT is required}"
: "${PP_SMOKE_CONFIG:?PP_SMOKE_CONFIG is required}"

cd "$PP_REPO_ROOT"
source scripts/activate_paperwork.sh
export PYTHONPATH="$PP_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Config: $PP_SMOKE_CONFIG"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

python -u scripts/smoke_baseline.py --config "$PP_SMOKE_CONFIG"

echo "Finished: $(date --iso-8601=seconds)"
echo "SMOKE TEST COMPLETED SUCCESSFULLY"
