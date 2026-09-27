#!/usr/bin/env bash

# Resource requests are supplied by scripts/submit_stage.sh so that each stage
# receives the appropriate GPU, memory, and time limit.

set -euo pipefail

: "${PP_REPO_ROOT:?PP_REPO_ROOT is required}"
: "${PP_STAGE:?PP_STAGE is required}"
: "${PP_CONFIG:?PP_CONFIG is required}"
: "${PP_USER_ID:?PP_USER_ID is required}"
: "${PP_WORKER_ID:?PP_WORKER_ID is required}"
: "${PP_NUM_WORKERS:?PP_NUM_WORKERS is required}"

cd "$PP_REPO_ROOT"

source scripts/activate_paperwork.sh
export PYTHONPATH="$PP_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Stage: $PP_STAGE"
echo "User: $PP_USER_ID"
echo "Worker: $PP_WORKER_ID/$PP_NUM_WORKERS"
echo "Config: $PP_CONFIG"
echo "Host: $(hostname)"
nvidia-smi

python -u scripts/run_gpu_stage.py \
  --stage "$PP_STAGE" \
  --config "$PP_CONFIG" \
  --user-id "$PP_USER_ID" \
  --worker-id "$PP_WORKER_ID" \
  --num-workers "$PP_NUM_WORKERS"
