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
source "$PP_REPO_ROOT/scripts/lib/qwen_receipt_runtime.sh"

IS_RECEIPT=0
if [[ "$PP_STAGE" == "qwen_attack" ]] && \
   pp_receipt_is_config "$PP_CONFIG"; then
  IS_RECEIPT=1
else
  CONFIG_CLASSIFICATION_STATUS=$?
  if [[ "$CONFIG_CLASSIFICATION_STATUS" -ne 1 ]]; then
    pp_receipt_die "could not safely classify config: $PP_CONFIG"
  fi
fi

if [[ "$IS_RECEIPT" -eq 1 ]]; then
  : "${PP_EXPECTED_GIT_SHA:?PP_EXPECTED_GIT_SHA is required for receipt runs}"
  : "${PP_TRACK:?PP_TRACK is required for receipt runs}"
  pp_receipt_validate_context \
    "$PP_REPO_ROOT" "$PP_CONFIG" "$PP_EXPECTED_GIT_SHA" "GPU worker execution"
  [[ "$PP_TRACK" == "receipt" ]] || pp_receipt_die \
    "exported track does not match the receipt config"
  [[ "$PP_CONFIG_MODE" != "smoke" ]] || pp_receipt_die \
    "receipt smoke must use scripts/submit_qwen_smoke.sh"
  [[ "$PP_CONFIG_READY_PRODUCTION" == "true" ]] || pp_receipt_die \
    "receipt production remains gated"
  [[ "${PP_RUN_ID:-}" == "$PP_CONFIG_RUN_ID" ]] || pp_receipt_die \
    "exported run ID does not match the committed receipt config"

  if ! command -v module >/dev/null 2>&1; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
  fi
  if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.3.0 cuda/12.4.0
  fi
  export PYTHONNOUSERSITE=1
  export HF_HOME="${HF_HOME:-$PP_REPO_ROOT/.cache/hf}"
  export TORCH_HOME="${TORCH_HOME:-$PP_REPO_ROOT/.cache/torch}"
  export HF_HUB_DISABLE_TELEMETRY=1
  mkdir -p "$HF_HOME" "$TORCH_HOME"
  umask 0002
  PYTHON_COMMAND=("$PP_PYTHON_BIN" -E -s -u)
else
  source scripts/activate_paperwork.sh
  export PYTHONPATH="$PP_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  PYTHON_COMMAND=(python -u)
fi

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Stage: $PP_STAGE"
echo "User: $PP_USER_ID"
echo "Worker: $PP_WORKER_ID/$PP_NUM_WORKERS"
echo "Config: $PP_CONFIG"
echo "Host: $(hostname)"
if [[ "$IS_RECEIPT" -eq 1 ]]; then
  echo "Pinned commit: $PP_EXPECTED_GIT_SHA"
  echo "Python: $PP_PYTHON_BIN"
  echo "Read-only artifacts: $PP_SHARED_ARTIFACT_ROOT"
  echo "Receipt outputs: $PP_RECEIPT_OUTPUT_ROOT"
fi
nvidia-smi

"${PYTHON_COMMAND[@]}" scripts/run_gpu_stage.py \
  --stage "$PP_STAGE" \
  --config "$PP_CONFIG" \
  --user-id "$PP_USER_ID" \
  --worker-id "$PP_WORKER_ID" \
  --num-workers "$PP_NUM_WORKERS"
