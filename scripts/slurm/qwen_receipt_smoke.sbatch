#!/usr/bin/env bash
# Owner-only Qwen receipt smoke. Submit through the fail-closed wrapper:
#   bash scripts/submit_qwen_smoke.sh \
#     configs/qwen_attack/receipt_smoke_saaketh.json
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --job-name=pp_qwen_receipt_smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${PP_REPO_ROOT:?Submit with scripts/submit_qwen_smoke.sh}"
: "${PP_CONFIG:?PP_CONFIG is required}"
: "${PP_EXPECTED_GIT_SHA:?PP_EXPECTED_GIT_SHA is required}"
: "${PP_TRACK:?PP_TRACK is required}"
: "${PP_USER_ID:?PP_USER_ID is required}"
: "${PP_WORKER_ID:?PP_WORKER_ID is required}"
: "${PP_NUM_WORKERS:?PP_NUM_WORKERS is required}"

cd "$PP_REPO_ROOT"
source "$PP_REPO_ROOT/scripts/lib/qwen_receipt_runtime.sh"
CONFIG_ARG="${1:-$PP_CONFIG}"
if [[ "$CONFIG_ARG" != /* ]]; then
  CONFIG_ARG="$PP_REPO_ROOT/$CONFIG_ARG"
fi
pp_receipt_validate_context \
  "$PP_REPO_ROOT" "$PP_CONFIG" "$PP_EXPECTED_GIT_SHA" "GPU smoke execution"
[[ "$CONFIG_ARG" == "$PP_VALIDATED_CONFIG" ]] || pp_receipt_die \
  "Slurm config argument does not match exported PP_CONFIG"
[[ "$PP_TRACK" == "receipt" && "$PP_CONFIG_MODE" == "smoke" ]] || \
  pp_receipt_die "this batch script only accepts the receipt smoke config"
[[ "$PP_CONFIG_READY_SMOKE" == "true" ]] || pp_receipt_die \
  "receipt GPU smoke remains gated"
[[ "$PP_USER_ID" == "user3" && "$PP_CONFIG_OWNER" == "user3" ]] || \
  pp_receipt_die "receipt smoke must run as its declared owner user3"
[[ "$PP_WORKER_ID" == "0" && "$PP_NUM_WORKERS" == "1" ]] || \
  pp_receipt_die "receipt smoke must run as one worker (0 of 1)"
[[ "${PP_RUN_ID:-}" == "$PP_CONFIG_RUN_ID" ]] || pp_receipt_die \
  "exported run ID does not match the committed smoke config"

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

echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Commit: $PP_EXPECTED_GIT_SHA"
echo "Config: $PP_VALIDATED_CONFIG"
echo "Python: $PP_PYTHON_BIN"
echo "Python prefix checked by launcher"
echo "Read-only artifacts: $PP_SHARED_ARTIFACT_ROOT"
echo "Receipt outputs: $PP_RECEIPT_OUTPUT_ROOT"
echo "Host: $(hostname)"
nvidia-smi
"$PP_PYTHON_BIN" -E -s -u scripts/check_qwen_processor_parity.py \
  --config "$PP_VALIDATED_CONFIG"
"$PP_PYTHON_BIN" -E -s -u scripts/smoke_qwen_attack.py \
  --config "$PP_VALIDATED_CONFIG" --user-id "$PP_USER_ID"
