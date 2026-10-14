#!/usr/bin/env bash
# Submit Saaketh's one-worker Qwen-receipt smoke with an exact runtime contract.
#
# Required environment variables:
#   PP_PYTHON_BIN           absolute full-runtime interpreter
#   PP_SHARED_ARTIFACT_ROOT absolute read-only baseline/data checkout
#   PP_RECEIPT_OUTPUT_ROOT  absolute writable receipt output root
#
# Usage:
#   bash scripts/submit_qwen_smoke.sh \
#     configs/qwen_attack/receipt_smoke_saaketh.json [--dry-run]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$REPO_ROOT/scripts/lib/qwen_receipt_runtime.sh"

CONFIG_PATH="configs/qwen_attack/receipt_smoke_saaketh.json"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) CONFIG_PATH="$arg" ;;
  esac
done
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

pp_receipt_validate_context "$REPO_ROOT" "$CONFIG_PATH" "" "smoke submission"
REPO_ROOT="$PP_VALIDATED_REPO_ROOT"
CONFIG_PATH="$PP_VALIDATED_CONFIG"
EXPECTED_GIT_SHA="$PP_VALIDATED_GIT_SHA"

[[ "$PP_CONFIG_MODE" == "smoke" ]] || pp_receipt_die \
  "owner smoke launcher requires mode=smoke"
[[ "$PP_CONFIG_OWNER" == "user3" ]] || pp_receipt_die \
  "receipt smoke owner_user_id must remain user3 (Saaketh)"
[[ "$PP_CONFIG_READY_SMOKE" == "true" ]] || pp_receipt_die \
  "receipt GPU smoke remains gated (ready_for_gpu_smoke is not true)"

umask 0002
LOG_DIR="$PP_RECEIPT_OUTPUT_ROOT/logs/$PP_CONFIG_RUN_ID"
if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$LOG_DIR"
  [[ -d "$LOG_DIR" && -w "$LOG_DIR" ]] || pp_receipt_die \
    "smoke log directory is not writable: $LOG_DIR"
fi

SBATCH_EXPORT="ALL,PP_REPO_ROOT=$REPO_ROOT,PP_STAGE=qwen_attack,PP_TRACK=receipt,PP_RUN_ID=$PP_CONFIG_RUN_ID,PP_CONFIG=$CONFIG_PATH,PP_USER_ID=user3,PP_WORKER_ID=0,PP_NUM_WORKERS=1,PP_EXPECTED_GIT_SHA=$EXPECTED_GIT_SHA,PP_PYTHON_BIN=$PP_PYTHON_BIN,PP_SHARED_ARTIFACT_ROOT=$PP_SHARED_ARTIFACT_ROOT,PP_RECEIPT_OUTPUT_ROOT=$PP_RECEIPT_OUTPUT_ROOT"
SBATCH_ARGS=(
  --parsable
  --account=yzhao010_1531
  --partition=gpu
  --nodes=1
  --ntasks=1
  --cpus-per-task=8
  --mem=64G
  --time=12:00:00
  --gpus-per-task=a100:1
  --constraint=a100-80gb
  --job-name=pp_qwen_receipt_smoke
  --chdir="$REPO_ROOT"
  --output="$LOG_DIR/%x_%j.out"
  --error="$LOG_DIR/%x_%j.err"
  --export="$SBATCH_EXPORT"
)

echo "Qwen-receipt owner smoke"
echo "Run: $PP_CONFIG_RUN_ID"
echo "Pinned commit: $EXPECTED_GIT_SHA"
echo "Config: $CONFIG_PATH"
echo "Python: $PP_PYTHON_BIN"
echo "Read-only artifacts: $PP_SHARED_ARTIFACT_ROOT"
echo "Shared receipt outputs: $PP_RECEIPT_OUTPUT_ROOT"
echo "GPU request: a100 (a100-80gb), 8 CPUs, 64G, 12:00:00"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'mkdir -p %q\n' "$LOG_DIR"
  printf 'sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" \
    "$REPO_ROOT/scripts/slurm/qwen_receipt_smoke.sbatch" "$CONFIG_PATH"
  printf '\n'
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable. Run this command from a USC CARC login node." >&2
  exit 1
}

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" \
  "$REPO_ROOT/scripts/slurm/qwen_receipt_smoke.sbatch" "$CONFIG_PATH")"
echo "Submitted Slurm job $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Logs:    tail -f $LOG_DIR/pp_qwen_receipt_smoke_${JOB_ID}.out"
