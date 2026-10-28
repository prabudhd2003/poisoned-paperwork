#!/usr/bin/env bash
# Submit a validated Qwen smoke. Receipt configs use the pinned shared-artifact
# runtime; resume configs retain the established paperwork environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$REPO_ROOT/scripts/lib/qwen_receipt_runtime.sh"
CONFIG_PATH="${1:?Usage: bash scripts/submit_qwen_smoke.sh configs/qwen_attack/<smoke>.json [--dry-run]}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 1 ]]; then
  echo "Unknown option: $2" >&2
  exit 2
fi
cd "$REPO_ROOT"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi
[[ -f "$CONFIG_PATH" ]] || { echo "Config not found: $CONFIG_PATH" >&2; exit 2; }

if pp_receipt_is_config "$CONFIG_PATH"; then
  pp_receipt_validate_context "$REPO_ROOT" "$CONFIG_PATH" "" "smoke submission"
  REPO_ROOT="$PP_VALIDATED_REPO_ROOT"
  CONFIG_PATH="$PP_VALIDATED_CONFIG"
  EXPECTED_GIT_SHA="$PP_VALIDATED_GIT_SHA"
  [[ "$PP_CONFIG_MODE" == "smoke" ]] || pp_receipt_die \
    "receipt smoke launcher requires mode=smoke"
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
    --parsable --account=yzhao010_1531 --partition=gpu --nodes=1 --ntasks=1
    --cpus-per-task=8 --mem=64G --time=02:00:00 --gpus-per-task=l40s:1
    --job-name=pp_qwen_receipt_smoke --chdir="$REPO_ROOT"
    --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err"
    --export="$SBATCH_EXPORT"
  )
  echo "Qwen-receipt owner smoke"
  echo "Run: $PP_CONFIG_RUN_ID"
  echo "Pinned commit: $EXPECTED_GIT_SHA"
  echo "Python: $PP_PYTHON_BIN"
  echo "Read-only artifacts: $PP_SHARED_ARTIFACT_ROOT"
  echo "Shared receipt outputs: $PP_RECEIPT_OUTPUT_ROOT"
  echo "GPU request: l40s, 8 CPUs, 64G, 02:00:00"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'mkdir -p %q\n' "$LOG_DIR"
    printf 'sbatch'; printf ' %q' "${SBATCH_ARGS[@]}" \
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
  echo "Logs: tail -f $LOG_DIR/pp_qwen_receipt_smoke_${JOB_ID}.out"
  exit 0
else
  CLASSIFICATION_STATUS=$?
  [[ "$CLASSIFICATION_STATUS" -eq 1 ]] || exit "$CLASSIFICATION_STATUS"
fi

[[ "$DRY_RUN" -eq 0 ]] || { echo "--dry-run is only supported for receipt smoke" >&2; exit 2; }
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
print("Qwen environment passed:", f"python={sys.executable}",
      f"torch={torch.__version__}", f"transformers={transformers.__version__}",
      f"pandas={pandas.__version__}", f"numpy={numpy.__version__}")
PY
python scripts/check_qwen_processor_parity.py --config "$CONFIG_PATH"
python scripts/smoke_qwen_attack.py --config "$CONFIG_PATH" --preflight
for conda_prefix_variable in $(compgen -A variable CONDA_PREFIX); do
  unset "$conda_prefix_variable"
done
unset CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
mkdir -p logs
JOB_ID="$(sbatch --parsable scripts/slurm/qwen_smoke.sbatch "$CONFIG_PATH")"
echo "Submitted validated Qwen L40S smoke job ${JOB_ID%%;*}"
echo "Monitor: squeue -j ${JOB_ID%%;*}"
echo "Output: logs/pp_qwen_smoke_${JOB_ID%%;*}.out"
echo "Errors: logs/pp_qwen_smoke_${JOB_ID%%;*}.err"
