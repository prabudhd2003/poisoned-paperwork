#!/usr/bin/env bash
# Submit one team member's checkpointed GPU shard to USC CARC.
#
# Usage:
#   bash scripts/submit_stage.sh STAGE USER_ID [CONFIG] [--dry-run]
#
# Example:
#   bash scripts/submit_stage.sh qwen_attack user1

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/submit_stage.sh STAGE USER_ID [CONFIG] [--dry-run]" >&2
  exit 2
fi

STAGE="$1"
USER_ID="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/$STAGE/active.json"
DRY_RUN=0

for arg in "${@:3}"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    CONFIG_PATH="$arg"
  fi
done

case "$USER_ID" in
  user1) NAME="Prabudhd"; WORKER_ID=0 ;;
  user2) NAME="Gary";     WORKER_ID=1 ;;
  user3) NAME="Saaketh";  WORKER_ID=2 ;;
  user4) NAME="Khalid";   WORKER_ID=3 ;;
  user5) NAME="Shail";    WORKER_ID=4 ;;
  *)
    echo "Unknown user ID '$USER_ID'. Use user1, user2, user3, user4, or user5." >&2
    exit 2
    ;;
esac

GPU_TYPE=""
CONSTRAINT=""
CPUS=8
MEMORY=""
TIME_LIMIT=""

case "$STAGE" in
  baseline)
    GPU_TYPE="l40s"; MEMORY="32G"; TIME_LIMIT="04:00:00" ;;
  donut_attack)
    GPU_TYPE="l40s"; MEMORY="48G"; TIME_LIMIT="08:00:00" ;;
  qwen_attack)
    GPU_TYPE="a100"; CONSTRAINT="a100-80gb"; MEMORY="64G"; TIME_LIMIT="12:00:00" ;;
  transfer_robustness)
    GPU_TYPE="l40s"; MEMORY="48G"; TIME_LIMIT="08:00:00" ;;
  eot_patch)
    GPU_TYPE="a100"; CONSTRAINT="a100-80gb"; MEMORY="64G"; TIME_LIMIT="12:00:00" ;;
  defenses)
    GPU_TYPE="a100"; CONSTRAINT="a100-80gb"; MEMORY="64G"; TIME_LIMIT="12:00:00" ;;
  *)
    echo "Unknown stage '$STAGE'." >&2
    echo "Choose: baseline, donut_attack, qwen_attack, transfer_robustness, eot_patch, defenses" >&2
    exit 2
    ;;
esac

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  echo "The stage owner must create configs/$STAGE/active.json before workers submit." >&2
  exit 2
fi

RUN_ID="$(python3 - "$CONFIG_PATH" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text()).get("run_id")
if not isinstance(value, str) or not value.strip():
    raise SystemExit("Config must contain a non-empty string field named 'run_id'.")
print(value)
PY
)"

mkdir -p "$REPO_ROOT/logs"

SBATCH_ARGS=(
  --parsable
  --account=yzhao010_1531
  --partition=gpu
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$TIME_LIMIT"
  --gpus-per-task="$GPU_TYPE:1"
  --job-name="pp_${STAGE}_w${WORKER_ID}"
  --chdir="$REPO_ROOT"
  --output="$REPO_ROOT/logs/%x_%j.out"
  --error="$REPO_ROOT/logs/%x_%j.err"
  --export="ALL,PP_REPO_ROOT=$REPO_ROOT,PP_STAGE=$STAGE,PP_RUN_ID=$RUN_ID,PP_CONFIG=$CONFIG_PATH,PP_USER_ID=$USER_ID,PP_WORKER_ID=$WORKER_ID,PP_NUM_WORKERS=5"
)

if [[ -n "$CONSTRAINT" ]]; then
  SBATCH_ARGS+=(--constraint="$CONSTRAINT")
fi

echo "Stage: $STAGE"
echo "Run: $RUN_ID"
echo "Team member: $USER_ID ($NAME)"
echo "Worker shard: $WORKER_ID of 5"
echo "GPU request: $GPU_TYPE ${CONSTRAINT:+($CONSTRAINT)}"
echo "CPU/RAM/time: $CPUS CPUs, $MEMORY, $TIME_LIMIT"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" "$REPO_ROOT/scripts/slurm/gpu_stage.sbatch"
  printf '\n'
  exit 0
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable. Run this command from a USC CARC login node." >&2
  exit 1
}

JOB_ID="$(sbatch "${SBATCH_ARGS[@]}" "$REPO_ROOT/scripts/slurm/gpu_stage.sbatch")"
echo "Submitted Slurm job $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Logs:    tail -f $REPO_ROOT/logs/pp_${STAGE}_w${WORKER_ID}_${JOB_ID}.out"
