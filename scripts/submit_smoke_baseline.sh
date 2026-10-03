#!/usr/bin/env bash
# Submit the owner-only baseline smoke test as a persistent Slurm batch job.
# The job remains queued/running after the terminal or laptop disconnects.
#
# Usage:
#   bash scripts/submit_smoke_baseline.sh [a100|a40|l40s] [TIME_LIMIT]
#
# Example:
#   bash scripts/submit_smoke_baseline.sh a100 02:00:00

set -euo pipefail

GPU_TYPE="${1:-a100}"
TIME_LIMIT="${2:-02:00:00}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/baseline/smoke.json"

case "$GPU_TYPE" in
  a100|a40|l40s) ;;
  *)
    echo "GPU must be a100, a40, or l40s (V100/P100 do not support this BF16 smoke config)." >&2
    exit 2
    ;;
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Smoke config not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty Git worktree. Commit or restore changes first." >&2
  exit 1
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is unavailable. Run this command from a USC CARC login node." >&2
  exit 1
}

mkdir -p "$REPO_ROOT/logs"

JOB_ID="$(sbatch \
  --parsable \
  --account=yzhao010_1531 \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time="$TIME_LIMIT" \
  --gpus-per-task="$GPU_TYPE:1" \
  --job-name=pp_baseline_smoke \
  --chdir="$REPO_ROOT" \
  --output="$REPO_ROOT/logs/%x_%j.out" \
  --error="$REPO_ROOT/logs/%x_%j.err" \
  --export="ALL,PP_REPO_ROOT=$REPO_ROOT,PP_SMOKE_CONFIG=$CONFIG_PATH" \
  "$REPO_ROOT/scripts/slurm/baseline_smoke.sbatch")"

echo "Submitted persistent baseline smoke job: $JOB_ID"
echo "Requested: 1 $GPU_TYPE GPU, 8 CPUs, 32G RAM, $TIME_LIMIT"
echo "You may close the terminal or laptop now."
echo "Status later: squeue -j $JOB_ID"
echo "Accounting:   sacct -j $JOB_ID --format=JobID,State,Elapsed,ExitCode,NodeList"
echo "Output log:   $REPO_ROOT/logs/pp_baseline_smoke_${JOB_ID}.out"
echo "Error log:    $REPO_ROOT/logs/pp_baseline_smoke_${JOB_ID}.err"
