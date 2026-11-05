#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---check}"

if [[ "$MODE" != "--check" && "$MODE" != "--submit" ]]; then
  echo "Usage: bash scripts/submit_remaining_experiments.sh [--check|--submit]" >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to continue with a dirty worktree." >&2
  exit 1
fi

source scripts/activate_paperwork.sh
python scripts/manage_carc_storage.py verify

JOBS=(
  "qwen_attack|outputs/runtime_configs/qwen_collected.json"
  "donut_attack|outputs/runtime_configs/donut_collected.json"
  "defenses|outputs/runtime_configs/defenses_donut_public_final_v1.json"
  "jpeg_eot_attack|outputs/runtime_configs/jpeg_eot_public_full_v1.json"
  "jpeg_eot_attack|outputs/runtime_configs/jpeg_eot_hardview_public_full_v1.json"
  "jpeg_eot_attack|outputs/runtime_configs/donut_jpeg_eot_public_full_v1.json"
  "jpeg_eot_attack|outputs/runtime_configs/donut_hardview_public_full_v1.json"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r stage config <<< "$entry"
  if [[ ! -s "$config" ]]; then
    echo "Missing runtime config: $config" >&2
    exit 1
  fi
  python scripts/preflight_stage.py --stage "$stage" --config "$config"
done

if [[ "$MODE" == "--check" ]]; then
  echo "ALL SEVEN EXPERIMENT ARRAYS PASSED PREFLIGHT"
  echo "Re-run with --submit to submit 7 arrays containing 35 worker tasks."
  exit 0
fi

for entry in "${JOBS[@]}"; do
  IFS='|' read -r stage config <<< "$entry"
  bash scripts/submit_stage.sh "$stage" all "$config"
done

echo "SUBMITTED ALL SEVEN EXPERIMENT ARRAYS"
