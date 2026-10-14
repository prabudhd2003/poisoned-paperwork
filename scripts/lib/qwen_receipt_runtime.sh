#!/usr/bin/env bash
# Shared fail-closed checks for Qwen-receipt launchers.
#
# Source this file from Bash.  The functions intentionally do not create any
# files, activate Conda, or infer an interpreter from the caller's shell.

pp_receipt_die() {
  echo "Qwen-receipt launcher error: $*" >&2
  return 1
}

pp_receipt_is_config() {
  local config_path="${1:?config path is required}"
  local classifier="python3"
  if [[ -n "${PP_PYTHON_BIN:-}" && -x "$PP_PYTHON_BIN" ]]; then
    classifier="$PP_PYTHON_BIN"
  fi
  # Return 0 for receipt, 1 for another track, and 2 for an unreadable or
  # malformed config. Callers distinguish 1 from 2 so JSON formatting cannot
  # route a receipt config through a generic launcher.
  "$classifier" -E -s - "$config_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"Cannot classify config {path}: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc
if not isinstance(value, dict):
    print(f"Config must contain a JSON object: {path}", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(0 if value.get("track") == "receipt" else 1)
PY
}

pp_receipt_require_absolute_path() {
  local name="${1:?variable name is required}"
  local value="${!name:-}"
  [[ -n "$value" ]] || pp_receipt_die "$name is required"
  [[ "$value" == /* ]] || pp_receipt_die "$name must be an absolute path: $value"
  [[ "$value" != *","* && "$value" != *$'\n'* ]] || \
    pp_receipt_die "$name may not contain a comma or newline (Slurm export safety)"
}

pp_receipt_read_config() {
  local python_bin="${1:?python interpreter is required}"
  local config_path="${2:?config path is required}"

  {
    IFS= read -r PP_CONFIG_STAGE
    IFS= read -r PP_CONFIG_TRACK
    IFS= read -r PP_CONFIG_MODE
    IFS= read -r PP_CONFIG_RUN_ID
    IFS= read -r PP_CONFIG_OWNER
    IFS= read -r PP_CONFIG_READY_SMOKE
    IFS= read -r PP_CONFIG_READY_PRODUCTION
  } < <("$python_bin" -E -s - "$config_path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    config = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Cannot read receipt config {path}: {exc}") from exc

def scalar(name, default=""):
    value = config.get(name, default)
    if value is None:
        value = ""
    if not isinstance(value, (str, bool, int, float)):
        raise SystemExit(f"Config field {name!r} must be scalar")
    text = str(value).lower() if isinstance(value, bool) else str(value)
    if "\n" in text or "\r" in text:
        raise SystemExit(f"Config field {name!r} may not contain a newline")
    return text

for field in (
    "stage", "track", "mode", "run_id", "owner_user_id",
    "ready_for_gpu_smoke", "ready_for_production",
):
    print(scalar(field))
PY
  )

  [[ -n "$PP_CONFIG_RUN_ID" ]] || pp_receipt_die \
    "receipt config must contain a non-empty run_id"
}

pp_receipt_validate_context() {
  local repo_root="${1:?repository root is required}"
  local config_path="${2:?config path is required}"
  local expected_git_sha="${3:-}"
  local purpose="${4:-submission}"
  local repo_real config_dir_real config_real config_rel actual_sha shared_real output_real

  pp_receipt_require_absolute_path PP_PYTHON_BIN
  pp_receipt_require_absolute_path PP_SHARED_ARTIFACT_ROOT
  pp_receipt_require_absolute_path PP_RECEIPT_OUTPUT_ROOT

  [[ -x "$PP_PYTHON_BIN" && ! -d "$PP_PYTHON_BIN" ]] || pp_receipt_die \
    "PP_PYTHON_BIN is not an executable file: $PP_PYTHON_BIN"
  [[ -d "$PP_SHARED_ARTIFACT_ROOT" && -r "$PP_SHARED_ARTIFACT_ROOT" && \
     -x "$PP_SHARED_ARTIFACT_ROOT" ]] || pp_receipt_die \
    "PP_SHARED_ARTIFACT_ROOT must be an existing readable/searchable directory"
  [[ -d "$PP_RECEIPT_OUTPUT_ROOT" && -w "$PP_RECEIPT_OUTPUT_ROOT" && \
     -x "$PP_RECEIPT_OUTPUT_ROOT" ]] || pp_receipt_die \
    "PP_RECEIPT_OUTPUT_ROOT must be an existing writable/searchable directory"

  repo_real="$(cd "$repo_root" && pwd -P)"
  shared_real="$(cd "$PP_SHARED_ARTIFACT_ROOT" && pwd -P)"
  output_real="$(cd "$PP_RECEIPT_OUTPUT_ROOT" && pwd -P)"
  PP_SHARED_ARTIFACT_ROOT="$shared_real"
  PP_RECEIPT_OUTPUT_ROOT="$output_real"

  if [[ "$shared_real" == "$repo_real" || "$shared_real" == "$repo_real"/* || \
        "$repo_real" == "$shared_real"/* ]]; then
    pp_receipt_die "shared receipt artifacts and the code checkout must be separate trees"
  fi
  if [[ "$output_real" == "$repo_real" || "$output_real" == "$repo_real"/* || \
        "$repo_real" == "$output_real"/* ]]; then
    pp_receipt_die "receipt outputs and the code checkout must be separate trees"
  fi
  if [[ "$output_real" == "$shared_real" || "$output_real" == "$shared_real"/* || \
        "$shared_real" == "$output_real"/* ]]; then
    pp_receipt_die "receipt outputs and read-only shared artifacts must be separate trees"
  fi

  [[ -f "$config_path" && ! -L "$config_path" ]] || pp_receipt_die \
    "receipt config must be an existing non-symlink file: $config_path"
  config_dir_real="$(cd "$(dirname "$config_path")" && pwd -P)"
  config_real="$config_dir_real/$(basename "$config_path")"
  case "$config_real" in
    "$repo_real"/*) config_rel="${config_real#"$repo_real"/}" ;;
    *) pp_receipt_die "receipt config must stay inside the code checkout: $config_real" ;;
  esac

  git -C "$repo_real" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    pp_receipt_die "not a Git checkout: $repo_real"
  git -C "$repo_real" ls-files --error-unmatch -- "$config_rel" >/dev/null 2>&1 || \
    pp_receipt_die "receipt config is not tracked by Git: $config_rel"
  actual_sha="$(git -C "$repo_real" rev-parse --verify HEAD)"
  [[ "$actual_sha" =~ ^[0-9a-f]{40}$ ]] || pp_receipt_die \
    "could not resolve an exact 40-character Git commit"
  if [[ -n "$expected_git_sha" && "$actual_sha" != "$expected_git_sha" ]]; then
    pp_receipt_die \
      "Git commit changed after submission: expected $expected_git_sha, found $actual_sha"
  fi
  if [[ -n "$(git -C "$repo_real" status --porcelain --untracked-files=normal)" ]]; then
    pp_receipt_die "the code checkout must be clean before receipt $purpose"
  fi

  "$PP_PYTHON_BIN" -E -s - "$PP_PYTHON_BIN" "$PP_RECEIPT_OUTPUT_ROOT" <<'PY'
import os
import pathlib
import site
import stat
import sys

requested_path = pathlib.Path(sys.argv[1]).absolute()
output_root = pathlib.Path(sys.argv[2]).resolve(strict=True)
requested = requested_path.resolve(strict=True)
actual = pathlib.Path(sys.executable).resolve(strict=True)
prefix = pathlib.Path(sys.prefix).resolve(strict=True)
if not os.path.samefile(requested_path, sys.executable):
    raise SystemExit(
        f"PP_PYTHON_BIN resolved to {requested}, but Python reports {actual}"
    )
launcher_prefix = requested_path.parent.parent.resolve(strict=True)
if prefix not in actual.parents and prefix != launcher_prefix:
    raise SystemExit(
        f"Interpreter {requested_path} is inconsistent with sys.prefix {prefix}"
    )
leaked = [entry for entry in sys.path if ".local/lib" in entry]
if leaked:
    raise SystemExit(f"User-site paths leaked into the isolated runtime: {leaked}")
if site.ENABLE_USER_SITE:
    raise SystemExit("Python user-site remains enabled; invoke with -s")
output_mode = stat.S_IMODE(output_root.stat().st_mode)
if not output_mode & stat.S_IWGRP or not output_mode & stat.S_IXGRP:
    raise SystemExit(
        "PP_RECEIPT_OUTPUT_ROOT must be group-writable/searchable: "
        f"{output_root} (mode={output_mode:o})"
    )
if not output_mode & stat.S_ISGID:
    raise SystemExit(
        "PP_RECEIPT_OUTPUT_ROOT must have setgid enabled: "
        f"{output_root} (mode={output_mode:o})"
    )
print(f"Receipt Python executable: {requested_path} (resolved: {actual})")
print(f"Receipt Python prefix: {prefix}")
print(f"Receipt Python version: {sys.version.split()[0]}")
print(f"Receipt output mode: {output_mode:o} (gid={output_root.stat().st_gid})")
PY

  pp_receipt_read_config "$PP_PYTHON_BIN" "$config_real"
  [[ "$PP_CONFIG_STAGE" == "qwen_attack" && "$PP_CONFIG_TRACK" == "receipt" ]] || \
    pp_receipt_die \
      "strict receipt launcher requires stage=qwen_attack and track=receipt"

  PP_VALIDATED_REPO_ROOT="$repo_real"
  PP_VALIDATED_CONFIG="$config_real"
  PP_VALIDATED_CONFIG_REL="$config_rel"
  PP_VALIDATED_GIT_SHA="$actual_sha"
}
