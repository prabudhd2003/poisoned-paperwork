#!/usr/bin/env bash
# Build the "paperwork" conda environment on USC CARC and register it as a
# Jupyter kernel that code-server / OnDemand Jupyter will show.
#
#   bash scripts/setup_carc.sh
#
# Run it on a GPU node so torch is verified against a real device:
#   salloc --account=yzhao010_1531 --partition=gpu --gres=gpu:1 \
#          --cpus-per-task=8 --mem=32G --time=2:00:00

set -euo pipefail

ENV_NAME=paperwork
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Caches live inside the repo, under .cache/ (already in .gitignore).
CACHE_ROOT="$REPO_ROOT/.cache"

if ! command -v module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash 2>/dev/null || {
    echo "ERROR: Lmod not available." >&2; exit 1; }
fi

echo "==> Modules"
module purge
# cuda sits under the gcc/13.3.0 branch of the hierarchy, so gcc must load
# first. The cuda module is optional anyway: the cu124 torch wheels bundle
# their own CUDA runtime.
module load gcc/13.3.0 cuda/12.4.0 conda

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> Caches at $CACHE_ROOT (never \$HOME -- models are tens of GB)"
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "==> Creating conda env '$ENV_NAME' (Python 3.11)"
  conda create -n "$ENV_NAME" python=3.11 -y
else
  echo "==> Reusing existing conda env '$ENV_NAME'"
fi

conda activate "$ENV_NAME"

# CRITICAL: without this, ~/.local/lib/python3.11/site-packages leaks onto the
# env's path. pip then reports packages as "already satisfied" and installs
# NOTHING into the env -- including ipykernel, which is why the kernel never
# shows up in the picker.
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel setuptools

echo "==> PyTorch (CUDA 12.4 wheels)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Project requirements"
pip install -r "$REPO_ROOT/env/requirements.txt"

echo "==> Registering Jupyter kernel"
python -m ipykernel install --user \
  --name "$ENV_NAME" --display-name "$ENV_NAME (Python 3.11)"

echo "==> Persisting env vars for shell activation"
conda env config vars set PYTHONNOUSERSITE=1 HF_HOME="$CACHE_ROOT/hf" >/dev/null

# Belt and braces: VS Code sometimes launches the interpreter directly and
# ignores the kernelspec env block, so the notebooks also set HF_HOME in their
# first cell. That notebook cell is the guarantee; this is the convenience.
echo "==> Baking env vars into the kernel spec"
python - <<PY
import json, pathlib
p = pathlib.Path.home()/".local/share/jupyter/kernels/${ENV_NAME}/kernel.json"
s = json.loads(p.read_text())
s["env"] = {"HF_HOME": "${CACHE_ROOT}/hf",
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1"}
p.write_text(json.dumps(s, indent=1))
print("patched", p)
PY

echo "==> Freezing resolved versions"
pip freeze > "$REPO_ROOT/env/requirements.lock.txt"

echo "==> Verifying"
python - <<'PY'
import sys, torch, transformers, datasets
print(f"  prefix       {sys.prefix}")
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  datasets     {datasets.__version__}")
print(f"  cuda avail   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device       {torch.cuda.get_device_name(0)}")
    print(f"  bf16 ok      {torch.cuda.is_bf16_supported()}")
else:
    print("  (no GPU on this node -- re-verify on a GPU node)")
leaked = [p for p in sys.path if ".local/lib" in p]
print(f"  user-site    {leaked or 'clean'}")
PY

cat <<MSG

Done. Reload the code-server window (Developer: Reload Window), then pick
"$ENV_NAME (Python 3.11)" in the notebook kernel selector.

Shell use:  source $REPO_ROOT/scripts/activate_paperwork.sh
Commit env/requirements.lock.txt if it changed.
MSG
