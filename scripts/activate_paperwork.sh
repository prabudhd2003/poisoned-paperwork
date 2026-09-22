# Activate the "paperwork" conda env. SOURCE this file, do not run it:
#
#   source scripts/activate_paperwork.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="$REPO_ROOT/.cache"

if ! command -v module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi

module purge
module load gcc/13.3.0 cuda/12.4.0 conda

# conda activate needs this in a fresh shell unless you ran `conda init bash`.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate paperwork

export PYTHONNOUSERSITE=1          # keep ~/.local off the path
export HF_HOME="$CACHE_ROOT/hf"    # models into the repo .cache, not $HOME
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_TELEMETRY=1

echo "paperwork active -- $(python -V 2>&1), HF_HOME=$HF_HOME"
