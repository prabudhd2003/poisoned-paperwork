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

# Slurm exports the submitter's environment by default. If the submit shell was
# already inside an environment, CONDA_PREFIX/CONDA_SHLVL can survive while the
# module commands above replace PATH with the new base installation. Conda may
# then incorrectly no-op because the inherited environment name matches. Clear
# only Conda's inherited activation bookkeeping and resolve the environment
# afresh on this host.
for conda_prefix_variable in $(compgen -A variable CONDA_PREFIX); do
  unset "$conda_prefix_variable"
done
unset CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
conda activate paperwork
hash -r

if [[ -z "${CONDA_PREFIX:-}" || "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]]; then
  echo "paperwork activation failed: python=$(command -v python), CONDA_PREFIX=${CONDA_PREFIX:-unset}" >&2
  return 1
fi

export PYTHONNOUSERSITE=1          # keep ~/.local off the path
export HF_HOME="$CACHE_ROOT/hf"    # models into the repo .cache, not $HOME
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "paperwork active -- $(python -V 2>&1), python=$(command -v python), HF_HOME=$HF_HOME"
