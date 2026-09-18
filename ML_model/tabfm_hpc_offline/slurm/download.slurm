#!/bin/bash
#SBATCH --job-name=tabfm-dl
#SBATCH --partition=transfer         # a partition/node WITH internet (edit me)
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/tabfm-dl-%j.out

# Download TabFM weights on an internet-capable node.
# Many clusters expose internet only on login or dedicated data-transfer nodes.
# If your scheduler has NO online partition, just run download_weights.py
# directly on the login node instead of sbatch'ing this.

set -euo pipefail
mkdir -p logs

# --- edit these -------------------------------------------------------------
WEIGHTS_DIR="${WEIGHTS_DIR:-/scratch/$USER/tabfm}"   # scratch/project, not home!
TASK="${TASK:-classification}"                        # classification|regression|both
# export HF_TOKEN=hf_xxx   # only if the repo is gated
# ----------------------------------------------------------------------------

module load python/3.11 || true          # adjust to your site's module system
source "${VENV:-$HOME/tabfm-venv}/bin/activate"

# Keep HF caches off the home quota.
export HF_HOME="${WEIGHTS_DIR}/.hf_home"

python download_weights.py --task "${TASK}" --dest "${WEIGHTS_DIR}" --workers 8
echo "Weights ready at ${WEIGHTS_DIR}"
