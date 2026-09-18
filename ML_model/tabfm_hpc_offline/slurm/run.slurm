#!/bin/bash
#SBATCH --job-name=tabfm-run
#SBATCH --partition=gpu               # edit to your GPU (or CPU) partition
#SBATCH --gres=gpu:1                  # drop this line for a CPU-only run
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                     # TabFM memory grows with training rows
#SBATCH --output=logs/tabfm-run-%j.out

# Train + infer on an OFFLINE compute node. Weights must already be on disk.

set -euo pipefail
mkdir -p logs

# --- edit these -------------------------------------------------------------
WEIGHTS_DIR="${WEIGHTS_DIR:-/scratch/$USER/tabfm}"
TASK="${TASK:-classification}"
TRAIN="${TRAIN:-data/train.csv}"
TEST="${TEST:-data/test.csv}"
TARGET="${TARGET:-label}"
OUT="${OUT:-results/${SLURM_JOB_ID:-local}}"
# ----------------------------------------------------------------------------

module load python/3.11 cuda/12.4 || true   # adjust to your site
source "${VENV:-$HOME/tabfm-venv}/bin/activate"

# Belt-and-braces: force offline even though run_tabfm.py also sets it.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${WEIGHTS_DIR}/.hf_home"

python run_tabfm.py \
    --task "${TASK}" \
    --weights "${WEIGHTS_DIR}" \
    --train "${TRAIN}" \
    --test "${TEST}" \
    --target "${TARGET}" \
    --out "${OUT}" \
    --device auto

echo "Artifacts in ${OUT}"
