#!/bin/bash
#SBATCH --job-name=savant2-features
#SBATCH --output=logs/features_%A_%a.out
#SBATCH --error=logs/features_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-19
#SBATCH --partition=Orion
# --------------------------------------------------------
set -euo pipefail

# Extract v2 features over every complex in a manifest.
#
# One core per task is deliberate: extraction is single-threaded numpy/scipy at
# ~0.3 s per complex, so 20 one-core tasks beat 5 four-core ones. The 47,619
# training complexes are ~4 CPU-hours total, about 12 minutes wall clock here.
#
# For comparison, producing the v1 feature table required a HADDOCK3
# minimisation per complex.
#
#   sbatch --array=0-19 sbatch/extract_features.sbatch work/manifest_training.csv
#   sbatch --array=0-3  sbatch/extract_features.sbatch work/manifest_benchmark.csv

MANIFEST="${1:-work/manifest_training.csv}"

# Derive the shard count from the array itself. Setting it independently is a
# trap: --array=0-3 against a hardcoded NUM_SHARDS=20 silently processes 4/20ths
# of the manifest and every downstream number is then computed on a fifth of the
# data, with nothing in the logs to say so. SLURM_ARRAY_TASK_COUNT is the size of
# the submitted array, so the two can never disagree.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
echo "== shard ${SLURM_ARRAY_TASK_ID:-0} of ${NUM_SHARDS} =="
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "${NUM_SHARDS}" -ne "${SLURM_ARRAY_TASK_COUNT}" ]; then
    echo "WARNING: NUM_SHARDS=${NUM_SHARDS} but the array has ${SLURM_ARRAY_TASK_COUNT} tasks;" >&2
    echo "         part of the manifest will not be processed." >&2
fi

cd "$(dirname "$0")/.."
mkdir -p logs

python scripts/01_extract_features.py \
    --manifest "${MANIFEST}" \
    --num-shards "${NUM_SHARDS}"

echo "Next: python scripts/02_train.py"
