#!/bin/bash
#SBATCH --job-name=savant2-esm
#SBATCH --output=logs/esm_%A_%a.out
#SBATCH --error=logs/esm_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --partition=GPU
# --------------------------------------------------------
set -euo pipefail

# ESM-2 pocket embeddings + per-complex attention tensors.
#
# Shards are sorted by sequence so each protein is embedded once per shard, and
# the on-disk cache under work/esm_cache is shared across shards - so proteins
# appearing in several shards are still only paid for once.
#
# GPU because ESM is the cost here; the geometry around it is milliseconds.
# On CPU this still runs, roughly 20x slower.
#
#   sbatch --array=0-7 sbatch/build_attention.sbatch work/manifest_training.csv
#   sbatch --array=0   sbatch/build_attention.sbatch work/manifest_benchmark.csv

MANIFEST="${1:-work/manifest_training.csv}"
# Derived from the array size, not set independently - see the note in
# extract_features.sbatch for why a mismatch here is silent and expensive.
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
echo "== shard ${SLURM_ARRAY_TASK_ID:-0} of ${NUM_SHARDS} =="

cd "$(dirname "$0")/.."
mkdir -p logs work/esm_cache

python - <<'PY' || exit 1
import sys, torch
if not torch.cuda.is_available():
    print("no GPU visible; ESM will run on CPU (slow but correct)", file=sys.stderr)
else:
    try:
        torch.zeros(1, device="cuda").add_(1)
    except RuntimeError as exc:
        sys.exit(f"CUDA unusable: {exc}\ntorch {torch.__version__} built for CUDA "
                 f"{torch.version.cuda}; check the GPU model matches the wheel.")
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY

python scripts/04_build_attention_examples.py \
    --manifest "${MANIFEST}" \
    --num-shards "${NUM_SHARDS}"

echo "Next: python scripts/05_train_attention.py --examples work/attention_training"
