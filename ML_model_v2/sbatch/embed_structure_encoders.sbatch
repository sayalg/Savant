#!/bin/bash
#SBATCH --job-name=savant-struct-emb
#SBATCH --output=logs/struct_emb_%A_%a.out
#SBATCH --error=logs/struct_emb_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
# --------------------------------------------------------
set -euo pipefail

# Fold-based protein embeddings (ESMFold2, and best-effort SaProt), sharded across up to
# 12 GPUs. One GPU per task - folding is the bottleneck, not CPU, so more GPUs beats more
# cores per GPU.
#
# BEFORE submitting this: run the selfcheck ONCE, on a single GPU node, and read its
# output. It reports pass/fail for the whole chain on one protein and, for esmfold2,
# prints a warm-call GPU-hour projection you should sanity-check against 28.9 GPU-hours
# (the number measured earlier this project at batch 4) before trusting the array's
# runtime budget:
#
#   python scripts/embed_structure_encoders.py --mode esmfold2 --selfcheck
#   python scripts/embed_structure_encoders.py --mode saprot   --selfcheck   # best-effort,
#                                                                            # see the
#                                                                            # script's
#                                                                            # docstring
#
# Usage:
#   sbatch --array=0-11 sbatch/embed_structure_encoders.sbatch esmfold2 work/protein_manifest.txt
#   # after all 12 tasks complete:
#   python scripts/embed_structure_encoders.py --mode esmfold2 --manifest work/protein_manifest.txt --merge
#
# Shard count is taken from SLURM_ARRAY_TASK_COUNT inside the python script itself, not
# passed here - the same reasoning as extract_features.sbatch: a --array value that
# disagrees with a separately-set shard count silently drops part of the manifest with
# nothing in the logs to say so, and that mistake has already happened once in this repo.

MODE="${1:?usage: sbatch ... embed_structure_encoders.sbatch <esmfold2|saprot> <manifest.txt>}"
MANIFEST="${2:?usage: sbatch ... embed_structure_encoders.sbatch <esmfold2|saprot> <manifest.txt>}"

echo "== shard ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} == mode ${MODE} =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$(dirname "$0")/.."
mkdir -p logs work/emb

python scripts/embed_structure_encoders.py \
    --mode "$MODE" \
    --manifest "$MANIFEST" \
    --batch-size "${BATCH_SIZE:-2}" \
    --on-invalid "${ON_INVALID:-fail}"

echo "== shard ${SLURM_ARRAY_TASK_ID} done =="
