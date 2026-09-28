#!/bin/bash
#SBATCH --job-name=pfpl-embed
#SBATCH --output=logs/embed_%A_%a.out
#SBATCH --error=logs/embed_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=FP32
# ---------------------------------------------------------------------------
# Phase 4: per-residue protein embeddings, one GPU per shard.
#
# PER-RESIDUE, NOT POOLED. A pooled vector says which family a protein belongs to
# and nothing about which residues line the pocket; that representation scored
# per-target rho 0.158 against 0.325 for a protein-blind control.
#
# Estimate the storage first - Phase 4's gate asks you to confirm anything over
# 100 GB before launching:
#   python scripts/phase4_embed.py --estimate -o plm=bottleneck
#
# Shard geometry comes from SLURM_ARRAY_TASK_COUNT inside the script, never from
# here. NOTE: `sbatch --array=9` sets that count to 1, so re-running a single shard
# of a 12-way array needs --shard-id 9 --n-shards 12 passed explicitly. The script
# detects and refuses the ambiguous case; a silent shard-count mismatch has already
# cost this project one full benchmark run.
#
#   python scripts/phase4_embed.py --selfcheck            # on a GPU node, once
#   sbatch --array=0-11 slurm/40_embed_protein.sbatch
#   PLM=full sbatch --array=0-11 slurm/40_embed_protein.sbatch
#   python scripts/phase4_embed.py --merge  -o plm=bottleneck
#   python scripts/phase4_embed.py --verify -o plm=bottleneck
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

# Fragmentation was part of the OOM ("5.81 GiB reserved but unallocated"): variable
# sequence lengths leave holes the caching allocator cannot reuse.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Re-running the array is safe: finished shards are skipped, so only the ones that
# failed are recomputed. Keep the SAME --array range as the first run (0-11) - the
# shard geometry comes from the task count.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pfpl_run phase4_embed.py -o "plm=${PLM:-bottleneck}" --batch-tokens "${BATCH_TOKENS:-8192}" "$@"
