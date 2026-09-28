#!/bin/bash
#SBATCH --job-name=pfpl-train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=FP32
# ---------------------------------------------------------------------------
# Phase 9: the pre-registered ablation matrix, one cell per array task.
#
# The grid lives in configs/preregistration.yaml and is expanded by
# scripts/phase9_ablation.py, so this file contains no science and cannot drift out
# of sync with what was declared.
#
#   N=$(python scripts/phase9_ablation.py --emit --count)
#   sbatch --array=0-$((N-1))%12 slurm/60_train_grid.sbatch
#
# The %12 caps concurrency at the 12-GPU allocation. Every cell is independent, so
# the queue does the scheduling.
#
# Single cell, by hand:
#   sbatch slurm/60_train_grid.sbatch -o model=variant_c -o plm=none
#
# Checkpoint/resume is automatic (train/loop.py writes last.pt each epoch), so a
# wall-time kill costs one epoch, not the run. Re-submitting the same array is safe.
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ $# -eq 0 ]; then
    CELL=$(python "${PFPL_ROOT}/scripts/phase9_ablation.py" --cell "${SLURM_ARRAY_TASK_ID}" \
             ${GRID_SEEDS:+--seeds "$GRID_SEEDS"} ${GRID_FULL:+--full})
    echo "== cell ${SLURM_ARRAY_TASK_ID}: ${CELL} =="
    # shellcheck disable=SC2086
    set -- ${CELL}
fi

# train.py exits 0 normally. A failed GATE is information, not a crash, so the
# downstream report must still run - hence afterany in submit_wave2.sh.
pfpl_run train.py "$@"
