#!/bin/bash
#SBATCH --job-name=bench-savant
#SBATCH --output=logs/savant_%A_%a.out
#SBATCH --error=logs/savant_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --partition=Orion
# --------------------------------------------------------
set -euo pipefail

# Stage 2 of 3: score the Boltz structures with Savant (RDKit + HADDOCK3).
#
# CPU only, but the slowest stage per pair: each one runs HADDOCK3 topology
# generation and energy minimisation. Array size and NUM_SHARDS must agree
# (--array=0-19 <-> NUM_SHARDS=20); pairs are handed out round-robin so the
# large complexes spread evenly across tasks.
#
# Already-scored pairs are skipped, so re-running the array retries only the
# failures listed in work/results/savant_failures_shard*.csv.
#
#   sbatch sbatch/run_savant.sbatch
# -------------------------------------------------------------------
mkdir -p logs

NUM_SHARDS="${NUM_SHARDS:-20}"

command -v savant   >/dev/null || { echo "savant not on PATH" >&2; exit 1; }
command -v haddock3 >/dev/null || { echo "haddock3 not on PATH (Savant shells out to it)" >&2; exit 1; }

# HADDOCK run directories are large and write-heavy; keep them on node-local
# scratch and let the script delete each one after the pair is scored.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

python scripts/04_run_savant.py \
    --num-shards "${NUM_SHARDS}" \
    --ncores "${SLURM_CPUS_PER_TASK:-8}" \
    --scratch "${TMPDIR}"

echo "Next (after all array tasks finish): python scripts/05_run_prodigy.py"
