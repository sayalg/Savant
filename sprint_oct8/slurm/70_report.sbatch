#!/bin/bash
#SBATCH --job-name=pfpl-report
#SBATCH --output=logs/report_%j.out
#SBATCH --error=logs/report_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Gate G2 and the leaderboard. Wired `afterany` so a grid where one cell OOMs still
# produces the table for the cells that finished - a partial leaderboard beats no
# leaderboard on a deadline.
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

set +e
pfpl_run phase9_ablation.py --gate --split "${SPLIT:-protein_cluster}"
gate=$?
set -e
echo "== gate exit ${gate} (3 = ran fine, Tier 1 not cleared) =="

pfpl_run report.py
pfpl_run status.py || true
