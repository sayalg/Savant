#!/bin/bash
#SBATCH --job-name=bench-prodigy
#SBATCH --output=logs/prodigy_%j.out
#SBATCH --error=logs/prodigy_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=Orion
# --------------------------------------------------------
set -euo pipefail

# Stage 3 of 3: score the same structures with PRODIGY-LIG.
#
# Seconds per pair, so no array - the whole benchmark fits in one short serial
# job. Run it AFTER Savant if you want the electrostatics-aware variant too: it
# reads the HADDOCK elec term out of each Savant JSON and runs PRODIGY a second
# time with -e. Without those JSONs it still produces the contacts-only score.
#
#   sbatch sbatch/run_prodigy.sbatch
# --------------------------------------------------------------

cd "$(dirname "$0")/.."          # ML_model/benchmark/eval
mkdir -p logs

command -v prodigy_lig >/dev/null || { echo "prodigy_lig not on PATH (pip install prodigy-lig)" >&2; exit 1; }

python scripts/05_run_prodigy.py

echo "Next: python scripts/06_merge_scores.py"
