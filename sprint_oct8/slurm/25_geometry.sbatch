#!/bin/bash
#SBATCH --job-name=pfpl-geom
#SBATCH --output=logs/geom_%A_%a.out
#SBATCH --error=logs/geom_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Phase 2: distogram and pocket labels from crystal complexes.
#
# THIS IS NOT OPTIONAL FOR A PHYSICS CLAIM. PLAN.md Section 2.1 lists the supervised
# distogram anchor as a REQUIRED mitigation, and Section 0.3.4 keeps it on through
# every later stage. Without it, the affinity loss is free to push probability mass
# toward each pair's energy minimum, and the model converges on an over-attractive
# geometry that scores well on pK and is not physics.
#
#   PDBBIND=/scratch/$USER/pdbbind_v2020 sbatch --array=0-7 slurm/25_geometry.sbatch
#   python scripts/phase2_geometry.py --merge
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

[ -n "${PDBBIND:-}" ] || { echo "set PDBBIND=<dir> (this project never downloads it; stage a licensed copy)" >&2; exit 1; }
pfpl_run phase2_geometry.py --pdbbind "$PDBBIND" ${ALLOW_IC50:+--allow-ic50} "$@"
