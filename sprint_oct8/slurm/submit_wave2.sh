#!/bin/bash
# Wave 2: the MVP physics model and the ablation matrix (Phases 7-9, Gate G2).
#
#     [phase7 selftest]   <- runs HERE, synchronously, and blocks on failure
#            |
#     [60 train grid: N cells, %12 concurrent]
#            |
#     [70 report + GATE G2]
#
# The selfcheck runs before anything is submitted because a wiring fault - a
# detached gradient, a mask on the wrong axis - costs the whole grid, and on 600,000
# pairs a broken model and a working one both plateau at the label variance.
#
# NO `cd` ANYWHERE. Submit from the project root, or export PFPL_ROOT.
#
#   ./slurm/submit_wave2.sh
#   DRY=1 ./slurm/submit_wave2.sh
#   GRID_FULL=1 ./slurm/submit_wave2.sh        # full product, not one-at-a-time
#   GRID_SEEDS=3 ./slurm/submit_wave2.sh       # 3 seeds per cell
#   MAX_CONCURRENT=12 ./slurm/submit_wave2.sh
set -euo pipefail

: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "${PFPL_ROOT}/src/pfpl" ]; then
    echo "run this from the project root, or set PFPL_ROOT=/abs/path" >&2
    exit 1
fi
# This launcher runs python ITSELF on the login node - the Phase 7 selftest and the
# grid enumeration - so it needs the same environment the sbatch scripts get. It did
# not source this before, which is why the selftest died on
# "GLIBCXX_3.4.31 not found" (optree's _C.so) while every sbatch job was fine, and
# why CONDA_LIBS_FIRST=1 on the command line changed nothing: the variable is read
# here, in a file that was never loaded.
source "${PFPL_ROOT}/slurm/_prelude.sh"
mkdir -p "${PFPL_ROOT}/logs"

DRY="${DRY:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"
SPLIT="${SPLIT:-protein_cluster}"

sub() {
    local desc="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "DRY: sbatch $*   # $desc" >&2
        echo "000000"
        return
    fi
    local id
    id=$(sbatch --parsable "$@")
    echo "  submitted ${id}  ${desc}" >&2
    echo "$id"
}

# ---- Gate G1 must have happened --------------------------------------------
if ! ls "${PFPL_WORK}"/runs/baselines_*/reports.json >/dev/null 2>&1; then
    cat >&2 <<'EOF'
No Phase 6 baselines found.

PLAN.md Section 0.3.1: "Build honest splits and all baselines first. Every later
claim is relative to them. The physics work may not be judged against numbers that
do not exist yet."

This is not procedural. On the previous corpus a ligand-only model beat every
protein-aware model by a factor of two on held-out clusters; running the baselines
first is the difference between finding that out in week one and week three.

    ./slurm/submit_wave1.sh
EOF
    exit 1
fi

# ---- Phase 7 gate, synchronously -------------------------------------------
echo "== phase 7 selftest (physics + wiring) ==" >&2
if [ "$DRY" != "1" ]; then
    if ! python "${PFPL_ROOT}/scripts/phase7_selftest.py" --overfit; then
        echo "" >&2
        echo "Phase 7 gate FAILED - not submitting the grid." >&2
        echo "Section 8: all physics tests must pass and the model must overfit a" >&2
        echo "tiny subset before any training run. Fix it here, not at scale." >&2
        exit 1
    fi
fi

# ---- the grid ---------------------------------------------------------------
N=$(python "${PFPL_ROOT}/scripts/phase9_ablation.py" --emit --count \
      ${GRID_SEEDS:+--seeds "$GRID_SEEDS"} ${GRID_FULL:+--full})
echo "== ablation matrix: ${N} cells, ${MAX_CONCURRENT} concurrent ==" >&2
python "${PFPL_ROOT}/scripts/phase9_ablation.py" --emit \
    ${GRID_SEEDS:+--seeds "$GRID_SEEDS"} ${GRID_FULL:+--full} >&2

GRID=$(sub "ablation grid (${N} cells)" \
    --array="0-$((N - 1))%${MAX_CONCURRENT}" \
    --export="ALL,GRID_SEEDS=${GRID_SEEDS:-},GRID_FULL=${GRID_FULL:-}" \
    "${PFPL_ROOT}/slurm/60_train_grid.sbatch")

# afterany, not afterok: a grid where one cell OOMs should still produce the table
# for the cells that finished, and a FAILED GATE is information rather than a crash.
REP=$(sub "Gate G2 + leaderboard" \
    --dependency="afterany:${GRID}" \
    --export="ALL,SPLIT=${SPLIT}" \
    "${PFPL_ROOT}/slurm/70_report.sbatch")

cat >&2 <<EOF

submitted ${N} training cells (job ${GRID}) and the report (job ${REP}).

  squeue -u \$USER -o '%.10i %.24j %.9T %.10M %.6D %R'
  tail -f ${PFPL_ROOT}/logs/train_${GRID}_0.out
  python ${PFPL_ROOT}/scripts/status.py

GATE G2 (Tier 1), Section 9, applied by scripts/phase9_ablation.py --gate:
  the best MVP configuration must beat the MATCHED no-physics control
  (model=variant_c) on ${SPLIT} by dPearson >= 0.02 with a bootstrap CI
  excluding zero, AND beat ligand_only and target_mean.

  If it fails, the diagnosis ORDER is PLM mode -> geometry/calibration ->
  individual terms -> data quality, and --gate prints it filled in with this
  run's numbers. Section 8 is explicit: if it still fails, STOP the physics
  expansion and deliver the best fusion model with an honest report of the
  ablations. Do not proceed to Wave 3 by default.

Every cell appends a row to reports/compute_ledger.csv. Section 7.4 asks for a
multiplicity-aware reading: a dPearson of 0.02 found after ${N} trials is weak
evidence, and the report states the count so that is visible rather than implied.
EOF
