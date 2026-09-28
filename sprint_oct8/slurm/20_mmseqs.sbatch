#!/bin/bash
#SBATCH --job-name=pfpl-mmseqs
#SBATCH --output=logs/mmseqs_%j.out
#SBATCH --error=logs/mmseqs_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Cluster the protein set at 30% identity. This is what makes the headline split
# honest: a protein-GROUPED split let 21 near-identical HIV-1 RT constructs sit in
# the fit fold while another RT was "held out" at 0.94 k-mer containment, and an
# internal r of 0.580 came out as 0.327 externally.
#
# mmseqs runs inside the colabfold container already on this cluster. Set
# MMSEQS_SIF to a different image, or MMSEQS_BIN if it is on PATH natively.
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

FASTA="${PFPL_WORK}/proteins.fasta"
OUT="${PFPL_WORK}/mmseqs"
ID="${MMSEQS_ID:-0.30}"
COV="${MMSEQS_COV:-0.50}"

[ -f "$FASTA" ] || { echo "missing $FASTA - run scripts/phase1_harmonize.py first" >&2; exit 1; }
mkdir -p "$OUT/tmp"

# Where mmseqs comes from, in order: MMSEQS_BIN, mmseqs on PATH, then the colabfold
# image. The first image below is the confirmed one on this cluster (2026-09-25);
# /apps/pkg/colabfold/colabfold.sif never existed. The rest are fallbacks.
SIF_CANDIDATES=(
    "${MMSEQS_SIF:-}"
    "/users/sguirale/containers/colabfold_1.6.0-cuda12.sif"
    "/users/${PFPL_USER}/containers/colabfold_1.6.0-cuda12.sif"
    "/users/${PFPL_USER}/alphafold2/colabfold_1.6.1-cuda12.sif"
    "${HOME}/alphafold2/colabfold_1.6.1-cuda12.sif"
    "${HOME}/containers/colabfold.sif"
)
SING="$(command -v singularity || command -v apptainer || true)"

if [ -n "${MMSEQS_BIN:-}" ]; then
    MODE=bin
elif command -v mmseqs >/dev/null 2>&1; then
    MODE=path
else
    MODE=sif
    SIF=""
    for c in "${SIF_CANDIDATES[@]}"; do
        [ -n "$c" ] && [ -f "$c" ] && { SIF="$c"; break; }
    done
    if [ -z "$SIF" ] || [ -z "$SING" ]; then
        {
            echo "no mmseqs available."
            echo "  singularity/apptainer: ${SING:-NOT FOUND (try: module load singularity)}"
            echo "  images checked:"
            for c in "${SIF_CANDIDATES[@]}"; do [ -n "$c" ] && echo "    $c"; done
            echo "  fix: MMSEQS_SIF=/path/to/colabfold.sif sbatch slurm/20_mmseqs.sbatch"
            echo "   or: MMSEQS_BIN=/path/to/mmseqs sbatch slurm/20_mmseqs.sbatch"
        } >&2
        exit 1
    fi
    echo "== mmseqs from ${SIF} =="
fi

run_mmseqs() {
    case "$MODE" in
        bin)  "$MMSEQS_BIN" "$@" ;;
        path) mmseqs "$@" ;;
        # env -u LD_PRELOAD: the libstdc++ redirect in the prelude is for the conda
        # env on the host and must not follow us into the container.
        sif)  env -u LD_PRELOAD "$SING" exec --bind "${PFPL_WORK}:${PFPL_WORK}" \
                  "$SIF" mmseqs "$@" ;;
    esac
}

echo "== clustering $(grep -c '^>' "$FASTA") sequences at ${ID} identity, ${COV} coverage =="
run_mmseqs createdb "$FASTA" "$OUT/db"
run_mmseqs cluster "$OUT/db" "$OUT/clu" "$OUT/tmp" \
    --min-seq-id "$ID" -c "$COV" --cov-mode 0 --threads "${SLURM_CPUS_PER_TASK:-16}"
run_mmseqs createtsv "$OUT/db" "$OUT/db" "$OUT/clu" "$OUT/clusters.tsv"

echo "== $(cut -f1 "$OUT/clusters.tsv" | sort -u | wc -l) clusters over $(wc -l < "$OUT/clusters.tsv") sequences =="
