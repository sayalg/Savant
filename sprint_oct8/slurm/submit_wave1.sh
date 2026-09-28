#!/bin/bash
# Wave 1: foundations and honest baselines (PLAN.md Section 8, Phases 0-6).
#
# Submitted as one DAG. The shape of it:
#
#     [05 acquire]  <- the only network step; checkpointed, resumable
#          |
#     [10 phase1 harmonise]  <- needs quarantine keys from phase 0 first
#              |
#      +-------+-------+--------------------+
#      |               |                    |
#  [20 mmseqs]   [30 ligand phys]   [40 embed protein]   <- all three concurrent
#      |               |                    |
#      +-------+-------+--------------------+
#              |
#      [45 merge embeddings + phase5 splits]
#              |
#      [50 baselines]  <- GATE G1
#
# The three middle jobs never touch each other, which is the whole reason to use a
# cluster for this rather than a workstation.
#
# NO `cd` ANYWHERE. Submit from the project root, or export PFPL_ROOT.
#
#   ./slurm/submit_wave1.sh
#   DRY=1 ./slurm/submit_wave1.sh              # print, do not submit
#   N_EMBED=11 N_LIGPHYS=19 ./slurm/submit_wave1.sh
#   PDBBIND=/other/dir ./slurm/submit_wave1.sh   # default: work/data/raw/pdbbind
#   CONDA_LIBS_FIRST=1 ./slurm/submit_wave1.sh  # see slurm/_prelude.sh
set -euo pipefail

: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "${PFPL_ROOT}/src/pfpl" ]; then
    echo "run this from the project root, or set PFPL_ROOT=/abs/path" >&2
    exit 1
fi
export PFPL_ROOT
: "${PFPL_WORK:=${PFPL_ROOT}/work}"
export PFPL_WORK
mkdir -p "${PFPL_ROOT}/logs" "${PFPL_WORK}"

DRY="${DRY:-0}"
N_EMBED="${N_EMBED:-11}"        # --array=0-N_EMBED
N_LIGPHYS="${N_LIGPHYS:-19}"
N_GEOM="${N_GEOM:-7}"
PLM="${PLM:-bottleneck}"

sub() {  # sub <description> <sbatch args...>
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

# ---- preflight -------------------------------------------------------------
# Phase 0 runs on the login node: it is quick, interactive, and the one step whose
# output everything else depends on.
if [ ! -f "${PFPL_ROOT}/splits/quarantine.json" ]; then
    cat >&2 <<'EOF'
splits/quarantine.json is missing.

Quarantine keys must exist BEFORE the corpus is built. The previous project's
headline result - per-target rho +0.28 on influenza A neuraminidase - was measured
on a protein that was in its own training fold, with 55 of 81 benchmark ligands
present as training pairs. The number was real; it was not a prediction.

    python scripts/phase0_verify.py --benchmark <your benchmark csv>
EOF
    exit 1
fi

# ---- PDBbind location --------------------------------------------------------
# Defaults to the config's own path. Drop the tarballs you downloaded (P-L.tar,
# index.tar.gz) there and the acquire job unpacks them in place - no PDBBIND needed.
if [ -z "${PDBBIND:-}" ] && [ -d "${PFPL_WORK}/data/raw/pdbbind" ]; then
    PDBBIND="${PFPL_WORK}/data/raw/pdbbind"
    echo "== PDBbind: using ${PDBBIND} ==" >&2
fi

# ---- phase 1a: acquire ------------------------------------------------------
# Skipped only when MANIFEST.json shows EVERY required source present and ChEMBL
# complete. A manifest merely existing is not enough: a run that failed on one
# source still writes it, and the old check then skipped acquisition for good.
# ACQUIRE=force re-runs it anyway (the ChEMBL pager resumes from its checkpoint;
# finished downloads are kept, so a re-run only fetches what is missing).
acquired() {
    python3 - "${PFPL_WORK}/data/raw" <<'PY'
import json, sys, pathlib
raw = pathlib.Path(sys.argv[1])
try:
    m = json.loads((raw / "MANIFEST.json").read_text())
    ck = json.loads((raw / "chembl" / "checkpoint.json").read_text())
except (OSError, ValueError):
    sys.exit(1)
need = ("chembl", "bindingdb", "papyrus", "iuphar", "uniprot")
missing = [s for s in need if s not in m]
if not ck.get("done"):
    missing.append("chembl (partial)")
## PDBbind tarballs dropped in but not unpacked yet: the acquire job stages them.
pb = raw / "pdbbind"
if pb.is_dir() and any(pb.glob("*.tar*")) and not (m.get("pdbbind") or {}).get("staged"):
    missing.append("pdbbind (tarballs not unpacked)")
if missing:
    print("   not yet acquired: " + ", ".join(missing), file=sys.stderr)
sys.exit(1 if missing else 0)
PY
}
ACQ=""
if [ "${ACQUIRE:-auto}" = "force" ] || ! acquired; then
    echo "== phase 1a: acquire raw sources (network; checkpointed) ==" >&2
    ACQ=$(sub "acquire ChEMBL + BindingDB + Papyrus + IUPHAR" \
        --export="ALL,PDBBIND=${PDBBIND:-}" \
        "${PFPL_ROOT}/slurm/05_acquire.sbatch")
else
    echo "== phase 1a: raw sources already acquired (ACQUIRE=force to redo) ==" >&2
fi

# Skipped when the corpus already exists and acquisition is not being re-run, so a
# partial wave can be resumed without another 200 GB harmonisation. PHASE1=force
# rebuilds it. Every stage below is idempotent: finished embedding shards and
# completed ligand-physics shards are skipped, interrupted ones resume.
P1=""
if [ -z "$ACQ" ] && [ -f "${PFPL_WORK}/pairs.parquet" ] && [ "${PHASE1:-auto}" != "force" ]; then
    echo "== phase 1b: corpus exists (work/pairs.parquet); PHASE1=force to rebuild ==" >&2
else
    echo "== phase 1b: harmonise ==" >&2
    P1=$(sub "harmonise the Kd/Ki corpus" \
        ${ACQ:+--dependency=afterok:${ACQ} --kill-on-invalid-dep=yes} \
        --job-name=pfpl-phase1 "${PFPL_ROOT}/slurm/10_phase1.sbatch")
fi
AFTER_P1=${P1:+--dependency=afterok:${P1} --kill-on-invalid-dep=yes}

echo "== embeddings, clustering and ligand physics (concurrent) ==" >&2
MM=$(sub "mmseqs @30% identity" ${AFTER_P1} \
    "${PFPL_ROOT}/slurm/20_mmseqs.sbatch")
EMB=$(sub "ESM per-residue embeddings (${PLM})" ${AFTER_P1} \
    --array="0-${N_EMBED}" --export="ALL,PLM=${PLM}" \
    "${PFPL_ROOT}/slurm/40_embed_protein.sbatch")
LIG=$(sub "ligand physics tier ${TIER:-A}" ${AFTER_P1} \
    --array="0-${N_LIGPHYS}" --export="ALL,TIER=${TIER:-A}" \
    "${PFPL_ROOT}/slurm/30_ligand_phys.sbatch")

DEPS="afterok:${EMB},afterok:${MM}"
# Ligand physics is afterANY: Tier A rarely fails outright, and a partial table
# degrades the triangle penalty rather than invalidating the run. A stalled optional
# stage must not hold up Gate G1.
DEPS="${DEPS},afterany:${LIG}"

if [ -n "${PDBBIND:-}" ]; then
    GEOM=$(sub "geometry labels from ${PDBBIND}" ${AFTER_P1} \
        --array="0-${N_GEOM}" --export="ALL,PDBBIND=${PDBBIND}" \
        "${PFPL_ROOT}/slurm/25_geometry.sbatch")
    DEPS="${DEPS},afterany:${GEOM}"
else
    cat >&2 <<'EOF'

  NOTE: PDBBIND is not set, so no geometry labels will be built.

  That means no distogram anchor. PLAN.md Section 2.1 lists the anchor as a
  REQUIRED mitigation: without it the affinity loss can push probability mass
  toward each pair's energy minimum and converge on an over-attractive,
  geometrically impossible solution that scores well on pK. Wave 1 and the
  baselines are unaffected - but a Wave 2 physics claim is not supportable.

      PDBBIND=/scratch/$USER/pdbbind_v2020 ./slurm/submit_wave1.sh

EOF
fi

echo "== merge + splits ==" >&2
MERGE=$(sub "merge caches, build and freeze the splits" \
    --dependency="${DEPS}" --kill-on-invalid-dep=yes \
    --partition=Orion --time=04:00:00 --mem=128G --cpus-per-task=8 \
    --job-name=pfpl-merge --output=logs/merge_%j.out --error=logs/merge_%j.err \
    --export="ALL,PLM=${PLM}" \
    --wrap "set -euo pipefail
            source \"${PFPL_ROOT}/slurm/_prelude.sh\"
            python \"${PFPL_ROOT}/scripts/phase4_embed.py\" --merge -o plm=${PLM}
            python \"${PFPL_ROOT}/scripts/phase4_embed.py\" --verify -o plm=${PLM} || \
              echo 'WARNING: cache verification failed - see the Phase 4 gate'
            python \"${PFPL_ROOT}/scripts/phase3_ligand_phys.py\" --merge || \
              echo 'ligand physics merge skipped (optional)'
            python \"${PFPL_ROOT}/scripts/phase2_geometry.py\" --merge || \
              echo 'no geometry shards - Stage 1 and the anchor are unavailable'
            python \"${PFPL_ROOT}/scripts/phase5_splits.py\"")

echo "== baselines (GATE G1) ==" >&2
BASE=$(sub "Phase 6 baselines" \
    --dependency="afterok:${MERGE}" --kill-on-invalid-dep=yes \
    --export="ALL,PLM=${PLM}" "${PFPL_ROOT}/slurm/50_baselines.sbatch")

cat >&2 <<EOF

submitted. watch with:
  squeue -u \$USER -o '%.10i %.24j %.9T %.10M %.6D %R'
  tail -f ${PFPL_ROOT}/logs/train_*.out

if it goes quiet without producing output:
  python ${PFPL_ROOT}/scripts/status.py

key artifacts as they land:
  work/pairs.parquet              the corpus, one row per (protein, ligand)
  work/mmseqs/clusters.tsv        the honest split's input
  work/cache/plm_${PLM}_*/        per-residue embeddings (ragged fp16 memmap)
  splits/*.npz + splits/README.md frozen splits with their overlap statistics
  work/runs/baselines_*/          GATE G1 - read the ligand_only row FIRST
  reports/phase_1.md, phase_6.md

GATE G1 (job ${BASE}): does anything beat ligand_only?
  If not, that is the result, and Wave 2 does not fix it - Section 5.3.5 says
  tighten the split before building anything.
EOF
