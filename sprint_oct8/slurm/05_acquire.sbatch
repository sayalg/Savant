#!/bin/bash
#SBATCH --job-name=pfpl-acquire
#SBATCH --output=logs/acquire_%j.out
#SBATCH --error=logs/acquire_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Phase 1a: acquire the raw Kd/Ki sources. The ONLY step that touches the network.
#
# Long and I/O-bound, not CPU-bound: 2 cores and 16 GB are plenty, and the 24 h
# wall time is for the ChEMBL pager (~700k activities at a polite request rate).
# It is CHECKPOINTED - a wall-time kill resumes from the last 20k-row shard rather
# than starting over, so re-submitting this job is always safe.
#
# Separate from harmonisation on purpose: this needs no RDKit (so it cannot die on
# the NumPy ABI break), while Phase 1b needs RDKit and ~200 GB of RAM. Mixing them
# wastes an allocation either way.
#
#   sbatch slurm/05_acquire.sbatch                    # everything
#   sbatch slurm/05_acquire.sbatch --chembl           # just resume ChEMBL
#   PDBBIND=/scratch/$USER/pdbbind_v2020 sbatch slurm/05_acquire.sbatch
#
# PDBbind is licensed and is never downloaded - PDBBIND only points the check at a
# copy you staged yourself.
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

if [ $# -eq 0 ]; then
    set -- --all
fi
pfpl_run phase1a_acquire.py "$@" ${PDBBIND:+--pdbbind-dir "$PDBBIND"}
pfpl_run phase1a_acquire.py --status
