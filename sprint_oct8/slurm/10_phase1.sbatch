#!/bin/bash
#SBATCH --job-name=pfpl-phase1
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Phase 1: harvest and harmonise the affinity corpus.
#
# CPU only, but memory-hungry: RDKit standardisation of ~1.8M rows plus the
# cross-source dedupe keeps several copies of the frame alive. If it OOMs, raise
# --mem before reaching for chunking - the dedupe needs the whole table at once.
#
# Compute nodes have network, so the papyrus and chembl loaders can fetch here.
#
#   sbatch slurm/10_phase1.sbatch
#   sbatch slurm/10_phase1.sbatch -o data.sources.papyrus.enabled=true
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

# One variable drives BOTH the labels (here) and the geometry (25_geometry.sbatch).
# Without this the config's default path and $PDBBIND could diverge: you would stage
# a licensed copy, get geometry from it, and silently get no crystal-linked LABELS
# because Phase 1 was still looking in work/data/raw/pdbbind.
pfpl_run phase1_harmonize.py \
    ${PDBBIND:+-o "data.sources.pdbbind.path=${PDBBIND}"} "$@"
