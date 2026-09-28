#!/bin/bash
#SBATCH --job-name=pfpl-ligphys
#SBATCH --output=logs/ligphys_%A_%a.out
#SBATCH --error=logs/ligphys_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Phase 3: ligand physics, Tiers A-C. CPU array. Tier A is ~1 s/ligand on ONE core
# (the 16-conformer ETKDG + MMFF ensemble is the cost, not the descriptors); RDKit
# is now told to use every CPU in the allocation, which is why this asks for 8.
#
# CHECKPOINTED every 500 ligands. A task that hits the wall time resumes where it
# stopped when the array is resubmitted with the SAME --array range.
#
# Run the selfcheck FIRST and read its projection before trusting --time:
#   python scripts/phase3_ligand_phys.py --selfcheck --tier ABC
#
# The conformer ensemble computed here is what bounds the triangle penalty. Without
# it Variant A has no geometric self-consistency signal at all.
#
#   sbatch --array=0-19 slurm/30_ligand_phys.sbatch
#   TIER=ABC sbatch --array=0-39 slurm/30_ligand_phys.sbatch
#   python scripts/phase3_ligand_phys.py --merge
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

pfpl_run phase3_ligand_phys.py --tier "${TIER:-A}" --n-conf "${N_CONF:-16}" "$@"
