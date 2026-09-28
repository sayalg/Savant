#!/bin/bash
#SBATCH --job-name=pfpl-baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --partition=Orion
# ---------------------------------------------------------------------------
# Phase 6 / Gate G1. The most important job in Wave 1 and the cheapest.
#
# "The physics work may not be judged against numbers that do not exist yet."
#
# On the previous corpus a ligand-only model - ECFP plus 20 RDKit descriptors, no
# protein input at all - reached median per-target rho +0.325 on 613 held-out
# protein clusters while the best 7,680-dimension ESM model reached 0.158. Run this
# BEFORE any training job, not after.
: "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PFPL_ROOT}/slurm/_prelude.sh"

pfpl_run phase6_baselines.py -o "plm=${PLM:-bottleneck}" "$@"
