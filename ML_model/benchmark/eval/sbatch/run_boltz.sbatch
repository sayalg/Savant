#!/bin/bash
#SBATCH --job-name=bench-boltz
#SBATCH --output=logs/boltz_%A_%a.out
#SBATCH --error=logs/boltz_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --partition=GPU
# --------------------------------------------------------
set -euo pipefail

# Stage 1 of 3: predict every complex with Boltz-2.
#
# One array task per YAML shard from 01_prepare_inputs.py. --array MUST be
# 0-(num_shards-1) and match the --num-shards used there; the task id is the
# shard directory name.
#
# Resume is free: `boltz predict` skips any input that already has a prediction
# unless --override is passed, so a requeued or timed-out task picks up where it
# stopped. Re-run the same sbatch to fill gaps.
#
#   sbatch --array=0-24 sbatch/run_boltz.sbatch
#   sbatch --array=7,13 sbatch/run_boltz.sbatch     # just those shards

export BOLTZ_CACHE="${BOLTZ_CACHE:-$HOME/.boltz}"
# --------------------------------------------------------

cd "$(dirname "$0")/.."          # ML_model/benchmark/eval
mkdir -p logs work/boltz

SHARD=$(printf "shard_%02d" "${SLURM_ARRAY_TASK_ID:-0}")
IN_DIR="work/yaml/${SHARD}"
OUT_DIR="work/boltz/${SHARD}"

if [ ! -d "${IN_DIR}" ]; then
    echo "No such shard: ${IN_DIR} (did 01_prepare_inputs.py run with the same --num-shards?)" >&2
    exit 1
fi

# Fail early and loudly if the YAMLs point at MSAs that were never fetched -
# otherwise boltz burns a GPU allocation before erroring on the first input.
FIRST_MSA=$(grep -h -m1 '^\s*msa:' "${IN_DIR}"/*.yaml 2>/dev/null | sed "s/.*msa: *'\(.*\)'/\1/" || true)
if [ -n "${FIRST_MSA}" ] && [ ! -s "${FIRST_MSA}" ]; then
    echo "YAMLs reference ${FIRST_MSA}, which is missing or empty." >&2
    echo "Run 02_fetch_msas.py first, or re-run 01 with --msa-mode server." >&2
    exit 1
fi

echo "== boltz predict: ${IN_DIR} -> ${OUT_DIR} ($(ls "${IN_DIR}" | wc -l) inputs) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --output_format pdb: HADDOCK3 (inside Savant) needs PDB, and PRODIGY-LIG is
#   happier with it. The chain ids and the HETATM/LIG ligand residue that both
#   tools depend on come straight from the YAML written in step 01.
# --diffusion_samples 1: one pose per pair is what the downstream tools score.
#   Affinity still ensembles internally (--diffusion_samples_affinity, default 5).
# BOLTZ_EXTRA_ARGS: e.g. "--use_msa_server" if step 01 ran with --msa-mode server,
#   or "--use_potentials" for inference-time steering.
boltz predict "${IN_DIR}" \
    --out_dir "${OUT_DIR}" \
    --output_format pdb \
    --accelerator gpu \
    --devices 1 \
    --diffusion_samples 1 \
    --recycling_steps 3 \
    --num_workers 2 \
    --preprocessing-threads "${SLURM_CPUS_PER_TASK:-8}" \
    ${BOLTZ_EXTRA_ARGS:-}

echo "== ${SHARD} done: $(find "${OUT_DIR}" -name '*_model_0.pdb' | wc -l) structures =="
echo "Next: python scripts/03_collect_boltz.py"
