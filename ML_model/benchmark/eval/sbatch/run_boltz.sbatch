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
mkdir -p logs work/boltz

SHARD=$(printf "shard_%02d" "${SLURM_ARRAY_TASK_ID:-0}")
IN_DIR="work/yaml/${SHARD}"
OUT_DIR="work/boltz/${SHARD}"

if [ ! -d "${IN_DIR}" ]; then
    echo "No such shard: ${IN_DIR} (did 01_prepare_inputs.py run with the same --num-shards?)" >&2
    exit 1
fi

# Check the MSAs the YAMLs point at before boltz burns a GPU allocation on an
# input it cannot load. Every `msa:` line in the shard is resolved (one sed over
# the shard, then one stat per distinct path).
N_REF=0
N_MISSING=0
MISSING_EXAMPLE=""
while IFS= read -r msa; do
    [ -n "${msa}" ] || continue
    N_REF=$((N_REF + 1))
    if [ ! -s "${msa}" ]; then
        N_MISSING=$((N_MISSING + 1))
        [ -n "${MISSING_EXAMPLE}" ] || MISSING_EXAMPLE="${msa}"
    fi
done <<EOF
$(sed -n "s/^[[:space:]]*msa:[[:space:]]*'\(.*\)'[[:space:]]*\$/\1/p" "${IN_DIR}"/*.yaml \
  2>/dev/null | sort -u)
EOF

if [ "${N_MISSING}" -gt 0 ]; then
    echo "${N_MISSING} of ${N_REF} referenced MSAs are missing or empty, e.g. ${MISSING_EXAMPLE}" >&2
    if [ "${N_MISSING}" -eq "${N_REF}" ]; then
        echo "None are present - run 02_fetch_msas.py first, or re-run 01 with --msa-mode server." >&2
        exit 1
    fi
    echo "Continuing: boltz will fail on those inputs only. Re-run 02_fetch_msas.py "\
         "to fill the gaps, then re-submit this shard to pick them up." >&2
fi

echo "== boltz predict: ${IN_DIR} -> ${OUT_DIR} ($(ls "${IN_DIR}" | wc -l) inputs) =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Touch the GPU through torch before boltz does. A torch built against a newer
# CUDA than the node's driver supports only fails once Lightning sets up the
# device - after model load, minutes into the allocation - and the traceback
# points at Lightning rather than at the real problem.
python - <<'PY' || exit 1
import sys
import torch

try:
    torch.zeros(1, device="cuda")
except RuntimeError as exc:
    sys.exit(
        f"CUDA unusable: {exc}\n\n"
        f"torch {torch.__version__} is built against CUDA {torch.version.cuda}. "
        "If the driver is older than that, reinstall torch for the driver's CUDA "
        "version, e.g. for a 12.8 driver:\n"
        f"  pip install --force-reinstall --no-deps torch=={torch.__version__.split('+')[0]} "
        "--index-url https://download.pytorch.org/whl/cu128"
    )
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
PY

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
