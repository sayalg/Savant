#!/bin/bash
#SBATCH --job-name=bench-boltz
#SBATCH --output=logs/boltz_%A_%a.out
#SBATCH --error=logs/boltz_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=FP32
#SBATCH --array=0-25
#SBATCH --partition=GPU
# The GPU partition is mixed hardware. Recent torch wheels carry no Volta
# (sm_70) kernels, so a task landing on a V100 dies before it starts; Boltz also
# wants bf16, which Volta lacks. Pin the type to Ampere or newer - list what the
# partition has with:  sinfo -p GPU -o '%20N %25f %15G'
# #SBATCH --gres=gpu:A40:1
# --------------------------------------------------------
set -euo pipefail

# Stage 1 of 3: predict every complex with Boltz-2.
#
# One array task per YAML shard from 01_prepare_inputs.py, which prints the
# exact --array range to use; the task id is the shard directory name. That
# range is NOT --num-shards: step 01 adds extra shards to hold the ligands too
# large for the affinity head, which cannot share a directory with the rest.
#
# Resumable: `boltz predict` skips inputs that already have a prediction, so a
# requeued or timed-out task picks up where it stopped. Re-run the same sbatch to
# fill gaps. The prune step below is what makes that safe - boltz's own skip test
# is satisfied by a directory that merely exists, half-written or not.
#
#   sbatch --array=0-25 sbatch/run_boltz.sbatch
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

# A shard must be homogeneous in affinity. Boltz enters its affinity stage if ANY
# record in the directory requests affinity, then iterates EVERY record in it and
# loads a pre_affinity_<id>.npz that only affinity records are given - so one
# oversized ligand among 39 normal ones kills the task partway through with
# FileNotFoundError. 01_prepare_inputs.py separates them; this catches a stale or
# hand-edited shard before the allocation is spent.
N_YAML=$(find "${IN_DIR}" -maxdepth 1 -name '*.yaml' | wc -l)
N_AFFINITY=$(grep -l '^properties:' "${IN_DIR}"/*.yaml 2>/dev/null | wc -l)
if [ "${N_AFFINITY}" -ne 0 ] && [ "${N_AFFINITY}" -ne "${N_YAML}" ]; then
    echo "${IN_DIR} mixes affinity and non-affinity inputs (${N_AFFINITY} of ${N_YAML})." >&2
    echo "Re-run: python scripts/01_prepare_inputs.py --num-shards <N>" >&2
    exit 1
fi

# Clear out prediction dirs a killed run left half-written. boltz skips a record
# whose predictions/<id>/ merely EXISTS, but queues affinity unless
# affinity_<id>.json exists - so a half-finished dir makes the affinity stage
# look for a pre_affinity_<id>.npz the skipped structure stage never wrote, and
# that FileNotFoundError kills the whole task.
python scripts/prune_partial_boltz.py --shard "${SHARD}"

echo "== boltz predict: ${IN_DIR} -> ${OUT_DIR} ($(ls "${IN_DIR}" | wc -l) inputs) =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Touch the GPU through torch before boltz does. Both ways this can fail - a
# torch built for a newer CUDA than the driver, or a GPU whose compute
# capability this torch has no kernels for - otherwise surface only once
# Lightning sets up the device, minutes into the allocation, with a traceback
# that points at Lightning rather than at the real problem.
python - <<'PY' || exit 1
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("torch cannot see a GPU on this node (--gres=gpu:1 not honoured?)")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()

try:
    torch.zeros(1, device="cuda").add_(1)
except RuntimeError as exc:
    lines = [f"CUDA unusable on {name} (compute capability {major}.{minor}): {exc}", ""]
    if f"sm_{major}{minor}" not in arch_list:
        # Mixed-hardware partitions hand out whatever is free; recent torch
        # wheels no longer carry Volta (sm_70) kernels.
        lines += [
            f"torch {torch.__version__} ships kernels for {', '.join(arch_list)} - "
            f"nothing for sm_{major}{minor}. This is the GPU, not the driver.",
            "Request a GPU torch supports, e.g.:",
            "  sbatch --gres=gpu:A40:1 --array=... sbatch/run_boltz.sbatch",
            "See what the partition offers with:  sinfo -p GPU -o '%20N %25f %15G'",
        ]
    else:
        lines += [
            f"torch {torch.__version__} is built against CUDA {torch.version.cuda}; "
            "if the driver is older, reinstall torch for the driver's CUDA:",
            f"  pip install --force-reinstall --no-deps "
            f"torch=={torch.__version__.split('+')[0]} "
            "--index-url https://download.pytorch.org/whl/cu128",
        ]
    sys.exit("\n".join(lines))

print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {name} "
      f"(sm_{major}{minor}), bf16 supported: {torch.cuda.is_bf16_supported()}")
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
