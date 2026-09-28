# Shared prelude. Sourced by every sbatch script in this directory.
#
# THERE IS NO `cd` ANYWHERE IN THIS PROJECT'S SLURM SCRIPTS.
#
# The old scripts opened with `cd "$(dirname "$0")/.."`, which breaks whenever $0 is
# not the on-disk path of the script: Slurm copies a --wrap body to a spool
# directory, a symlinked launcher resolves to the link, a container bind-mount
# rewrites the prefix, and `sbatch` invoked through a relative path from elsewhere
# lands somewhere else entirely. Each of those produced a job that ran in the wrong
# directory and wrote its artifacts where nothing would look for them.
#
# Instead: PFPL_ROOT is resolved once, everything uses absolute paths under it, and
# the Python side re-derives the same root independently from the package location
# (src/pfpl/paths.py), so the two cannot disagree.
#
# Slurm already starts a job in the directory `sbatch` was run from, which is why no
# `cd` was needed in the first place.
#
# Usage at the top of an sbatch script, after the #SBATCH block:
#
#     : "${PFPL_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
#     source "${PFPL_ROOT}/slurm/_prelude.sh"
#
# If you submit from somewhere other than the project root, set it explicitly:
#
#     PFPL_ROOT=/scratch/$USER/Savant/sprint_oct8 sbatch slurm/40_embed_protein.sbatch

set -euo pipefail

# The canonical install. Everything this project reads or writes lives under it.
#
# `${USER:-...}` rather than `${USER}`: this file runs under `set -u`, and a Slurm
# job started from a minimal environment (or a container) may have no USER at all,
# which would abort the prelude before it printed anything useful.
PFPL_USER="${USER:-${SLURM_JOB_USER:-$(id -un 2>/dev/null || echo unknown)}}"
PFPL_DEFAULT_ROOT="/scratch/${PFPL_USER}/Savant/sprint_oct8"

if [ -z "${PFPL_ROOT:-}" ]; then
    for cand in "${SLURM_SUBMIT_DIR:-}" "${PFPL_DEFAULT_ROOT}" "$PWD"; do
        if [ -n "$cand" ] && [ -d "${cand}/src/pfpl" ]; then
            PFPL_ROOT="$cand"
            break
        fi
    done
fi
: "${PFPL_ROOT:=${PFPL_DEFAULT_ROOT}}"
export PFPL_ROOT

if [ ! -d "${PFPL_ROOT}/src/pfpl" ]; then
    echo "PFPL_ROOT=${PFPL_ROOT} does not look like the project root" >&2
    echo "  (expected ${PFPL_ROOT}/src/pfpl to exist)" >&2
    echo "" >&2
    echo "  This project expects to live entirely at:" >&2
    echo "      ${PFPL_DEFAULT_ROOT}" >&2
    echo "  Submit from there, or set it explicitly:" >&2
    echo "      PFPL_ROOT=/abs/path/to/sprint_oct8 sbatch slurm/<script>.sbatch" >&2
    exit 1
fi

export PYTHONPATH="${PFPL_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Everything under one root. The env vars exist so a single directory can be moved
# if a filesystem fills up, but the defaults are all inside PFPL_ROOT so the project
# is self-contained with nothing set.
: "${PFPL_WORK:=${PFPL_ROOT}/work}"
export PFPL_WORK
export PFPL_LOGS="${PFPL_LOGS:-${PFPL_ROOT}/logs}"

# ------------------------------------------------------- containment
# Redirect every cache a dependency would otherwise put in $HOME, plus TMPDIR.
#
# This is not tidiness. HPC $HOME is small and quota'd, and /tmp on a compute node
# is node-local and reclaimed when the job ends - so an unredirected HuggingFace or
# Triton cache is how a job dies with "Disk quota exceeded" after six hours of
# embedding extraction, and an unredirected TMPDIR is how an xtb shard silently
# loses its scratch files mid-array.
#
# `:=` throughout, so a value you exported yourself always wins.
# src/pfpl/paths.py:contain() applies the identical map on the Python side, and
# `python -m pfpl.paths` exits non-zero if anything still points outside the root.
: "${HF_HOME:=${PFPL_WORK}/hf}"
: "${HUGGINGFACE_HUB_CACHE:=${PFPL_WORK}/hf/hub}"
: "${TORCH_HOME:=${PFPL_WORK}/torch}"
: "${XDG_CACHE_HOME:=${PFPL_WORK}/xdg}"
: "${MPLCONFIGDIR:=${PFPL_WORK}/mpl}"
: "${TRITON_CACHE_DIR:=${PFPL_WORK}/triton}"
: "${CUDA_CACHE_PATH:=${PFPL_WORK}/nv}"
: "${NUMBA_CACHE_DIR:=${PFPL_WORK}/numba}"
: "${WANDB_DIR:=${PFPL_WORK}/wandb}"
: "${WANDB_CACHE_DIR:=${PFPL_WORK}/wandb/cache}"
: "${TMPDIR:=${PFPL_WORK}/tmp}"
TMP="${TMPDIR}"; TEMP="${TMPDIR}"
export HF_HOME HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR
export TRITON_CACHE_DIR CUDA_CACHE_PATH NUMBA_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR
export TMPDIR TMP TEMP

mkdir -p "${PFPL_LOGS}" "${PFPL_WORK}" "${PFPL_ROOT}/reports" "${PFPL_ROOT}/splits" \
         "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" \
         "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${NUMBA_CACHE_DIR}" \
         "${WANDB_DIR}" "${TMPDIR}"

# ---------------------------------------------------------------- library order
# Opt-in: make the conda env's libstdc++ win over the node's. Needed when a compiled
# extension was built with a newer GCC than /lib64/libstdc++.so.6 provides (symptom:
# "version `GLIBCXX_3.4.31' not found"). The cause is that conda links with
# DT_RUNPATH which - unlike DT_RPATH - is NOT inherited by dynamically loaded
# extension modules, so the .so falls through to the ldconfig cache and finds /lib64.
#
# On this cluster the env already had GLIBCXX_3.4.35 while the error asked for
# 3.4.31: nothing was missing, the load ORDER was wrong. LD_PRELOAD is the surgical
# fix - it redirects only libstdc++ and cannot shadow the CUDA driver libraries the
# way prepending $CONDA_PREFIX/lib to LD_LIBRARY_PATH can.
#
#     CONDA_LIBS_FIRST=1 ./slurm/submit_wave1.sh     # preload libstdc++ only
#     CONDA_LIBS_FIRST=2 ./slurm/submit_wave1.sh     # broader, riskier
if [ -n "${CONDA_PREFIX:-}" ]; then
    case "${CONDA_LIBS_FIRST:-0}" in
        1) [ -e "${CONDA_PREFIX}/lib/libstdc++.so.6" ] && \
             export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" ;;
        2) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" ;;
    esac
fi

# ------------------------------------------------------------------- HuggingFace
# Compute nodes have outbound network, so a cache miss just downloads. HF_HOME still
# points inside the project: a cold array would otherwise have every task pull the
# same multi-GB checkpoint simultaneously, which is slower than one staged copy and
# invites Hub rate-limiting. HF_OFFLINE=1 pins a run to the cache for reproducibility.
export TOKENIZERS_PARALLELISM=false
if [ "${HF_OFFLINE:-0}" = "1" ]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

echo "== $(date '+%F %T')  host $(hostname)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-?}} =="
echo "   PFPL_ROOT=${PFPL_ROOT}"
echo "   PFPL_WORK=${PFPL_WORK}"
echo "   TMPDIR   =${TMPDIR}"
echo "   python   $(command -v python)"

# Fail the job here rather than after six hours if anything still escapes the root.
# PFPL_SKIP_CONTAIN_CHECK=1 to bypass (it is a one-second import).
if [ "${PFPL_SKIP_CONTAIN_CHECK:-0}" != "1" ]; then
    python -m pfpl.paths >/dev/null || {
        echo "CONTAINMENT CHECK FAILED - see above. Re-run 'python -m pfpl.paths'." >&2
        exit 1
    }
fi

# Run a project script by its name under $PFPL_ROOT/scripts, from anywhere.
pfpl_run() {
    local s="$1"; shift
    echo "-- python ${PFPL_ROOT}/scripts/${s} $*"
    python "${PFPL_ROOT}/scripts/${s}" "$@"
}
