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
# RECLAIM, not `:=`. Slurm on this cluster exports TMPDIR=/tmp into every job, so
# `:=` kept /tmp and every Wave 1 log printed `TMPDIR =/tmp`. The rule now matches
# src/pfpl/paths.py:contain() exactly: a value already inside the root is kept,
# anything outside it is replaced, and PFPL_ALLOW_EXTERNAL=VAR1,VAR2 opts a
# variable out (e.g. a read-only site-wide HF_HOME).
_pfpl_contain() {  # _pfpl_contain VAR default
    local var="$1" def="$2" cur="${!1:-}"
    case ",${PFPL_ALLOW_EXTERNAL:-}," in *",${var},"*) [ -n "$cur" ] && return ;; esac
    case "$cur" in "${PFPL_ROOT}"/*|"${PFPL_ROOT}") return ;; esac
    printf -v "$var" '%s' "$def"
}
if [ "${PFPL_NO_CONTAIN:-0}" != "1" ]; then
    _pfpl_contain HF_HOME               "${PFPL_WORK}/hf"
    _pfpl_contain HUGGINGFACE_HUB_CACHE "${PFPL_WORK}/hf/hub"
    _pfpl_contain TORCH_HOME            "${PFPL_WORK}/torch"
    _pfpl_contain XDG_CACHE_HOME        "${PFPL_WORK}/xdg"
    _pfpl_contain MPLCONFIGDIR          "${PFPL_WORK}/mpl"
    _pfpl_contain TRITON_CACHE_DIR      "${PFPL_WORK}/triton"
    _pfpl_contain CUDA_CACHE_PATH       "${PFPL_WORK}/nv"
    _pfpl_contain NUMBA_CACHE_DIR       "${PFPL_WORK}/numba"
    _pfpl_contain WANDB_DIR             "${PFPL_WORK}/wandb"
    _pfpl_contain WANDB_CACHE_DIR       "${PFPL_WORK}/wandb/cache"
    _pfpl_contain TMPDIR                "${PFPL_WORK}/tmp"
fi
TMP="${TMPDIR}"; TEMP="${TMPDIR}"
export HF_HOME HUGGINGFACE_HUB_CACHE TORCH_HOME XDG_CACHE_HOME MPLCONFIGDIR
export TRITON_CACHE_DIR CUDA_CACHE_PATH NUMBA_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR
export TMPDIR TMP TEMP

# ---- the ONE deliberate exception: the container runtime's scratch -----------
# Singularity/Apptainer unpacks the SIF into a FUSE rootfs under TMPDIR. A FUSE
# mount cannot live on /scratch here - fusermount3 fails with "Permission denied"
# on both mount and cleanup, and the container never starts:
#
#   FATAL: While running host post start tasks: while unmounting fuse directory:
#          .../work/tmp/rootfs-2049131468/root: exit status 1
#
# That is what containing TMPDIR did to the mmseqs job. It needs a LOCAL
# filesystem, and node-local is correct for it anyway: the rootfs is ephemeral,
# it is megabytes, and it is meant to vanish when the job ends. Nothing the
# project produces goes here - TMPDIR itself stays contained, so xtb and every
# Python tempfile still write under the root.
: "${PFPL_CONTAINER_TMPDIR:=${SLURM_TMPDIR:-/tmp}/pfpl-${PFPL_USER}-${SLURM_JOB_ID:-$$}}"
export SINGULARITY_TMPDIR="${PFPL_CONTAINER_TMPDIR}"
export APPTAINER_TMPDIR="${PFPL_CONTAINER_TMPDIR}"
mkdir -p "${PFPL_CONTAINER_TMPDIR}" 2>/dev/null || true
# Leftovers from a killed job would otherwise accumulate on the node.
pfpl_cleanup_container_tmp() {
    [ -n "${PFPL_CONTAINER_TMPDIR:-}" ] && rm -rf "${PFPL_CONTAINER_TMPDIR}" 2>/dev/null
    return 0
}
trap pfpl_cleanup_container_tmp EXIT

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
