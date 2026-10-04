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

# Print WHO WE ARE as early as possible. Everything below this line can fail, and a
# failure that happens before the first echo is indistinguishable from the script
# never having run - which is exactly how a bad ldconfig probe turned
# `./slurm/submit_wave2.sh` into a silent no-op.
echo "== $(date '+%F %T')  host $(hostname)  job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:-?}} =="

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
# AUTOMATIC as of 2026-10-04, not opt-in. It had to be remembered at every submit,
# and it was forgotten - optree's _C.so brought the Wave 2 selftest down with
# exactly this error. Deciding it by comparing versions is strictly better than
# asking a human to remember a flag, and it is a no-op when the node's library is
# already new enough.
#
#     CONDA_LIBS_FIRST=0   off
#     CONDA_LIBS_FIRST=2   broader, riskier (prepends $CONDA_PREFIX/lib)
#     (unset / auto)       preload conda's libstdc++ only when it is NEWER
# This whole block runs in a SUBSHELL with `set +e`, and its result is read back as
# text. An environment probe must never be able to take the job down with it: the
# first version called `/sbin/ldconfig` directly, which is exit 127 where ldconfig
# lives in /usr/sbin, and under the `set -euo pipefail` every caller sets that
# killed the shell - before this file prints anything, so submit_wave2.sh produced
# no output whatsoever. Hence also the early header above: silence must be
# impossible.
_pfpl_glibcxx_max() {  # highest GLIBCXX_3.4.N a libstdc++ exports, or 0
    local so="$1" v
    [ -r "$so" ] || { echo 0; return 0; }
    v=$( { strings -a "$so" 2>/dev/null || tr -c '[:print:]' '\n' <"$so" 2>/dev/null; } \
         | sed -n 's/^GLIBCXX_3\.4\.\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1 )
    case "$v" in (*[!0-9]*|"") echo 0 ;; (*) echo "$v" ;; esac
    return 0
}
_pfpl_pick_libstdcxx() {  # -> "<path> <conda_ver> <sys_ver>", or nothing
    set +e
    local conda_so="${CONDA_PREFIX}/lib/libstdc++.so.6" sys_so cv sv
    [ -r "$conda_so" ] || return 0
    ## LD_PRELOAD is only meaningful as an absolute POSIX path. Anything else (a
    ## Windows-style C:/... under Git Bash, a relative path) makes the dynamic
    ## loader fail on every subsequent fork, which is far worse than not preloading.
    case "$conda_so" in (/*) ;; (*) return 0 ;; esac
    sys_so=$( { command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null; } \
              | awk '/libstdc\+\+\.so\.6 /{print $NF; exit}' )
    for c in "$sys_so" /lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6; do
        [ -n "$c" ] && [ -r "$c" ] && { sys_so="$c"; break; }
    done
    cv=$(_pfpl_glibcxx_max "$conda_so")
    sv=$(_pfpl_glibcxx_max "${sys_so:-}")
    ## Both must be real numbers and conda's strictly newer, or we do nothing.
    [ "$cv" -gt 0 ] 2>/dev/null || return 0
    [ "$cv" -gt "$sv" ] 2>/dev/null || return 0
    echo "$conda_so $cv $sv"
}
if [ -n "${CONDA_PREFIX:-}" ]; then
    case "${CONDA_LIBS_FIRST:-auto}" in
        0) : ;;
        2) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" ;;
        *) _pfpl_pick=$( _pfpl_pick_libstdcxx ) || _pfpl_pick=""
           if [ -n "${_pfpl_pick:-}" ]; then
               ## `read`, NOT `set --`. This file is SOURCED, so `set --` would
               ## overwrite the caller's positional parameters - and eight sbatch
               ## scripts pass "$@" through to their python script afterwards, so
               ## `sbatch slurm/05_acquire.sbatch --chembl` would have become
               ## `phase1a_acquire.py <libstdc++ path> 35 30`.
               read -r _pfpl_so _pfpl_cv _pfpl_sv <<<"${_pfpl_pick}" || true
               export LD_PRELOAD="${_pfpl_so}${LD_PRELOAD:+:$LD_PRELOAD}"
               echo "   libstdc++  preloading conda's (GLIBCXX_3.4.${_pfpl_cv}" \
                    "> node's 3.4.${_pfpl_sv})"
           fi ;;
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
