"""Where everything lives. One root, nothing outside it, no dependence on the cwd.

**Everything this project reads or writes lives under `PFPL_ROOT`**, which on this
cluster is `/scratch/$USER/Savant/sprint_oct8`. That is a hard requirement, not a
convention: HPC `$HOME` is small and quota'd, `/tmp` on a compute node is node-local
and vanishes when the job ends, and a cache that silently lands in either is how a
job dies at 3am with "Disk quota exceeded" after six hours of embedding extraction.

Two separate problems are solved here.

**1. Finding the root without a `cd`.** Every sbatch script used to start with
`cd "$(dirname "$0")/.."`, which breaks whenever `$0` is not the script's on-disk
path - Slurm copies a `--wrap` body to a spool directory, a symlinked launcher
resolves to the link, a container bind-mount rewrites the prefix. The root is
resolved here instead, in this order:

  1. $PFPL_ROOT                     explicit, set by slurm/_prelude.sh
  2. $SLURM_SUBMIT_DIR              where `sbatch` was run
  3. the package's own location     src/pfpl/paths.py -> ../../..
  4. /scratch/$USER/Savant/sprint_oct8   the canonical install
  5. the current directory          last resort, interactive only

(3) is what makes it reliable: an importable `pfpl` always knows where its own
source tree is, whatever the cwd happens to be.

**2. Keeping third-party libraries inside it.** `contain()` redirects every
HOME-bound cache a dependency would otherwise create - HuggingFace, torch.hub,
matplotlib, Triton, CUDA JIT, Numba, W&B - plus `TMPDIR`, which is what `xtb` and
every `tempfile` call in `features/ligand_phys.py` inherit. It runs on import with
setdefault semantics, so an explicit value you set yourself always wins, and
`PFPL_NO_CONTAIN=1` disables it entirely.
"""
from __future__ import annotations

import os
from pathlib import Path

## The canonical install on this cluster. Used as a fallback when the package
## location does not look like a project root (a stray `pip install -e` elsewhere,
## or a container whose bind-mount hides the source tree).
CANONICAL = "/scratch/$USER/Savant/sprint_oct8"


def _looks_like_root(p: Path) -> bool:
    return (p / "src" / "pfpl").is_dir() and (p / "configs").is_dir()


def _resolve_root() -> tuple[Path, str]:
    for name in ("PFPL_ROOT", "SLURM_SUBMIT_DIR"):
        cand = os.environ.get(name)
        if cand:
            p = Path(os.path.expandvars(cand)).expanduser().resolve()
            if _looks_like_root(p):
                return p, name
    ## src/pfpl/paths.py -> src/pfpl -> src -> <root>
    pkg_root = Path(__file__).resolve().parents[2]
    if _looks_like_root(pkg_root):
        return pkg_root, "package location"
    canon = Path(os.path.expandvars(CANONICAL)).expanduser()
    if _looks_like_root(canon):
        return canon.resolve(), "canonical scratch path"
    cwd = Path.cwd().resolve()
    if _looks_like_root(cwd):
        return cwd, "cwd"
    print(f"WARNING: could not identify the pfpl root; using {pkg_root}. "
          f"Set PFPL_ROOT to silence this.", flush=True)
    return pkg_root, "fallback"


ROOT, ROOT_SOURCE = _resolve_root()

## Small, version-controlled, and deliberately under ROOT rather than under WORK:
## PLAN.md Section 5.3 requires the splits to be frozen, and every number in the
## paper is measured against a specific one.
CONFIGS = ROOT / "configs"
SWEEPS = ROOT / "sweeps"
SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports"
TESTS = ROOT / "tests"

## Bulk artifacts. The env vars exist so a single directory can be moved if a
## filesystem fills up; the DEFAULTS all sit under ROOT so the project is
## self-contained with nothing set.
WORK = Path(os.environ.get("PFPL_WORK") or ROOT / "work").expanduser().resolve()
DATA = Path(os.environ.get("PFPL_DATA") or WORK / "data").expanduser().resolve()
CACHE = Path(os.environ.get("PFPL_CACHE") or WORK / "cache").expanduser().resolve()
RUNS = Path(os.environ.get("PFPL_RUNS") or WORK / "runs").expanduser().resolve()
LOGS = Path(os.environ.get("PFPL_LOGS") or ROOT / "logs").expanduser().resolve()

RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

## Scratch space for subprocesses. `xtb` writes its scratch files here (Tier B/C
## ligand physics shells out), and so does every `tempfile` call, because contain()
## points TMPDIR at it. On a compute node the system /tmp is node-local and is
## reclaimed when the job ends, which silently loses a half-finished shard.
TMP = WORK / "tmp"
HF = WORK / "hf"

_DIRS = (WORK, DATA, CACHE, RUNS, LOGS, RAW, INTERIM, PROCESSED, SPLITS, REPORTS,
         TMP, HF)

## Every cache a dependency creates in $HOME if left alone. Value is a subdirectory
## of WORK. TMPDIR/TMP/TEMP are listed for all three spellings because Python's
## tempfile checks TMPDIR, TEMP and TMP in that order, and a subprocess may read
## any of them.
_CONTAINED = {
    "HF_HOME": "hf",                  # transformers + datasets + hub
    "HUGGINGFACE_HUB_CACHE": "hf/hub",
    "TORCH_HOME": "torch",            # torch.hub checkpoints
    "XDG_CACHE_HOME": "xdg",          # catches many well-behaved libraries
    "MPLCONFIGDIR": "mpl",            # matplotlib font cache
    "TRITON_CACHE_DIR": "triton",     # torch.compile / Triton kernels
    "CUDA_CACHE_PATH": "nv",          # CUDA JIT
    "NUMBA_CACHE_DIR": "numba",
    "WANDB_DIR": "wandb",             # PLAN.md Section 7.1 asks for W&B offline
    "WANDB_CACHE_DIR": "wandb/cache",
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
}


def _inside(p) -> bool:
    try:
        Path(p).expanduser().resolve().relative_to(ROOT)
        return True
    except (ValueError, OSError):
        return False


def allowed_external() -> set[str]:
    """Vars explicitly permitted to point outside, e.g. a site-wide model cache:

        PFPL_ALLOW_EXTERNAL=HF_HOME,HUGGINGFACE_HUB_CACHE
    """
    return {v.strip().upper()
            for v in (os.environ.get("PFPL_ALLOW_EXTERNAL") or "").split(",")
            if v.strip()}


def contain(create: bool = False) -> dict:
    """Point every HOME-bound cache under WORK. Returns {var: path} for what changed.

    A var is set when unset, and RECLAIMED when its current value lies outside ROOT.
    Reclaiming matters: Windows and many login shells pre-set `TMP`/`TEMP`, so pure
    setdefault semantics would leave a hole exactly where it hurts - `tempfile`
    reads TMPDIR, TEMP, TMP in that order and a subprocess may read any of them.

    A value that already points inside ROOT is left alone, so a deliberate
    rearrangement within the project still wins. To keep something genuinely
    outside - a read-only shared model cache is the usual case - name it in
    `PFPL_ALLOW_EXTERNAL`. `PFPL_NO_CONTAIN=1` disables all of this.
    """
    if os.environ.get("PFPL_NO_CONTAIN") == "1":
        return {}
    keep = allowed_external()
    applied = {}
    for var, sub in _CONTAINED.items():
        cur = os.environ.get(var)
        if var in keep or (cur and _inside(cur)):
            continue
        p = WORK / sub
        os.environ[var] = str(p)
        applied[var] = str(p)
        if create:
            p.mkdir(parents=True, exist_ok=True)
    return applied


## Applied on import so that anything importing pfpl - including a subprocess it
## later spawns, which inherits os.environ - is contained without each script
## having to remember. Directories are created lazily by ensure(), because import
## should not touch the filesystem.
CONTAINED = contain(create=False)


def ensure() -> None:
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)
    for v in set(CONTAINED.values()):
        Path(v).mkdir(parents=True, exist_ok=True)


def escapes() -> list[str]:
    """Configured paths NOT under ROOT, excluding those allowed via
    PFPL_ALLOW_EXTERNAL. An empty list means the project is fully contained."""
    keep = allowed_external()
    out = []
    for name, p in (("WORK", WORK), ("DATA", DATA), ("CACHE", CACHE), ("RUNS", RUNS),
                    ("LOGS", LOGS), ("SPLITS", SPLITS), ("REPORTS", REPORTS),
                    ("TMP", TMP)):
        if name not in keep and not _inside(p):
            out.append(f"{name}={p}")
    for var in _CONTAINED:
        v = os.environ.get(var)
        if v and var not in keep and not _inside(v):
            out.append(f"${var}={v}")
    return out


def describe() -> dict:
    return {
        "ROOT": str(ROOT), "resolved_via": ROOT_SOURCE,
        "WORK": str(WORK), "DATA": str(DATA), "CACHE": str(CACHE),
        "RUNS": str(RUNS), "LOGS": str(LOGS), "SPLITS": str(SPLITS),
        "REPORTS": str(REPORTS), "TMP": str(TMP),
        "contained_env": sorted(CONTAINED),
        "escapes_root": escapes(),
    }


def shell_exports() -> str:
    """The same containment as shell `export` lines, for slurm/_prelude.sh."""
    lines = [f'export PFPL_ROOT="{ROOT}"', f'export PFPL_WORK="{WORK}"']
    lines += [f'export {k}="{v}"' for k, v in sorted(CONTAINED.items())]
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys
    ensure()
    print(json.dumps(describe(), indent=1))
    bad = escapes()
    if bad:
        print("\nNOT CONTAINED - these point outside the project root:", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        print("\nOn a cluster this is how a job dies with 'Disk quota exceeded' after\n"
              "hours of work, or loses a shard to a node-local /tmp that is reclaimed\n"
              "when the job ends. Unset the offending variable, or point it under\n"
              f"{ROOT}.", file=sys.stderr)
        sys.exit(1)
    print(f"\nContained: everything is under {ROOT}")
