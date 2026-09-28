"""Dependency preflight. Fail in two seconds, not after loading 600k rows.

Carried over from the previous sprint, where every one of these failure modes cost
real time on the cluster. Run it standalone for a full report:

    python -m pfpl.preflight

TWO LESSONS BAKED IN, both from misreporting a real environment:

1. Import the SUBMODULE that exercises the C extension, not the top-level package.
   `import rdkit` succeeds on a NumPy-1-built RDKit; `import rdkit.Chem` is what
   actually fails. An earlier version reported "rdkit 2023.09.6 OK" for an
   installation that could not load a molecule.
2. Report the INSTALLED version even when the import fails, via importlib.metadata.
   "MISSING" is wrong and misleading when the package is present but ABI-broken -
   the version is the thing that tells you what to upgrade.
"""
from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version as _dist_version

CHECKS = [
    ("numpy", "numpy", "everything", True),
    ("pandas", "pandas", "everything", True),
    ("yaml", "pyyaml", "every config in configs/", True),
    ("scipy", "scipy", "metrics, bootstrap CIs", True),
    ("sklearn.model_selection", "scikit-learn", "splits, PCA, baseline models", True),
    ("rdkit.Chem", "rdkit", "standardisation, descriptors, conformers, scaffolds", True),
    ("torch", "torch", "the model, the encoders", True),
    ("transformers", "transformers", "ESM-2 / ESM-C embedding extraction", True),
    ("xgboost", "xgboost", "ligand-only and GBDT baselines (Phase 6)", True),
    ("h5py", "h5py", "sharded embedding cache (Phase 4)", True),
    ("catboost", "catboost", "optional GBDT baseline", False),
    ("sentencepiece", "sentencepiece", "ProtT5 tokenizer, if that PLM is used", False),
    ("Bio.PDB", "biopython", "mmCIF/PDB parsing for geometry labels (Phase 2)", False),
    ("tabulate", "tabulate", "markdown tables in reports", False),
]

GLIBCXX_MARKERS = ("glibcxx_", "libstdc++.so", "glibc_", "cxxabi_")
ABI_MARKERS = (
    "compiled using numpy 1.x", "numpy.dtype size changed", "binary incompatibility",
    "_multiarray_umath", "numpy.core.multiarray failed to import",
    "numpy.ufunc size changed",
)

GLIBCXX_FIX = """
A compiled extension needs a NEWER libstdc++ than the node provides. The symbol
version in the error maps to a GCC release:

    GLIBCXX_3.4.30 = GCC 12    GLIBCXX_3.4.31 = GCC 13.1
    GLIBCXX_3.4.32 = GCC 13.2  GLIBCXX_3.4.33 = GCC 14

STEP 1 - find out whether the env ALREADY has a new enough copy. This decides which
of the two fixes you need, and they are not interchangeable:

    strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep -o 'GLIBCXX_3\\.4\\.[0-9]*' \\
        | sort -V | tail -3

CASE A - the env's newest is >= the version in the error (this is the common case,
and it is what happened on this cluster: the env had 3.4.35 while the error asked
for 3.4.31). Nothing needs installing. It is a LOAD-ORDER problem: conda links with
DT_RUNPATH, which unlike DT_RPATH is NOT inherited by dynamically loaded extension
modules, so the .so falls through to the ldconfig cache and finds /lib64. Redirect
just that one library:

    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

Every sbatch script here takes:   CONDA_LIBS_FIRST=1 ./slurm/submit_wave1.sh
Broader fallback (riskier - can shadow CUDA):  CONDA_LIBS_FIRST=2

CASE B - the env's newest is older than the error's, or the file is absent:

    conda install --solver=libmamba -c conda-forge "libstdcxx-ng>=13.2"

Full report:  python env/gpu_check.py --libs
"""


def _is_abi_break(msg: str) -> bool:
    return any(k in msg.lower() for k in ABI_MARKERS)


def _is_glibcxx(msg: str) -> bool:
    return any(k in msg.lower() for k in GLIBCXX_MARKERS)


def _installed(dist: str) -> str | None:
    try:
        return _dist_version(dist)
    except (PackageNotFoundError, Exception):
        return None


def _fix_text(broken: list[tuple[str, str | None]]) -> str:
    names = ", ".join(f"{d} {v or '?'}" for d, v in broken)
    return "\n".join([
        "", "=" * 78, "NumPy ABI BREAK", "=" * 78,
        f"NumPy here is {_installed('numpy') or 'unknown'}, but these were compiled",
        f"against NumPy 1.x and cannot run against 2.x:", f"    {names}", "",
        "Nothing in the Python layer can work around a C-API ABI mismatch.", "",
        "RECOMMENDED (one command, no API churn):", "",
        '    pip install "numpy<2"', "",
        "That resolves to 1.26.4 and fixes every package above at once. NumPy 2 wheels",
        "stay backward compatible down to 1.19, so scipy/pandas/torch keep working.", "",
        "ALTERNATIVE (forward-looking, more risk):", "",
        '    pip install -U "scikit-learn>=1.6" "rdkit>=2025.03"', "",
        "Then re-run:  python -m pfpl.preflight", "=" * 78, "",
    ])


def require(module: str, dist: str, what: str, hint: str = "") -> None:
    """Import or exit with a targeted message. Call BEFORE any expensive work."""
    try:
        importlib.import_module(module)
        return
    except Exception as e:
        v = _installed(dist)
        print(f"\n{'=' * 78}\nPREFLIGHT FAILED: {what} needs {dist}\n{'=' * 78}",
              file=sys.stderr)
        print(f"{dist} {v or '(not installed)'}: "
              f"{type(e).__name__}: {str(e).strip().splitlines()[0][:110]}", file=sys.stderr)
        if _is_glibcxx(str(e)):
            print(GLIBCXX_FIX, file=sys.stderr)
        elif v is not None and _is_abi_break(str(e)):
            print(_fix_text([(dist, v)]), file=sys.stderr)
        elif v is None:
            print(f"\n    pip install {dist}\n", file=sys.stderr)
        if hint:
            print(hint, file=sys.stderr)
        sys.exit(2)


def require_rdkit(what: str = "this step") -> None:
    require("rdkit.Chem", "rdkit", what)


def require_torch(what: str = "this step") -> None:
    require("torch", "torch", what)


def require_h5py(what: str = "the embedding cache") -> None:
    require("h5py", "h5py", what)


def gpu_usable() -> bool:
    """One predicate for "can we actually run on a GPU".

    `torch.cuda.is_available()` alone is not enough: with CUDA_VISIBLE_DEVICES=""
    it can return True while device_count() is 0, so a script testing only
    is_available() picks "cuda" for a device that is not there. Every device
    decision in this project goes through here so the guard and the choice agree.
    """
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count())
    except Exception:
        return False


def pick_device() -> str:
    return "cuda" if gpu_usable() else "cpu"


def warn_if_cpu(what: str = "this step") -> bool:
    if gpu_usable():
        return True
    try:
        import torch
        built = torch.version.cuda
    except Exception:
        built = None
    print("\n" + "!" * 78, file=sys.stderr)
    print(f"NO GPU VISIBLE - {what} is running on CPU", file=sys.stderr)
    print("!" * 78, file=sys.stderr)
    if built is None:
        print("torch has no CUDA build (torch.version.cuda is None), so no allocation\n"
              "will help. Reinstall a CUDA torch into this env.", file=sys.stderr)
    else:
        print("Most likely you are on a LOGIN NODE. The sbatch scripts request a GPU;\n"
              "running a script directly does not. Get an allocation first:\n\n"
              "    srun --partition=GPU --gres=gpu:1 --constraint=FP32 \\\n"
              "         --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash\n",
              file=sys.stderr)
    print("Full diagnosis:  python env/gpu_check.py\n" + "!" * 78 + "\n", file=sys.stderr)
    return False


def check(verbose: bool = True) -> dict:
    state: dict[str, dict] = {}
    for target, dist, why, required in CHECKS:
        v = _installed(dist)
        try:
            importlib.import_module(target)
            status, err = "ok", ""
        except Exception as e:
            err = (str(e).strip().splitlines() or [""])[0][:58]
            ## Order matters: a GLIBCXX failure can mention numpy in its traceback
            ## (torch imports numpy), so test the libstdc++ signature FIRST or the
            ## NumPy advice wins and sends you down the wrong path.
            if _is_glibcxx(str(e)):
                status = "glibcxx"
            elif v is None:
                status = "missing"
            elif _is_abi_break(str(e)):
                status = "abi-break"
            else:
                status = "broken"
        state[dist] = dict(status=status, version=v, required=required, error=err)
        if verbose:
            tag = {"ok": "ok", "missing": "NOT INSTALLED", "abi-break": "ABI BREAK",
                   "glibcxx": "LIBSTDC++ TOO OLD", "broken": "IMPORT FAILS"}[status]
            print(f"  {dist:<16} {(v or '-'):<12} {tag:<18} {why}"
                  + (f"\n{'':>50}<- {err}" if status != "ok" else ""))
    return state


def main() -> None:
    from .paths import describe
    print("=" * 78)
    print("pfpl preflight")
    print("=" * 78)
    print(f"  {'python':<16} {sys.version.split()[0]:<12} {'ok':<18} {sys.executable}")
    st = check()
    print("=" * 78)
    for k, v in describe().items():
        print(f"  {k:<14} {v}")
    print("=" * 78)

    glibcxx = [d for d, s in st.items() if s["status"] == "glibcxx"]
    if glibcxx:
        print(f"  {len(glibcxx)} package(s) blocked by an old system libstdc++: {glibcxx}")
        print(GLIBCXX_FIX)
        sys.exit(2)

    abi = [(d, s["version"]) for d, s in st.items() if s["status"] == "abi-break"]
    missing = [d for d, s in st.items() if s["status"] == "missing" and s["required"]]
    broken = [d for d, s in st.items() if s["status"] == "broken" and s["required"]]

    ## xgboost does `from .sklearn import ...` at package import, so a broken
    ## scikit-learn takes xgboost down with it and the failure carries sklearn's ABI
    ## message. Naming xgboost there sends you upgrading the wrong package.
    if st.get("scikit-learn", {}).get("status") in ("abi-break", "broken") \
            and st.get("xgboost", {}).get("status") != "ok":
        abi = [(d, v) for d, v in abi if d != "xgboost"]
        print("  NOTE: xgboost fails only because it imports scikit-learn at package\n"
              "        import. Fix scikit-learn and xgboost recovers on its own.")
    if abi:
        print(_fix_text(abi))
        sys.exit(2)
    if missing or broken:
        if missing:
            print(f"  NOT READY - not installed: {missing}\n    pip install {' '.join(missing)}")
        for d in broken:
            print(f"  NOT READY - {d} ({st[d]['version']}) import fails: {st[d]['error']}")
        sys.exit(2)
    opt = [d for d, s in st.items() if s["status"] != "ok" and not s["required"]]
    print(f"  READY{f' (optional not installed: {opt})' if opt else ''}")


if __name__ == "__main__":
    main()
