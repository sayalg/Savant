#!/usr/bin/env python3
"""Why is torch on the CPU? Distinguishes the four causes that look identical.

`device cpu` in a selfcheck has several possible causes and they need different
fixes, so guessing wastes a submission cycle:

  1. running on the LOGIN NODE          - most common; sbatch asked for a GPU, but
                                          invoking the script directly did not
  2. allocation has no --gres=gpu:1     - `srun --partition=GPU` alone gives no GPU
  3. torch is a CPU-ONLY build          - torch.version.cuda is None; no allocation
                                          will ever help
  4. driver / visibility problem        - GPU allocated but torch cannot open it

Run it exactly where the selfcheck ran:

    python env/gpu_check.py                    # tells you if you are on a login node
    srun --partition=GPU --gres=gpu:1 --constraint=FP32 --pty python env/gpu_check.py
    python env/gpu_check.py --nodes            # which nodes have GPUs, and which
                                               # carry the FP32 feature
    python env/gpu_check.py --libs             # system vs conda-env libstdc++, and
                                               # which of the two GLIBCXX fixes applies
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

SLURM_VARS = ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
              "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE",
              "SLURM_JOB_CONSTRAINT", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES")

SRUN = ("srun --partition=GPU --gres=gpu:1 --constraint=FP32 "
        "--cpus-per-task=4 --mem=48G --time=01:00:00 --pty bash")


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    if not shutil.which(cmd[0]):
        return 127, f"{cmd[0]} not on PATH"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def driver_max_cuda() -> str | None:
    """The 'CUDA Version: X.Y' in nvidia-smi's header = the MAXIMUM the driver
    supports, not what is installed. A wheel built for any CUDA <= this works."""
    rc, out = run(["nvidia-smi"])
    if rc != 0:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def wheel_index(max_cuda: str | None) -> tuple[str, str]:
    """Pick the PyTorch wheel channel for a driver. Returns (tag, index url)."""
    tag = "cu128"
    if max_cuda:
        try:
            major, minor = (int(x) for x in max_cuda.split(".")[:2])
            if major >= 13:
                tag = "cu128"        # cu128 wheels run fine on a 13.x driver
            elif (major, minor) >= (12, 8):
                tag = "cu128"
            elif (major, minor) >= (12, 6):
                tag = "cu126"
            elif major == 12:
                tag = "cu121"
            else:
                tag = "cu118"
        except ValueError:
            pass
    return tag, f"https://download.pytorch.org/whl/{tag}"


def torch_build_markers() -> tuple[bool, list[str]]:
    """Second, unambiguous signal: a CPU-only wheel ships no CUDA libraries."""
    try:
        import glob
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        found = [os.path.basename(p) for p in glob.glob(os.path.join(lib, "*cuda*"))]
        return bool(found), found[:6]
    except Exception:
        return False, []


GCC_BY_GLIBCXX = {
    "3.4.29": "GCC 11", "3.4.30": "GCC 12", "3.4.31": "GCC 13.1",
    "3.4.32": "GCC 13.2", "3.4.33": "GCC 14", "3.4.34": "GCC 15",
    "3.4.35": "GCC 16",
}


def gcc_for(v: str) -> str:
    """Name the toolchain, and degrade gracefully past the end of the table rather
    than printing 'GCC ?' for a version that is simply newer than we tabulated."""
    if v in GCC_BY_GLIBCXX:
        return GCC_BY_GLIBCXX[v]
    try:
        patch = int(v.split(".")[2])
        newest = max(int(k.split(".")[2]) for k in GCC_BY_GLIBCXX)
        if patch > newest:
            return f"newer than {GCC_BY_GLIBCXX[f'3.4.{newest}']}"
    except (IndexError, ValueError):
        pass
    return "GCC ?"


def glibcxx_versions(path: str) -> list[str]:
    """Sorted GLIBCXX_* symbol versions exported by a libstdc++, newest last."""
    if not os.path.exists(path):
        return []
    rc, out = run(["strings", path])
    if rc != 0:
        ## no `strings` on the node - fall back to scanning the binary ourselves
        try:
            blob = open(path, "rb").read()
            out = "\n".join(m.decode() for m in re.findall(rb"GLIBCXX_[0-9.]+", blob))
        except Exception:
            return []
    vs = {m.group(1) for m in re.finditer(r"GLIBCXX_([0-9]+\.[0-9]+\.[0-9]+)", out)}
    return sorted(vs, key=lambda s: [int(x) for x in s.split(".")])


def show_libs() -> None:
    """Which libstdc++ the system has, which the conda env has, and who wins."""
    print("=" * 78)
    print("libstdc++ / GLIBCXX REPORT")
    print("=" * 78)
    prefix = os.environ.get("CONDA_PREFIX", "")
    cands = [("system", "/lib64/libstdc++.so.6"),
             ("system(alt)", "/usr/lib64/libstdc++.so.6"),
             ("conda env", os.path.join(prefix, "lib", "libstdc++.so.6") if prefix else "")]
    best = {}
    for label, path in cands:
        if not path:
            continue
        vs = glibcxx_versions(path)
        if not vs:
            print(f"  {label:<12} {path}\n               (absent or unreadable)")
            continue
        top = vs[-1]
        best[label] = top
        print(f"  {label:<12} {path}")
        print(f"               newest GLIBCXX_{top}  ({gcc_for(top)})"
              f"   [{len(vs)} versions]")

    print(f"\n  CONDA_PREFIX      {prefix or '(unset)'}")
    print(f"  LD_LIBRARY_PATH   {os.environ.get('LD_LIBRARY_PATH', '(unset)')}")

    ## which one a freshly dlopen'd extension will actually bind
    rc, out = run(["ldconfig", "-p"])
    if rc == 0:
        hits = [l.strip() for l in out.splitlines() if "libstdc++.so.6" in l][:4]
        print("\n  ldconfig cache:")
        for h in hits:
            print(f"    {h}")

    print("\n" + "=" * 78)
    sysv = best.get("system") or best.get("system(alt)")
    envv = best.get("conda env")
    if envv and sysv and [int(x) for x in envv.split(".")] > [int(x) for x in sysv.split(".")]:
        print(f"  CASE A: the env's libstdc++ (GLIBCXX_{envv}) is NEWER than the system's")
        print(f"  (GLIBCXX_{sysv}). Nothing needs installing - this is LOAD ORDER. conda")
        print("  links with DT_RUNPATH, which unlike DT_RPATH is not inherited by")
        print("  dynamically loaded extension modules, so the .so finds /lib64 instead.")
        print("\n  Redirect only that library (surgical, cannot shadow the CUDA libs):\n")
        print(f'      export LD_PRELOAD="{prefix}/lib/libstdc++.so.6"\n')
        print("  For jobs:   CONDA_LIBS_FIRST=1 ./sbatch/submit_all.sh")
        print("  Broader fallback if another library is stale too (riskier):")
        print("              CONDA_LIBS_FIRST=2 ./sbatch/submit_all.sh")
        print("\n  Then re-run `python env/gpu_check.py` to confirm CUDA still resolves.")
    elif envv and sysv:
        print(f"  CASE B: the env's libstdc++ (GLIBCXX_{envv}) is NOT newer than the")
        print(f"  system's (GLIBCXX_{sysv}), so LD_PRELOAD/LD_LIBRARY_PATH cannot help.")
        print("  Install a newer one into the env:\n")
        print('      conda install -c conda-forge "libstdcxx-ng>=13.2"')
        print("      # if the solve hangs:  conda install --solver=libmamba ...\n")
    else:
        print("  Could not read both libraries - run this on the compute node where the")
        print("  import failed. If the env's copy is missing entirely:\n")
        print('      conda install -c conda-forge "libstdcxx-ng>=13.2"\n')
    print("  Symbol -> toolchain: " + ", ".join(
        f"GLIBCXX_{k}={v}" for k, v in list(GCC_BY_GLIBCXX.items())[1:5]))
    print("=" * 78)


def show_nodes() -> None:
    """Which nodes actually have GPUs, and which features they advertise."""
    print("=" * 78)
    print("GPU NODES AND FEATURES")
    print("=" * 78)
    queries = [
        (["sinfo", "-o", "%20P %12N %8T %10c %10m %20G %s"],
         "partitions: name, nodes, state, cpus, mem, GRES (gpu:type:count), features"),
        (["sinfo", "-p", "GPU", "-N", "-o", "%N %8T %20G %s"],
         "per-node view of the GPU partition - GRES column must be non-null"),
        (["sinfo", "-o", "%20N %s"],
         "features per node - the --constraint=FP32 value must appear here"),
    ]
    for cmd, why in queries:
        print(f"\n$ {' '.join(cmd)}\n  # {why}")
        rc, out = run(cmd)
        print("\n".join("  " + l for l in out.splitlines()[:40]) or "  (no output)")
        if rc == 127:
            print("  -> not a Slurm login node, or sinfo unavailable here")
            break
    print("\n" + "=" * 78)
    print("Read the GRES column: `gpu:a100:4` means the node has GPUs; `(null)` means")
    print("it does not, whatever the partition is called. The features column is what")
    print("--constraint matches, so FP32 must literally appear there for our sbatch")
    print("scripts to be schedulable.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", action="store_true",
                    help="query Slurm for which nodes have GPUs and which have FP32")
    ap.add_argument("--libs", action="store_true",
                    help="report system vs conda-env libstdc++ / GLIBCXX versions")
    a = ap.parse_args()
    if a.nodes:
        show_nodes()
        return
    if a.libs:
        show_libs()
        return

    print("=" * 78)
    print("GPU CHECK - run this exactly where the selfcheck ran")
    print("=" * 78)
    print(f"  hostname      {os.uname().nodename if hasattr(os, 'uname') else 'n/a'}")
    print("\n  Slurm environment:")
    env = {k: os.environ.get(k) for k in SLURM_VARS}
    for k, v in env.items():
        print(f"    {k:<24} {v if v is not None else '(unset)'}")

    in_slurm = bool(env["SLURM_JOB_ID"])
    gres = any(env[k] for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                                "SLURM_GPUS_ON_NODE", "GPU_DEVICE_ORDINAL"))
    cvd = env["CUDA_VISIBLE_DEVICES"]
    visible = cvd not in (None, "", "NoDevFiles")

    print("\n  nvidia-smi:")
    rc, out = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                   "--format=csv,noheader"])
    print("\n".join("    " + l for l in out.splitlines()[:8]) or "    (no output)")
    smi_ok = rc == 0 and bool(out) and "not on PATH" not in out

    print("\n  torch:")
    try:
        import torch
        built = torch.version.cuda
        avail = torch.cuda.is_available()
        n = torch.cuda.device_count() if avail else 0
        print(f"    torch.__version__        {torch.__version__}")
        print(f"    torch.version.cuda       {built}   <- None means a CPU-ONLY build")
        print(f"    torch.cuda.is_available  {avail}")
        print(f"    device_count             {n}")
        has_libs, libs = torch_build_markers()
        print(f"    CUDA libs in torch/lib   {libs if has_libs else 'NONE -> CPU-only build'}")
        maxc = driver_max_cuda()
        if maxc:
            print(f"    driver supports CUDA <=  {maxc}   (nvidia-smi header; a wheel built "
                  f"for any CUDA <= this works)")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"      [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    except Exception as e:
        print(f"    torch import FAILED: {type(e).__name__}: {e}")
        built, avail, n = None, False, 0

    print("\n" + "=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)
    if avail and n:
        print("  GPU is visible and usable. If a selfcheck still says 'device cpu',")
        print("  it was run in a DIFFERENT shell than this one - re-run it here.")
        print("=" * 78)
        return

    if built is None:
        has_libs, libs = torch_build_markers()
        maxc = driver_max_cuda()
        tag, idx = wheel_index(maxc)
        print("  CAUSE 3: torch is a CPU-ONLY build (torch.version.cuda is None).")
        print(f"  Confirmed by torch/lib: {'CUDA libs present?! ' + str(libs) if has_libs else 'no CUDA libraries present'}")
        print(f"  Driver supports CUDA up to {maxc or 'unknown'} (from nvidia-smi's header).")
        print("  No allocation will fix this. Reinstall a CUDA build:\n")
        print(f"      pip install --force-reinstall --index-url {idx} torch\n")
        print("  NOTE on version matching: the wheel's CUDA runtime only has to be")
        print(f"  <= the driver's maximum ({maxc or '?'}), it does not have to equal it.")
        print(f"  So {tag} is the right choice here, and an older cu12x wheel would also")
        print("  work if that build is unavailable for your torch version.\n")
        print("  ORDER MATTERS if you are also pinning numpy: installing torch can pull")
        print("  numpy back up to 2.x and re-break rdkit/scikit-learn. Do torch FIRST,")
        print("  then re-pin, then verify:\n")
        print(f"      pip install --force-reinstall --index-url {idx} torch")
        print('      pip install "numpy<2"')
        print("      python -m pfpl.preflight")
        print("      python env/gpu_check.py\n")
        print("  Afterwards torch.version.cuda must be non-None and torch/lib must")
        print("  contain libtorch_cuda.so.")
    elif not in_slurm:
        print("  CAUSE 1: you are on a LOGIN NODE - SLURM_JOB_ID is unset.")
        print("  sbatch scripts request a GPU, but running the script directly does not.")
        print("  Get an interactive allocation first:\n")
        print(f"      {SRUN}\n")
        print("  then re-run the selfcheck inside it. Or submit it as a job:")
        print("      sbatch --partition=GPU --gres=gpu:1 --constraint=FP32 \\")
        print("             --wrap 'python scripts/02_esm_residue.py --selfcheck'")
    elif not gres and not visible:
        print("  CAUSE 2: inside a Slurm job, but NO GPU was allocated to it.")
        print(f"  partition={env['SLURM_JOB_PARTITION']}  SLURM_JOB_GPUS unset.")
        print("  Asking for --partition=GPU is not enough; the GRES must be requested:")
        print(f"\n      {SRUN}\n")
        print("  Every sbatch script here already carries --gres=gpu:1 --constraint=FP32.")
    elif not smi_ok:
        print("  CAUSE 4: a GPU appears allocated but nvidia-smi cannot talk to a driver.")
        print("  That is a node-level problem, not a code one - report the node")
        print(f"  ({env['SLURM_JOB_NODELIST']}) to the cluster admins, and meanwhile")
        print("  exclude it:  sbatch --exclude=<node> ...")
    else:
        print("  CAUSE 4: GPU allocated and nvidia-smi works, but torch cannot open it.")
        print(f"  CUDA_VISIBLE_DEVICES={cvd!r}, torch built for CUDA {built}.")
        print("  Usually a driver/toolkit mismatch: the driver is older than the CUDA")
        print("  version torch was built against. Compare the driver_version above with")
        print("  torch.version.cuda and install a torch built for an older CUDA.")
    print("\n  See which nodes actually have GPUs:  python env/gpu_check.py --nodes")
    print("=" * 78)
    sys.exit(1)


if __name__ == "__main__":
    main()
