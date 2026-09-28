#!/usr/bin/env python3
"""Where did the pipeline actually stop? Traces artifacts, runs, logs and Slurm.

"Everything completed but the report says no runs" almost always means a job was
CANCELLED rather than run. `slurm/submit_wave*.sh` wires stages with
`--dependency=afterok:...` plus `--kill-on-invalid-dep=yes`, so an upstream failure
cancels everything downstream - and the report job is wired `afterany`, so it still
runs and finds an empty runs/ directory. `squeue` is empty either way, which is why
"everything completed" and "nothing ran" look identical from the outside.

This walks the expected artifacts in dependency order, names the first missing one,
lists what each run directory produced, pulls the first real error out of each log,
and asks Slurm for the job states.

    python scripts/status.py
    python scripts/status.py --logs embed --lines 12
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


from pfpl import paths  # noqa: E402

## (base, relative path, produced by, needed by, log prefix, optional)
##
## `splits/` lives under ROOT and is version-controlled (the splits are frozen and
## the numbers are measured against them), while bulk artifacts live under WORK,
## which is usually on scratch. The two bases are different directories on a
## cluster, so each entry names its own.
CHAIN = [
    ("splits", "quarantine.json", "scripts/phase0_verify.py --benchmark",
     "phase1 and every training run", None, False),
    ("work", "pairs.parquet", "scripts/phase1_harmonize.py", "everything", None, False),
    ("work", "proteins.parquet", "scripts/phase1_harmonize.py", "everything", None, False),
    ("work", "proteins.fasta", "scripts/phase1_harmonize.py", "mmseqs", None, False),
    ("work", "mmseqs/clusters.tsv", "slurm/20_mmseqs.sbatch", "the honest splits",
     "mmseqs", False),
    ("work", "ligand_phys.parquet", "scripts/phase3_ligand_phys.py --merge",
     "the triangle penalty, tiers AB/ABC", "ligphys", True),
    ("work", "geometry/geometry.pkl", "scripts/phase2_geometry.py --merge",
     "the distogram ANCHOR (Section 2.1) and Stage 1", "geom", True),
    ("splits", "protein_cluster.npz", "scripts/phase5_splits.py",
     "every training run", None, False),
    ("work", "protein_clusters.npy", "scripts/phase5_splits.py", "splits", None, False),
]

ERR = re.compile(r"(Traceback|Error|error:|ERROR|FAILED|Killed|OOM|oom-kill|CANCELLED|"
                 r"DependencyNeverSatisfied|slurmstepd|No such file|exceeded memory|"
                 r"CUDA out of memory|SystemExit|PREFLIGHT FAILED)", re.I)
NOISE = re.compile(r"UNEXPECTED|MISSING from the checkpoint|Loading weights|"
                   r"newly initialized|Consider training|TOKENIZERS_PARALLELISM|"
                   r"HF_TOKEN|lm_head|pooler\.dense|FutureWarning|UserWarning", re.I)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def sacct() -> None:
    if not any(os.access(os.path.join(p, "sacct"), os.X_OK)
               for p in os.environ.get("PATH", "").split(os.pathsep) if p):
        print("  sacct not on PATH (not a Slurm node) - skipping")
        return
    cmd = ["sacct", "-X", "--units=G", "-o",
           "JobID%14,JobName%24,State%22,ExitCode%8,Elapsed%10,MaxRSS%8,Reason%26"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=45).stdout
    except Exception as e:
        print(f"  sacct failed: {e}")
        return
    lines = [ln for ln in out.splitlines() if ln.strip()]
    keep = [ln for ln in lines if "pfpl" in ln or "JobID" in ln or "---" in ln]
    for ln in (keep or lines)[-40:]:
        print("  " + ln)
    bad = [ln for ln in keep if re.search(r"CANCELLED|FAILED|TIMEOUT|OUT_OF_ME", ln)]
    if bad:
        print("\n  *** Non-success states above. `DependencyNeverSatisfied` means the")
        print("      job NEVER RAN because an upstream job failed. Fix that one; the")
        print("      downstream jobs produced nothing because they did not execute. ***")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=None, metavar="PREFIX")
    ap.add_argument("--lines", type=int, default=6)
    a = ap.parse_args()

    print("=" * 84)
    print("PFPL PIPELINE STATUS")
    print("=" * 84)
    d = paths.describe()
    for k in ("ROOT", "resolved_via", "WORK", "RUNS", "LOGS", "SPLITS", "REPORTS",
              "TMP"):
        print(f"  {k:<14} {d[k]}")
    bad = d["escapes_root"]
    print(f"  {'contained':<14} "
          + ("yes - every cache and temp dir is under ROOT"
             if not bad else f"NO: {bad}"))
    if bad:
        print("\n  Anything outside the root will be written to $HOME or a node-local")
        print("  /tmp. On this cluster that means a quota failure hours into a job, or")
        print("  a shard whose scratch files vanish when the job ends.")
        print("  Diagnose:  python -m pfpl.paths")

    print("\n--- artifacts, in dependency order ------------------------------------")
    first_missing = None
    bases = {"work": paths.WORK, "splits": paths.SPLITS}
    for base, rel, by, needed, pfx, optional in CHAIN:
        p = (bases[base] / rel).resolve()
        rel = f"{base}/{rel}"
        if p.exists():
            size = p.stat().st_size if p.is_file() else sum(
                f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"  ok        {rel:<32} {human(size):>9}")
        else:
            tag = "OPTIONAL " if optional else "MISSING  "
            print(f"  {tag} {rel:<32} {'':>9}  <- {by}; needed by {needed}")
            if first_missing is None and not optional:
                first_missing = (rel, by, pfx)

    ## The anchor deserves its own line: its absence changes what the results MEAN,
    ## not just whether they exist.
    if not (paths.WORK / "geometry" / "geometry.pkl").exists():
        print("\n  NOTE: no geometry labels, so there is no distogram anchor. PLAN.md")
        print("  Section 2.1 lists the anchor as a REQUIRED mitigation: without it the")
        print("  affinity loss can push probability mass toward each pair's energy")
        print("  minimum and produce an over-attractive, geometrically impossible")
        print("  solution that scores well on pK. Runs in this state are fusion models")
        print("  with extra steps and must not be reported as physics results.")

    print("\n--- embedding caches ----------------------------------------------------")
    ptrs = sorted(paths.WORK.glob("plm_*.path"))
    if not ptrs:
        print("  none. Only plm.mode=none can train.")
    for p in ptrs:
        d = Path(p.read_text().strip())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        print(f"  {p.stem.replace('plm_', ''):<22} dim {meta.get('dim', '?'):<6} "
              f"{meta.get('n_proteins', '?')} proteins  {human(meta.get('bytes', 0))}")

    print("\n--- runs/ ---------------------------------------------------------------")
    if not paths.RUNS.exists() or not any(paths.RUNS.iterdir()):
        print("  runs/ is EMPTY - no training job produced output.")
    else:
        for d in sorted(x for x in paths.RUNS.iterdir() if x.is_dir()):
            have = [f for f in ("reports.json", "run.json", "predictions.csv",
                                "best.pt", "history.json") if (d / f).exists()]
            incomplete = "reports.json" not in have
            print(f"  {'INCOMPLETE' if incomplete else 'ok        '}  {d.name[:52]:<54}"
                  f"{', '.join(have) or '(empty)'}")
        print("\n  scripts/report.py collects runs/*/reports.json - a directory without")
        print("  one is invisible to it, which reads as 'no runs yet'.")
        has_c = any("_C_" in d.name or d.name.startswith("C_")
                    for d in paths.RUNS.iterdir() if d.is_dir())
        print(f"\n  matched no-physics control (variant C) present: {has_c}")
        if not has_c:
            print("    Without it there is no Gate G2 - Section 9's Tier 1 rule is")
            print("    defined against the control, not against a baseline table.")

    print("\n--- first error in each log ---------------------------------------------")
    if not paths.LOGS.exists() or not any(paths.LOGS.iterdir()):
        print("  logs/ is empty or absent. Jobs cancelled for an unsatisfied dependency")
        print("  never started, so they never wrote a log at all.")
    else:
        shown = 0
        for f in sorted(paths.LOGS.iterdir()):
            if f.suffix not in (".out", ".err") or f.stat().st_size == 0:
                continue
            if a.logs and not f.name.startswith(a.logs):
                continue
            try:
                lines = f.read_text(errors="replace").splitlines()
            except Exception:
                continue
            hits = [ln for ln in lines if ERR.search(ln) and not NOISE.search(ln)]
            if not hits:
                continue
            print(f"\n  {f.name}  ({len(lines)} lines)")
            for ln in hits[:a.lines]:
                print(f"      {ln.strip()[:150]}")
            shown += 1
        if not shown:
            print("  no error lines matched"
                  + (f" in logs starting with {a.logs!r}" if a.logs else ""))

    print("\n--- slurm job states ----------------------------------------------------")
    sacct()

    print("\n" + "=" * 84)
    if first_missing:
        rel, by, pfx = first_missing
        print(f"  VERDICT: the chain breaks at {rel}, which `{by}` should have written.")
        if pfx:
            print(f"  Look at that stage:  python scripts/status.py --logs {pfx}")
        else:
            print("  That step runs on the login node, so there is no Slurm log - "
                  "re-run it\n  by hand and read the error:")
            print(f"      python {by}")
        print("  Everything downstream was cancelled, not run, so empty output is")
        print("  expected rather than a second bug.")
    elif not paths.RUNS.exists() or not any(paths.RUNS.iterdir()):
        print("  VERDICT: every input artifact exists but runs/ is empty, so the")
        print("  training jobs themselves failed or were cancelled. Check sacct above.")
        print("      sbatch --array=0-$(python scripts/phase9_ablation.py --emit --count) \\")
        print("             slurm/60_train_grid.sbatch")
    else:
        print("  VERDICT: artifacts and runs both present - re-run the report:")
        print("      python scripts/report.py")
    print("=" * 84)


if __name__ == "__main__":
    main()
