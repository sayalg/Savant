#!/usr/bin/env python
"""
Step 04 - HADDOCK3 energies + contact count for every Boltz structure.

    python 04_run_savant.py --num-shards 20     # shard from SLURM_ARRAY_TASK_ID
    python 04_run_savant.py --limit 2           # local smoke test

WHAT THIS PRODUCES, AND WHY IT NO LONGER PREDICTS
-------------------------------------------------
savant3's `structure-based` model consumes seven numbers it cannot compute:

    score  total  vdw  elec  desolv  bsa        <- HADDOCK3, [topoaa] + [emref]
    num_interacting_residues                    <- distance query on the pose

This step produces exactly those. Prediction moved to the scoring notebook,
where all three savant3 models load once and score the benchmark in a batch;
doing it per pair here would reload ESM2-650M (2.5 GB) once per structure.

It calls HADDOCK3 through `savant_v0.5/savant05.py`, which carries that code
itself, so this step needs **no model files** - only that one module and haddock3
on PATH. Sharing the tool's own docking code rather than keeping a second copy
here is deliberate: it emits the same `[topoaa]` + `[emref]` workflow and reads
the same `capri_ss.tsv` columns that produced combined_50k_filtered.csv, so the
benchmark features match the training features by construction, and there is no
duplicate implementation that can drift.

THE CUTOFF DEFAULT CHANGED: 4.5 -> 6.0
--------------------------------------
The old step 04 passed `--cutoff 4.5`, but the training set was built at 6.0.
On dock_1, whose stored value is 22 interacting residues:

    4.0 -> 16      4.5 -> 18      5.0 -> 20      5.5 -> 21      6.0 -> 22

That mismatch is visible in the v2 benchmark, whose num_interacting_residues
median is 17 against the training set's 20. Anything scored at 4.5 fed the model
a systematically low feature. 6.0 is now the default; --cutoff still overrides.

COST, AND WHETHER IT IS WORTH IT
--------------------------------
HADDOCK3 energy minimisation runs minutes per complex, so 500 pairs needs the
array. Worth knowing before spending it: savant_v2's `interactions.py` records an
ablation on the 1000-pair benchmark where these six scalars added +0.012 Pearson
over a plain contact count. The internal training comparison put structure-based
about +0.05 R2 over structure-free. Expect the benchmark gap to be small.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from common import (LIGAND_CHAIN, MANIFEST, PROTEIN_CHAIN, RESULTS, SAVANT_RESULTS,
                    STRUCT_DIR, ensure_dirs, log, slurm_shard, take_shard)

## Same names and order as 06_merge_scores.SAVANT_FEATURES and savant05's
## HADDOCK_COLS, so the three cannot drift apart silently.
SAVANT_FEATURES = ["score", "total", "vdw", "elec", "desolv", "bsa",
                   "num_interacting_residues"]


def load_savant05(savant_path: str | None):
    """Import savant v0.5, which now carries the HADDOCK3 code itself.

    This used to import the legacy `savant` package. v0.5 inlined that code, so
    the benchmark and the tool run byte-identical docking - there is no longer a
    second copy that can drift.
    """
    default = Path(__file__).resolve().parents[4] / "savant_v0.5"
    root = Path(savant_path) if savant_path else default
    sys.path.insert(0, str(root))
    try:
        import savant05
    except ImportError as exc:
        raise SystemExit(
            f"cannot import savant05 ({exc}).\n"
            f"Looked in {root}. This step needs savant_v0.5/savant05.py for its "
            "HADDOCK3 wrapper - it does NOT need any .pkl models.\n"
            "Point at it with --savant-path /path/to/savant_v0.5."
        )
    if not savant05.haddock3_available():
        raise SystemExit(
            "haddock3 is not on PATH. Activate the env that provides it, or load "
            "your cluster's HADDOCK3 module."
        )
    return savant05


def run_one(sv, pdb: str, out_json: str, bid: str, workdir: Path, args):
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        protein_pdb, ligand_pdb = sv.split_chains(
            Path(pdb), args.protein_chain, args.ligand_chain, workdir)
        energies = sv.run_haddock3(protein_pdb, ligand_pdb, workdir,
                                   ncores=args.ncores, nemsteps=args.nemsteps)
        ## read_pdb returns (sequence, smiles, contacts) at savant05.CONTACT_CUTOFF
        _, _, n_res = sv.read_pdb(Path(pdb), args.protein_chain, args.ligand_chain)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:160]}"

    features = {**energies, "num_interacting_residues": float(n_res)}
    missing = [k for k in SAVANT_FEATURES if k not in features]
    if missing:
        return False, f"HADDOCK3 returned no {missing}"

    payload = {
        "benchmark_id": bid,
        ## 06_merge_scores reads features[k] -> haddock_<k>
        "features": {k: float(features[k]) for k in SAVANT_FEATURES},
        "meta": {"cutoff": args.cutoff, "nemsteps": args.nemsteps},
    }
    tmp = out_json + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, out_json)     # atomic: a killed job leaves no half-written JSON
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--struct-dir", default=STRUCT_DIR)
    ap.add_argument("--out-dir", default=SAVANT_RESULTS)
    ap.add_argument("--savant-path", default=None,
                    help="directory holding savant05.py (default: <repo>/savant_v0.5)")
    ap.add_argument("--protein-chain", default=PROTEIN_CHAIN)
    ap.add_argument("--ligand-chain", default=LIGAND_CHAIN)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None,
                    help="default: SLURM_ARRAY_TASK_ID, else 0")
    ap.add_argument("--ncores", type=int, default=None, help="cores for HADDOCK3")
    ap.add_argument("--cutoff", type=float, default=6.0,
                    help="contact cutoff in A; 6.0 reproduces the training values")
    ap.add_argument("--nemsteps", type=int, default=50)
    ap.add_argument("--scratch", default=None,
                    help="parent for HADDOCK run directories (default: $TMPDIR)")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="keep HADDOCK run dirs (large: ~tens of MB per pair)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run pairs that already have JSON")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sv = load_savant05(args.savant_path)
    ## read_pdb reads the module global, so --cutoff has to be applied there
    sv.CONTACT_CUTOFF = args.cutoff
    shard_id = args.shard_id if args.shard_id is not None else slurm_shard(args.num_shards)
    scratch = args.scratch or os.environ.get("TMPDIR") or os.path.join(RESULTS, "savant_tmp")
    ## RESULTS too: the failure log lands there, and a run whose first pair fails
    ## would otherwise die writing it.
    ensure_dirs(args.out_dir, scratch, RESULTS)

    manifest = pd.read_csv(args.manifest)
    pairs = take_shard(list(manifest.itertuples(index=False)), shard_id, args.num_shards)
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"shard {shard_id}/{args.num_shards}: {len(pairs)} pairs, "
        f"cutoff {args.cutoff} A, nemsteps {args.nemsteps}")

    failures, n_done, n_skipped, t0 = [], 0, 0, time.time()
    for i, m in enumerate(pairs, 1):
        bid = m.benchmark_id
        pdb = os.path.join(args.struct_dir, f"{bid}.pdb")
        out_json = os.path.join(args.out_dir, f"{bid}.json")

        if os.path.exists(out_json) and not args.overwrite:
            n_skipped += 1
            continue
        if not os.path.exists(pdb):
            failures.append({"benchmark_id": bid,
                             "error": "no structure (Boltz stage incomplete)"})
            continue

        workdir = Path(scratch) / f"savant_{bid}"
        shutil.rmtree(workdir, ignore_errors=True)
        started = time.time()
        ok, error = run_one(sv, pdb, out_json, bid, workdir, args)
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

        if ok:
            n_done += 1
            log(f"[{i}/{len(pairs)}] {bid}: ok ({time.time() - started:.0f}s)")
        else:
            failures.append({"benchmark_id": bid, "error": error})
            log(f"[{i}/{len(pairs)}] {bid}: FAILED - {error}")

    ## Per-shard failure log: array tasks run concurrently and would interleave
    ## writes to one shared file.
    if failures:
        path = os.path.join(RESULTS, f"savant_failures_shard{shard_id:02d}.csv")
        pd.DataFrame(failures).to_csv(path, index=False)
        log(f"{len(failures)} failures -> {path}")

    log(f"shard {shard_id} done: {n_done} extracted, {n_skipped} already present, "
        f"{len(failures)} failed, {(time.time() - t0) / 60:.1f} min")
    log("Next: python scripts/06_merge_scores.py, then the scoring notebook "
        "(which runs the savant3 models themselves)")


if __name__ == "__main__":
    main()
