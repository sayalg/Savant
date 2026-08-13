#!/usr/bin/env python
"""
Step 04 - Score every Boltz structure with the Savant CLI.

One `savant` invocation per pair:

    savant work/structures/<id>.pdb -p A -l B -s <docked_smiles> -o <id>.json

`-s` matters: without it Savant perceives the ligand's chemistry from
coordinates alone, and Boltz writes no bond orders. Passing the same SMILES that
produced the pose keeps the RDKit half of the feature vector exact and leaves
only the HADDOCK half depending on the geometry.

This is the expensive stage - each pair runs HADDOCK3 topology generation plus
energy minimisation - so it is sharded across a Slurm array and every pair is
skipped if its result JSON already exists. Failures are logged per shard and
simply retried by re-running the array.

    python 04_run_savant.py --num-shards 20            # shard from SLURM_ARRAY_TASK_ID
    python 04_run_savant.py --num-shards 1 --limit 3   # local smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

import pandas as pd

from common import (LIGAND_CHAIN, MANIFEST, MODEL_DIR, PROTEIN_CHAIN, RESULTS,
                    SAVANT_RESULTS, STRUCT_DIR, ensure_dirs, log, slurm_shard, take_shard)


def run_one(pdb: str, smiles: str, out_json: str, workdir: str, args) -> tuple[bool, str]:
    cmd = [
        args.savant_bin, pdb,
        "-p", PROTEIN_CHAIN,
        "-l", LIGAND_CHAIN,
        "-s", smiles,
        "-o", out_json,
        "-w", workdir,
        "--model-dir", args.model_dir,
        "--cutoff", str(args.cutoff),
        "--nemsteps", str(args.nemsteps),
    ]
    if args.ncores:
        cmd += ["--ncores", str(args.ncores)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {args.timeout}s"
    except FileNotFoundError:
        raise SystemExit(f"'{args.savant_bin}' not on PATH - activate the Savant environment")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, f"exit {proc.returncode}: {tail[-1] if tail else 'no output'}"
    if not os.path.exists(out_json):
        return False, "savant exited 0 but wrote no JSON"
    # A job killed mid-write would leave truncated JSON that a resumed run would
    # mistake for a finished pair; drop it here instead.
    try:
        with open(out_json) as fh:
            json.load(fh)
    except json.JSONDecodeError as exc:
        os.remove(out_json)
        return False, f"unreadable JSON: {exc}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--struct-dir", default=STRUCT_DIR)
    ap.add_argument("--out-dir", default=SAVANT_RESULTS)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None,
                    help="default: SLURM_ARRAY_TASK_ID, else 0")
    ap.add_argument("--savant-bin", default="savant")
    ap.add_argument("--model-dir", default=MODEL_DIR, help="directory with the .pkl models")
    ap.add_argument("--ncores", type=int, default=None, help="cores for HADDOCK3")
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--nemsteps", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per pair")
    ap.add_argument("--scratch", default=None,
                    help="parent for HADDOCK run directories (default: $TMPDIR or work/savant_tmp)")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="keep HADDOCK run dirs (large: ~tens of MB per pair)")
    ap.add_argument("--overwrite", action="store_true", help="re-run pairs that already have JSON")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shard_id = args.shard_id if args.shard_id is not None else slurm_shard(args.num_shards)
    scratch = args.scratch or os.environ.get("TMPDIR") or os.path.join(RESULTS, "savant_tmp")
    ensure_dirs(args.out_dir, scratch)

    manifest = pd.read_csv(args.manifest)
    pairs = take_shard(list(manifest.itertuples(index=False)), shard_id, args.num_shards)
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"shard {shard_id}/{args.num_shards}: {len(pairs)} pairs")

    failures, n_done, n_skipped, t0 = [], 0, 0, time.time()
    for i, m in enumerate(pairs, 1):
        bid = m.benchmark_id
        pdb = os.path.join(args.struct_dir, f"{bid}.pdb")
        out_json = os.path.join(args.out_dir, f"{bid}.json")

        if os.path.exists(out_json) and not args.overwrite:
            n_skipped += 1
            continue
        if not os.path.exists(pdb):
            failures.append({"benchmark_id": bid, "error": "no structure (Boltz stage incomplete)"})
            continue

        workdir = os.path.join(scratch, f"savant_{bid}")
        shutil.rmtree(workdir, ignore_errors=True)
        started = time.time()
        ok, error = run_one(pdb, m.docked_smiles, out_json, workdir, args)
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

        if ok:
            n_done += 1
            log(f"[{i}/{len(pairs)}] {bid}: ok ({time.time() - started:.0f}s)")
        else:
            failures.append({"benchmark_id": bid, "error": error})
            log(f"[{i}/{len(pairs)}] {bid}: FAILED - {error}")

    # Per-shard failure log: array tasks run concurrently and would interleave
    # writes to a single shared file.
    if failures:
        path = os.path.join(RESULTS, f"savant_failures_shard{shard_id:02d}.csv")
        pd.DataFrame(failures).to_csv(path, index=False)
        log(f"{len(failures)} failures -> {path}")

    log(f"shard {shard_id} done: {n_done} scored, {n_skipped} already present, "
        f"{len(failures)} failed, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
