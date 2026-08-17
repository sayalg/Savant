#!/usr/bin/env python
"""
Step 05 - Score every Boltz structure with PRODIGY-LIG.

    prodigy_lig -i work/structures/<id>.pdb -c A B:LIG

`-c` takes two arguments: the protein chain(s), then <chain>:<3-letter residue>
for the ligand. Both come from the chain contract set in step 01 and verified in
step 03 - PRODIGY refuses to run if residue LIG is not present exactly once in
chain B.

PRODIGY-LIG has two published scoring functions: a contacts-only one, and one
that also takes the electrostatics energy from a HADDOCK water refinement (`-e`),
which is the better-performing variant in the paper. Savant already computes a
HADDOCK `elec` term for the same complex, so when its JSON is available this runs
PRODIGY a second time with that value and reports both numbers. The two are kept
in separate columns; the `elec` here comes from Savant's emref stage rather than
a full water refinement, so it is close to but not identical to the term the
published model was fitted on.

This stage is cheap (seconds per pair) - a single core for the whole benchmark
is fine.

    python 05_run_prodigy.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import pandas as pd

from common import (LIGAND_CHAIN, LIGAND_RESNAME, MANIFEST, PRODIGY_RESULTS, PROTEIN_CHAIN,
                    RESULTS, RT_LN10_KCAL, SAVANT_RESULTS, STRUCT_DIR, ensure_dirs, log,
                    slurm_shard, take_shard, write_json)


def parse_output(stdout: str) -> tuple[float, float | None]:
    """Pull (DGprediction, DGscore) out of prodigy_lig's tab-separated report.

    The columns depend on which scoring function ran, so they are matched by
    name rather than position:

        no -e   Job name <tab> DGprediction (low refinement) (Kcal/mol)
        with -e Job name <tab> DGprediction (Kcal/mol) <tab> DGscore

    Verbose mode appends ten contact-count columns to either form. DGscore is
    None when the run did not produce one.
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if "DGprediction" not in line or i + 1 >= len(lines):
            continue
        row = dict(zip((h.strip() for h in line.split("\t")),
                       (v.strip() for v in lines[i + 1].split("\t"))))
        dg = next((v for k, v in row.items() if k.startswith("DGprediction")), None)
        if dg is None:
            continue
        dgscore = row.get("DGscore")
        return float(dg), float(dgscore) if dgscore else None
    raise ValueError(f"unrecognised prodigy_lig output: {stdout.strip()[:200]!r}")


def run_prodigy(binary: str, pdb: str, cutoff: float, elec: float | None,
                timeout: int) -> tuple[float, float | None]:
    cmd = [
        binary,
        "-i", pdb,
        "-c", PROTEIN_CHAIN, f"{LIGAND_CHAIN}:{LIGAND_RESNAME}",
        "-d", str(cutoff),
    ]
    if elec is not None:
        cmd += ["-e", str(elec)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise SystemExit(f"'{binary}' not on PATH - pip install prodigy-lig")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(f"exit {proc.returncode}: {tail[-1] if tail else 'no output'}")
    return parse_output(proc.stdout)


def savant_elec(savant_dir: str, bid: str) -> float | None:
    path = os.path.join(savant_dir, f"{bid}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return float(json.load(fh)["features"]["elec"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--struct-dir", default=STRUCT_DIR)
    ap.add_argument("--out-dir", default=PRODIGY_RESULTS)
    ap.add_argument("--savant-dir", default=SAVANT_RESULTS,
                    help="Savant results, read only for the HADDOCK elec term")
    ap.add_argument("--prodigy-bin", default="prodigy_lig")
    ap.add_argument("--cutoff", type=float, default=10.5, help="contact distance cutoff (A)")
    ap.add_argument("--no-electrostatics", action="store_true",
                    help="skip the second, electrostatics-aware PRODIGY run")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shard_id = args.shard_id if args.shard_id is not None else slurm_shard(args.num_shards)
    ensure_dirs(args.out_dir)

    manifest = pd.read_csv(args.manifest)
    pairs = take_shard(list(manifest.itertuples(index=False)), shard_id, args.num_shards)
    if args.limit:
        pairs = pairs[: args.limit]
    log(f"shard {shard_id}/{args.num_shards}: {len(pairs)} pairs")

    failures, n_done, n_skipped, n_elec = [], 0, 0, 0
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

        try:
            dg, dgscore = run_prodigy(args.prodigy_bin, pdb, args.cutoff, None, args.timeout)
        except Exception as exc:
            failures.append({"benchmark_id": bid, "error": f"{type(exc).__name__}: {exc}"})
            log(f"[{i}/{len(pairs)}] {bid}: FAILED - {exc}")
            continue

        payload = {
            "benchmark_id": bid,
            "prodigy_dg_kcal_per_mol": dg,
            "prodigy_dgscore": dgscore,
            # deltaG = -1.364 * pKd at 298 K, matching the benchmark's own
            # deltaG_kcal_per_mol column, so this lands on the label's scale.
            "prodigy_pkd": -dg / RT_LN10_KCAL,
            "prodigy_dg_elec_kcal_per_mol": None,
            "prodigy_dgscore_elec": None,
            "prodigy_pkd_elec": None,
            "haddock_elec": None,
        }

        elec = None if args.no_electrostatics else savant_elec(args.savant_dir, bid)
        if elec is not None:
            try:
                dg_e, dgscore_e = run_prodigy(args.prodigy_bin, pdb, args.cutoff, elec, args.timeout)
                payload.update(
                    prodigy_dg_elec_kcal_per_mol=dg_e,
                    prodigy_dgscore_elec=dgscore_e,
                    prodigy_pkd_elec=-dg_e / RT_LN10_KCAL,
                    haddock_elec=elec,
                )
                n_elec += 1
            except Exception as exc:
                log(f"[{i}/{len(pairs)}] {bid}: electrostatics variant failed - {exc}")

        write_json(out_json, payload)
        n_done += 1

    if failures:
        path = os.path.join(RESULTS, f"prodigy_failures_shard{shard_id:02d}.csv")
        pd.DataFrame(failures).to_csv(path, index=False)
        log(f"{len(failures)} failures -> {path}")

    log(f"shard {shard_id} done: {n_done} scored ({n_elec} with electrostatics), "
        f"{n_skipped} already present, {len(failures)} failed")
    if not args.no_electrostatics and n_done and not n_elec:
        log("no Savant JSONs found - run 04_run_savant.py first if you want the "
            "electrostatics-aware PRODIGY score too")


if __name__ == "__main__":
    main()
