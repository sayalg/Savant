#!/usr/bin/env python3
"""Phase 3: ligand physics featurisation, Tiers A-C (PLAN.md Section 8, Phase 3).

CPU-node array work. Tier A is RDKit and always runs; Tiers B and C shell out to
xtb and are skipped cleanly (as NaN) when it is not on PATH - Section 6.3 allows a
surrogate, but a missing feature must degrade one channel rather than drop a ligand.

The conformer ensemble is the part with two jobs. Its mean internal distance matrix
is the bound in `physics/energy.triangle_penalty` - the constraint that stops
Variant A's independent distributions from describing an impossible geometry - and
its spread feeds the pair module's geometric prior. Without it Variant A has no
self-consistency signal at all, so this phase is not optional for the physics arm.

Gate: features cached, failure rate < 2% (or explained), timing reported.

    sbatch --array=0-19 slurm/30_ligand_phys.sbatch
    python scripts/phase3_ligand_phys.py --merge
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import conf, paths  # noqa: E402
from pfpl.features import ligand_phys as LP  # noqa: E402
from pfpl.features.cache import slurm_shard  # noqa: E402

OUT = paths.WORK / "ligand_phys"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def write_dist_store(df: pd.DataFrame) -> None:
    """Ragged fp16 store of each ligand's (S, S) site-distance matrix.

    work/ligand_dist/{dist.f16, offsets.npy, n_sites.npy}, indexed DIRECTLY by
    ligand_idx (row i is ligand_idx i; an index with no ligand has 0 sites).
    ~35 sites -> ~2.5 KB per ligand.
    """
    d = paths.WORK / "ligand_dist"
    d.mkdir(parents=True, exist_ok=True)
    li = df["ligand_idx"].astype(np.int64).to_numpy()
    ns = np.zeros(int(li.max()) + 1, np.int64)
    ns[li] = df["_n_sites"].fillna(0).astype(np.int64).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(ns * ns)])
    mm = np.memmap(d / "dist.f16.tmp", dtype=np.float16, mode="w+",
                   shape=(max(int(offsets[-1]), 1),))
    for i, b in zip(li, df["_site_dist"]):
        if ns[i]:
            mm[offsets[i]:offsets[i + 1]] = np.frombuffer(b, np.float16)
    mm.flush()
    del mm
    (d / "dist.f16.tmp").replace(d / "dist.f16")
    np.save(d / "offsets.npy", offsets)
    np.save(d / "n_sites.npy", ns)
    (d / "meta.json").write_text(json.dumps({
        "n_ligands": len(df), "missing": int((ns[li] == 0).sum()),
        "bound": "ensemble max over conformers", "order": "embed_ligand.build_graph"},
        indent=1))
    log(f"site-distance store: {len(df):,} ligands, {int((ns[li] == 0).sum()):,} without "
        f"conformers, {offsets[-1] * 2 / 1e9:.2f} GB -> {d}")


def ligand_table() -> pd.DataFrame:
    p = paths.WORK / "pairs.parquet"
    if not p.exists():
        raise SystemExit(f"{p} missing - run scripts/phase1_harmonize.py first")
    pairs = pd.read_parquet(p, columns=["ligand_idx", "inchikey_full",
                                        "ligand_smiles_std"])
    return (pairs.drop_duplicates("ligand_idx").sort_values("ligand_idx")
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--tier", default=None, help="A | AB | ABC (default: from config)")
    ap.add_argument("--n-conf", type=int, default=16)
    ap.add_argument("--xtb-bin", default="xtb")
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    cfg = conf.from_args(a)
    tier = (a.tier or cfg.data.get("ligand_feature_tier", "A")).upper()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.selfcheck:
        t0 = time.time()
        d = LP.compute("CC(=O)Oc1ccccc1C(=O)O", tier=tier, n_conf=a.n_conf,
                       xtb_bin=a.xtb_bin)
        dt = time.time() - t0
        print(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                          for k, v in d.items()}, indent=1, default=float))
        n = len(ligand_table()) if (paths.WORK / "pairs.parquet").exists() else 300_000
        print(f"\n  {dt:.2f} s for one ligand at tier {tier}")
        print(f"  {n:,} ligands -> {n * dt / 3600:.1f} CPU-hours "
              f"({n * dt / 3600 / 20:.1f} h across a 20-task array)")
        missing = [k for k, v in d.items() if isinstance(v, float) and not np.isfinite(v)]
        if missing:
            print(f"  NaN channels (tool not on PATH, or not requested): {missing}")
        log("SELFCHECK PASSED")
        return

    if a.merge:
        shards = sorted(p for p in OUT.glob("shard_*.parquet")
                        if "_part" not in p.name and ".tmp" not in p.name)
        if not shards:
            raise SystemExit(f"no shards in {OUT}")
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        df = df.sort_values("ligand_idx").reset_index(drop=True)
        lig = ligand_table()
        if len(df) != len(lig):
            raise SystemExit(
                f"merged {len(df):,} rows but there are {len(lig):,} ligands. A shard "
                "is missing or the sharding geometry changed between runs; a merge "
                "with holes would misalign every downstream index.")
        if "_site_dist" in df.columns:
            write_dist_store(df)
        else:
            print("  WARNING: shards predate the site-distance cache; the triangle "
                  "penalty will be inactive. Re-run the array to build it.")
        if "_error" in df:
            errs = df["_error"].dropna().value_counts().to_dict()
            if errs:
                print(f"  per-ligand exceptions (row kept as NaN): {errs}")
        ## Numeric columns only: phase6_baselines uses every column as a feature.
        df = df.drop(columns=[c for c in ("_site_dist", "_n_sites", "_error") if c in df])
        df.to_parquet(paths.WORK / "ligand_phys.parquet", index=False)
        cols = LP.tier_names(tier)
        present = [c for c in cols if c in df.columns]
        fail = {c: float(df[c].isna().mean()) for c in present}
        worst = sorted(fail.items(), key=lambda kv: -kv[1])[:6]
        meta = {"n_ligands": len(df), "tier": tier, "n_shards": len(shards),
                "nan_fraction_worst": dict(worst),
                "tier_a_failure_rate": float(df["mw"].isna().mean())}
        (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta, indent=1))
        rate = meta["tier_a_failure_rate"]
        print("\n" + "=" * 70)
        print(f"GATE - Phase 3: Tier-A failure rate {rate:.2%} "
              f"({'PASS' if rate < 0.02 else 'FAIL - Section 8 asks for <2% or an explanation'})")
        print("=" * 70)
        return

    lig = ligand_table()
    sid, nsh = (a.shard_id, a.n_shards) if a.shard_id is not None else slurm_shard()
    nsh = nsh or 1
    mine = lig.iloc[np.array_split(np.arange(len(lig)), nsh)[sid]]
    final = OUT / f"shard_{sid:04d}.parquet"
    if final.exists() and "_site_dist" in pq_columns(final):
        log(f"shard {sid} already complete - skipping")
        return

    ## CHECKPOINTED. The first run wrote nothing until the very end, so every task
    ## that hit the wall time lost all of its work. Now a part file lands every
    ## CHUNK ligands and a resubmitted task resumes after the last one.
    parts = sorted(OUT.glob(f"shard_{sid:04d}_part*.parquet"))
    done = (set(pd.concat([pd.read_parquet(p, columns=["ligand_idx"]) for p in parts])
                .ligand_idx) if parts else set())
    todo = mine[~mine.ligand_idx.isin(done)]
    threads = LP.n_threads()
    log(f"shard {sid}/{nsh}: {len(mine):,} ligands at tier {tier}, "
        f"{len(done):,} already done, {threads} threads")

    CHUNK = 500
    rows, t0, t_last, n_last = [], time.time(), time.time(), 0
    for n, r in enumerate(todo.itertuples(), 1):
        try:
            d = LP.compute(r.ligand_smiles_std, tier=tier, n_conf=a.n_conf,
                           xtb_bin=a.xtb_bin, threads=threads)
        except Exception as e:
            d = {k: np.nan for k in LP.tier_names(tier)}
            d |= {"_n_sites": 0, "_site_dist": b"", "_error": type(e).__name__}
        d["ligand_idx"] = int(r.ligand_idx)
        rows.append(d)
        if len(rows) >= CHUNK or n == len(todo):
            k = len(parts)
            tmp = OUT / f"shard_{sid:04d}_part{k:05d}.tmp.parquet"
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            tmp.replace(OUT / f"shard_{sid:04d}_part{k:05d}.parquet")
            parts.append(OUT / f"shard_{sid:04d}_part{k:05d}.parquet")
            rows = []
        if time.time() - t_last > 60:
            ## Rate over the last minute, not since the start: early ligands are
            ## not representative, and a stale ETA is how the 4,500-minute
            ## estimate went unquestioned.
            rate = (n - n_last) / (time.time() - t_last)
            log(f"  {len(done) + n:,}/{len(mine):,}  {rate:.2f}/s  "
                f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min")
            t_last, n_last = time.time(), n

    parts = sorted(OUT.glob(f"shard_{sid:04d}_part*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("ligand_idx", keep="last")
    if len(df) != len(mine):
        raise SystemExit(f"shard {sid}: {len(df):,} rows for {len(mine):,} ligands")
    tmp = final.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(final)
    for p in parts:
        p.unlink()
    log(f"shard {sid} done in {(time.time() - t0) / 60:.1f} min")


def pq_columns(p) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(p).names


if __name__ == "__main__":
    main()
