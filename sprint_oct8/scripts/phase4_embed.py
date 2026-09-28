#!/usr/bin/env python3
"""Phase 4: per-residue protein embedding cache (PLAN.md Section 8, Phase 4).

One cache per `plm.mode`, in fp16, RAGGED - a contiguous (total_residues, dim)
memmap plus an offset table, not a padded (n_proteins, L_max, dim) cube. With 6,176
proteins and a 1,022-residue window the padded form is 15.7 GB of which half is
padding; the ragged form memmaps off scratch and the dataloader never holds it.

Per-residue, not pooled. Pooling was the previous project's protein representation
and it is precisely what the pocket head and the physics must not have: a
7,680-dimensional pooled vector says which family a protein belongs to and nothing
about which residues line the pocket. That representation scored per-target rho
0.158 against 0.325 for a protein-blind control.

Gate: deterministic re-extraction matches the cache on a random sample; loader
throughput measured on the target filesystem; storage estimated and confirmed if
over 100 GB.

    python scripts/phase4_embed.py --estimate -o plm=bottleneck
    sbatch --array=0-11 slurm/40_embed_protein.sbatch
    python scripts/phase4_embed.py --merge -o plm=bottleneck
    python scripts/phase4_embed.py --verify -o plm=bottleneck
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

from pfpl import conf, paths, preflight  # noqa: E402
from pfpl.features.cache import content_hash, slurm_shard  # noqa: E402
from pfpl.features.embed_esm import (MAX_LEN, MODELS, POCKET_EXTRA_DIM,  # noqa: E402
                                     Embedder, model_id, pocket_extra_features)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def store_dir(cfg) -> Path:
    mode = str(cfg.plm.mode)
    key = content_hash({"model": model_id(mode), "max_len": MAX_LEN,
                        "corpus": _corpus_checksum()})
    return paths.CACHE / f"plm_{mode}_{key}"


def _corpus_checksum() -> str:
    p = paths.WORK / "proteins.parquet"
    if not p.exists():
        raise SystemExit(f"{p} missing - run scripts/phase1_harmonize.py first")
    prot = pd.read_parquet(p, columns=["protein_idx", "sequence_hash"])
    return content_hash(prot.sequence_hash.tolist())


def proteins() -> pd.DataFrame:
    return pd.read_parquet(paths.WORK / "proteins.parquet").sort_values("protein_idx")


def estimate(cfg) -> None:
    prot = proteins()
    lens = prot.sequence.str.len().clip(upper=MAX_LEN)
    dim = {"esm2_650m_full": 1280, "esm2_bottleneck": 1280, "esm2_pocket_window": 1280,
           "esmc_300m": 960, "esm2_3b": 2560}.get(str(cfg.plm.mode), 1280)
    gb = lens.sum() * dim * 2 / 1e9
    print(f"  {len(prot):,} proteins, {int(lens.sum()):,} residues after the "
          f"{MAX_LEN}-aa window")
    print(f"  {int((prot.sequence.str.len() > MAX_LEN).sum()):,} sequences truncated")
    print(f"  plm.mode={cfg.plm.mode}  dim={dim}  fp16")
    print(f"  embeddings: {gb:.1f} GB   + pocket_extra {lens.sum() * POCKET_EXTRA_DIM * 2 / 1e9:.2f} GB")
    print(f"  cache dir : {store_dir(cfg)}")
    if gb > 100:
        print("\n  OVER 100 GB. Section 8 Phase 4 asks you to confirm the storage")
        print("  budget with the user before launching the array.")


def run_shard(cfg, shard_id: int, n_shards: int, batch_tokens: int) -> None:
    preflight.warn_if_cpu("protein embedding extraction")
    prot = proteins()
    out = store_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    sp = out / f"shard_{shard_id:04d}.npz"
    if sp.exists() and sp.stat().st_size > 0:
        log(f"shard {shard_id} already present - skipping (idempotent by design)")
        return

    idx = np.array_split(np.arange(len(prot)), n_shards)[shard_id]
    mine = prot.iloc[idx]
    ## Sort by length so a batch is not 90% padding; the original order is restored
    ## by `index` at merge time.
    mine = mine.assign(_len=mine.sequence.str.len()).sort_values("_len")
    log(f"shard {shard_id}/{n_shards}: {len(mine):,} proteins "
        f"({mine._len.min()}-{mine._len.max()} aa)")

    emb = Embedder(str(cfg.plm.mode), want_contacts=True)
    log(f"model {emb.name} dim {emb.dim} on {emb.device}")

    vecs, extras, lengths, order, trunc = [], [], [], [], 0
    batch, tokens, t0 = [], 0, time.time()

    def flush():
        nonlocal batch, tokens, trunc
        if not batch:
            return
        res = emb.encode([r.sequence for r in batch])
        for r, d in zip(batch, res):
            vecs.append(d["emb"])
            extras.append(pocket_extra_features(
                d.get("contact_centrality"), r.sequence[:MAX_LEN]).astype(np.float16))
            lengths.append(d["length"])
            order.append(int(r.protein_idx))
            trunc += int(d["truncated"])
        batch, tokens = [], 0

    for n, r in enumerate(mine.itertuples(), 1):
        batch.append(r)
        tokens += min(len(r.sequence), MAX_LEN)
        if tokens >= batch_tokens:
            flush()
            if n % 200 < len(mine):
                rate = n / (time.time() - t0)
                log(f"  {n}/{len(mine)}  {rate:.1f} prot/s  "
                    f"eta {(len(mine) - n) / max(rate, 1e-6) / 60:.0f} min")
    flush()

    np.savez(out / f"shard_{shard_id:04d}.tmp.npz",
             emb=np.concatenate(vecs).astype(np.float16),
             extra=np.concatenate(extras).astype(np.float16),
             lengths=np.asarray(lengths, np.int32),
             index=np.asarray(order, np.int64),
             truncated=np.asarray([trunc], np.int64))
    (out / f"shard_{shard_id:04d}.tmp.npz").replace(sp)
    log(f"shard {shard_id} done: {len(order):,} proteins, {trunc} truncated, "
        f"{(time.time() - t0) / 60:.1f} min")


def merge(cfg) -> None:
    out = store_dir(cfg)
    shards = sorted(out.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {out}")
    prot = proteins()
    parts = [np.load(s) for s in shards]
    all_idx = np.concatenate([p["index"] for p in parts])
    if len(np.unique(all_idx)) != len(prot):
        missing = sorted(set(range(len(prot))) - set(all_idx.tolist()))
        raise SystemExit(
            f"{len(missing)} of {len(prot)} proteins absent from the shards "
            f"(first: {missing[:8]}).\n"
            "  A merge with holes silently shortens the array and every protein_idx\n"
            "  downstream points at the wrong protein. Re-run the missing shards.\n"
            f"  Have {len(shards)} shard files; check SLURM_ARRAY_TASK_COUNT - "
            "`sbatch --array=9` sets it to 1.")

    dim = parts[0]["emb"].shape[1]
    lengths = np.zeros(len(prot), np.int64)
    for p in parts:
        lengths[p["index"]] = p["lengths"]
    offsets = np.zeros(len(prot) + 1, np.int64)
    offsets[1:] = np.cumsum(lengths)
    total = int(offsets[-1])

    log(f"writing {total:,} x {dim} fp16 ({total * dim * 2 / 1e9:.2f} GB)")
    mm = np.memmap(out / "emb.f16", dtype=np.float16, mode="w+", shape=(total, dim))
    ex = np.memmap(out / "pocket_extra.f16", dtype=np.float16, mode="w+",
                   shape=(total, POCKET_EXTRA_DIM))
    for p in parts:
        cur = 0
        for pi, ln in zip(p["index"], p["lengths"]):
            a = int(offsets[pi])
            mm[a:a + ln] = p["emb"][cur:cur + ln]
            ex[a:a + ln] = p["extra"][cur:cur + ln]
            cur += ln
    mm.flush(); ex.flush()
    np.save(out / "offsets.npy", offsets)
    np.save(out / "lengths.npy", lengths)
    meta = {"plm_mode": str(cfg.plm.mode), "model": model_id(str(cfg.plm.mode)),
            "dim": int(dim), "n_proteins": int(len(prot)), "total_residues": total,
            "max_len": MAX_LEN, "n_shards": len(shards),
            "n_truncated": int(sum(int(p["truncated"][0]) for p in parts)),
            "bytes": int(total * dim * 2), "pocket_extra_dim": POCKET_EXTRA_DIM}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    ## A stable pointer so configs and scripts do not carry the content hash around.
    (paths.WORK / f"plm_{cfg.plm.mode}.path").write_text(str(out))
    print(json.dumps(meta, indent=1))
    log(f"merged -> {out}")


def verify(cfg, n: int = 6) -> None:
    """Phase 4 gate: deterministic re-extraction must match the cache."""
    from pfpl.data.datasets import ProteinStore
    store = ProteinStore(store_dir(cfg))
    prot = proteins()
    emb = Embedder(str(cfg.plm.mode), want_contacts=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(prot), size=min(n, len(prot)), replace=False)
    ok = True
    for i in pick:
        want, _ = store.get(int(prot.protein_idx.iat[int(i)]))
        got = emb.encode([prot.sequence.iat[int(i)]])[0]["emb"].astype(np.float32)
        if got.shape != want.shape:
            print(f"  idx {i}: shape {got.shape} vs cached {want.shape}")
            ok = False
            continue
        d = float(np.abs(got - want).max())
        print(f"  idx {i:>6}  len {want.shape[0]:>4}  max|delta| {d:.5f}")
        ok &= d < 2e-2          # fp16 round-trip plus nondeterministic matmul order
    t0 = time.time()
    for i in range(min(200, len(prot))):
        store.get(i)
    print(f"  loader throughput: {200 / (time.time() - t0):.0f} proteins/s off "
          f"{store.root}")
    print("\n" + "=" * 70)
    print(f"GATE - Phase 4: re-extraction {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    cfg = conf.from_args(a)

    if str(cfg.plm.mode) == "none":
        print("  plm.mode=none needs no embeddings at all - nothing to do.")
        print(f"  (Other modes: {sorted(MODELS)})")
        return
    if a.estimate:
        return estimate(cfg)
    if a.merge:
        return merge(cfg)
    if a.verify:
        return verify(cfg)
    if a.selfcheck:
        preflight.warn_if_cpu("the embedding selfcheck")
        emb = Embedder(str(cfg.plm.mode), want_contacts=True)
        seq = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
               "FEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFST"
               "LKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFL")
        t0 = time.time()
        r = emb.encode([seq])[0]
        print(f"  {emb.name}  dim {emb.dim}  device {emb.device}")
        print(f"  {len(seq)} residues -> emb {r['emb'].shape}, "
              f"centrality {r['contact_centrality'].shape} in {time.time() - t0:.2f}s")
        assert r["emb"].shape == (len(seq), emb.dim), "per-residue shape is wrong"
        assert np.isfinite(r["contact_centrality"]).all(), "centrality has non-finite values"
        log("SELFCHECK PASSED")
        return

    sid, nsh = (a.shard_id, a.n_shards) if a.shard_id is not None else slurm_shard()
    run_shard(cfg, sid, nsh or 1, a.batch_tokens)


if __name__ == "__main__":
    main()
