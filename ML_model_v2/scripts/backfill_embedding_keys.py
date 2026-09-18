#!/usr/bin/env python
"""One-time: write key files for embedding caches built before keys existed.

    python scripts/backfill_embedding_keys.py     # RUN BEFORE `get_more_data.py merge`

Why this exists, and why the ordering matters
---------------------------------------------
`train_savant_v2.ipynb` caches ligand/protein embeddings as bare `.npy` matrices with no
record of which text produced which row. That is fine until the corpus changes size - then
the notebook can only compare row counts, sees a mismatch, and re-embeds ~1.17M ligands
from scratch (the one step that genuinely needs a GPU).

The benchmark-gate filter added to `get_more_data.py` removes 447 ligands, so the next
re-run WOULD trigger exactly that. But the new ligand set is a strict subset of the old
one, so the existing embeddings are perfectly reusable if we can say which row is which.

This script reconstructs the old ordering - exactly as the notebook derived it, by
factorising over `input.csv` + the CURRENT `expanded_kd_ki.csv` - and writes
`<slug>_keys.npy` next to each cache. After that the notebook subsets instead of
re-embedding, and the whole re-run is CPU-viable.

**Run this BEFORE `get_more_data.py merge`** - or, if merge already ran, recover first.

Already re-ran merge? The ordering is still recoverable
-------------------------------------------------------
`merge` is deterministic given the raw sources (`chembl_activities_raw.jsonl`, the
BindingDB TSVs, `uniprot_sequences.csv`), and those are untouched by it. So the pre-filter
file can be regenerated without clobbering the good filtered one:

    python scripts/get_more_data.py merge --skip-benchmark-filter \\
        --out data/expanded_kd_ki_prefilter.csv
    python scripts/backfill_embedding_keys.py --extra data/expanded_kd_ki_prefilter.csv
    rm data/expanded_kd_ki_prefilter.csv        # optional, only needed for the backfill

The regenerated file reproduces the original ordering because the collapse is a
`groupby(["seq","smiles"])`, which pandas emits in sorted key order - the same order every
time from the same inputs. The script verifies this itself: it refuses to write a key file
whose length disagrees with the cache it would be labelling, so a bad reconstruction fails
loudly instead of silently mislabelling every row.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
SRC = REPO / "input" / "input.csv"
EXTRA = ROOT / "data" / "expanded_kd_ki.csv"
EMB = ROOT / "work" / "emb"

MIN_PKD, MAX_PKD = 4.0, 12.0
CENSOR_CAPS_NM = [10000.0, 100000.0, 1000000.0, 10000000.0]
SLUGS = {"chemberta-mlm77": "ligand", "esmc-300m": "protein"}
HF = {"chemberta-mlm77": "DeepChem/ChemBERTa-77M-MLM",
      "esmc-300m": "Synthyra/ESMplusplus_small"}
MAXLEN = {"chemberta-mlm77": 256, "esmc-300m": 1024}


def verify_order(slug, keys, emb_path, hf, n_samples, min_cos=0.99):
    """Re-embed a random sample and check each lands on the row the key file claims.

    This is the check that matters. The reconstruction could yield the right SET of texts
    in the wrong ORDER, in which case the key file would pair every row with the wrong
    molecule - silently, since reindex() would still find every text. Comparing an actual
    freshly-computed embedding against the cached row at that index is the only way to
    catch it. Cosine rather than exact equality because the cache may have been written on
    a GPU and this check may run on CPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    rng = np.random.default_rng(0)
    idx = rng.choice(len(keys), size=min(n_samples, len(keys)), replace=False)
    sample = [str(keys[i]) for i in idx]

    a = np.load(emb_path, mmap_mode="r")
    cached_rows = np.array(a[idx], dtype=np.float32)
    mm = getattr(a, "_mmap", None)
    del a
    if mm is not None:
        mm.close()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  {slug}: verifying {len(sample)} sampled rows on {dev} "
        f"(re-embedding to confirm the ordering)")
    mdl = AutoModel.from_pretrained(hf, trust_remote_code=True).to(dev).eval()
    try:
        tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    except Exception:
        tok = getattr(mdl, "tokenizer", None)
    with torch.inference_mode():
        b = tok(sample, padding=True, truncation=True, max_length=MAXLEN[slug],
                return_tensors="pt")
        b = {k: v.to(dev) for k, v in b.items()}
        h = mdl(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        fresh = ((h * m).sum(1) / m.sum(1).clamp_min(1)).float().cpu().numpy()
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()

    num = (fresh * cached_rows).sum(1)
    den = np.linalg.norm(fresh, axis=1) * np.linalg.norm(cached_rows, axis=1)
    cos = num / np.maximum(den, 1e-12)
    bad = int((cos < min_cos).sum())
    log(f"    cosine to cached row: min {cos.min():.4f}  median {np.median(cos):.4f}  "
        f"({bad} of {len(cos)} below {min_cos})")
    return bad == 0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_and_clean(path):
    """Byte-for-byte the notebook's own cleaning, so the factorisation order matches."""
    cols = ["seq", "smiles", "neg_log10_affinity_M"]
    have_kd = "affinity_Kd_nM" in pd.read_csv(path, nrows=0).columns
    if have_kd:
        cols.append("affinity_Kd_nM")
    raw = pd.read_csv(path, usecols=cols)
    if have_kd:
        cap = np.zeros(len(raw), bool)
        for c in CENSOR_CAPS_NM:
            cap |= np.isclose(raw.affinity_Kd_nM.to_numpy(np.float64), c, rtol=1e-6)
    else:
        cap = np.zeros(len(raw), bool)
    meas = raw[~cap]
    meas = meas[meas.neg_log10_affinity_M.between(MIN_PKD, MAX_PKD)].reset_index(drop=True)
    log(f"  {path.name}: {len(raw):,} rows -> {len(meas):,} measurements, "
        f"{int(cap.sum()):,} censor caps")
    return meas


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", default=str(EXTRA),
                    help="the expanded file the CACHES were built from. If merge has "
                         "already been re-run with the benchmark filter, this must point "
                         "at a regenerated PRE-filter copy, not the current one - see the "
                         "recovery recipe in the module docstring.")
    ap.add_argument("--verify", type=int, default=16,
                    help="re-embed this many sampled rows to confirm the reconstructed "
                         "ordering actually matches the cache (0 disables - not advised: "
                         "a same-set-wrong-order reconstruction silently mislabels every "
                         "row and nothing downstream would notice)")
    args = ap.parse_args()
    extra = Path(args.extra)

    if not EMB.exists():
        sys.exit(f"{EMB} not found - nothing to backfill")
    if not extra.exists():
        sys.exit(f"{extra} not found.\n\n"
                 "If merge already overwrote expanded_kd_ki.csv with the filtered version,\n"
                 "regenerate the pre-filter copy (non-destructively) and point at it:\n\n"
                 "    python scripts/get_more_data.py merge --skip-benchmark-filter \\\n"
                 "        --out data/expanded_kd_ki_prefilter.csv\n"
                 "    python scripts/backfill_embedding_keys.py \\\n"
                 "        --extra data/expanded_kd_ki_prefilter.csv\n")

    log(f"reconstructing the ordering the caches were built with, from {extra.name}")
    meas = pd.concat([load_and_clean(SRC), load_and_clean(extra)], ignore_index=True)
    li = pd.factorize(meas.smiles)[0]
    pi = pd.factorize(meas.seq)[0]
    meas["li"], meas["pi"] = li, pi
    ligands = meas.drop_duplicates("li").sort_values("li").smiles.to_numpy()
    proteins = meas.drop_duplicates("pi").sort_values("pi").seq.to_numpy()
    log(f"reconstructed: {len(ligands):,} ligands, {len(proteins):,} proteins")

    order = {"ligand": ligands, "protein": proteins}
    wrote = 0
    for slug, kind in SLUGS.items():
        p = EMB / f"{slug}.npy"
        kp = EMB / f"{slug}_keys.npy"
        if not p.exists():
            log(f"  {slug}: no cache at {p.name}, skipping")
            continue
        if kp.exists():
            log(f"  {slug}: {kp.name} already exists, leaving it alone")
            continue
        n = np.load(p, mmap_mode="r").shape[0]
        keys = order[kind]
        if n != len(keys):
            log(f"  {slug}: cache has {n:,} rows but reconstruction gives {len(keys):,} "
                f"{kind}s - MISMATCH, refusing to write a key file that would mislabel "
                f"every row. This cache must be rebuilt on a GPU.")
            continue

        ## A length match is NOT enough. The reconstruction could produce the same SET of
        ## texts in a different ORDER - reindex would then succeed and silently pair every
        ## ligand with another ligand's embedding, which is far worse than re-embedding.
        ## So actually re-embed a sample and check it against the row the key file claims.
        if args.verify > 0:
            ok = verify_order(slug, keys, p, HF[slug], args.verify)
            if not ok:
                log(f"  {slug}: ORDER VERIFICATION FAILED - not writing a key file. "
                    f"The reconstruction does not match the cache; rebuild on a GPU.")
                continue

        np.save(kp, np.asarray(keys, object))
        log(f"  {slug}: wrote {kp.name} ({len(keys):,} {kind}s) - cache is now "
            f"subsettable")
        wrote += 1

    if wrote:
        log(f"\n{wrote} key file(s) written. The notebook will now SUBSET these caches "
            "when the corpus shrinks, instead of re-embedding - so the re-run needs no GPU.")
        log("Safe to run `get_more_data.py merge` now.")
    else:
        log("\nNo key files written - see the messages above.")


if __name__ == "__main__":
    main()
