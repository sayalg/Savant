#!/usr/bin/env python
"""
Step 08 - Reconcile the shipped novelty tiers against a real mmseqs search.

Reads the .m8 produced by run_mmseqs_tiers.sbatch and recomputes, per benchmark
target, the best identity to any training chain. Adds two columns to the
benchmark and reports every disagreement with the locally assigned tier.

    python 08_reconcile_mmseqs.py --m8 ../work/mmseqs/bench_vs_train.m8 [--apply]

Without --apply nothing is overwritten; you get the report only.

mmseqs-based tiers:
    identity >= 0.90                 -> same_protein
    0.30 <= identity < 0.90          -> same_family   (homolog, detected by alignment)
    no hit at >=0.30 id / >=0.80 cov -> novel_family
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from common import OUT, ensure_dirs, log

COLS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]


def main(m8: str, apply: bool) -> None:
    ensure_dirs()
    bpath = os.path.join(OUT, "savant_benchmark_1000.csv")
    b = pd.read_csv(bpath)

    if not os.path.exists(m8):
        raise SystemExit(f"no such file: {m8} (run run_mmseqs_tiers.sbatch on the HPC first)")
    if os.path.getsize(m8) == 0:
        raise SystemExit(
            f"{m8} is empty. An empty result means the search FAILED (bad container "
            "path, missing FASTA, wrong working directory) - it does NOT mean the "
            "benchmark has no homology to training. Check the SLURM logs and re-run.")

    hits = pd.read_csv(m8, sep="\t", names=COLS)
    log(f"{len(hits)} hits over {hits['query'].nunique()} benchmark targets")

    # Sanity floor: `same_protein` targets are in training by accession, so mmseqs
    # MUST find near-identical hits for them. If it does not, the search is broken
    # and every downstream 'novel' verdict would be a false negative.
    n_targets = b["uniprot_id"].nunique()
    expect = set(b.loc[b["novelty_tier"] == "same_protein", "uniprot_id"])
    found = set(hits["query"]) & expect
    if expect and len(found) < 0.5 * len(expect):
        raise SystemExit(
            f"only {len(found)}/{len(expect)} known same_protein targets got any hit "
            f"({hits['query'].nunique()}/{n_targets} targets hit overall). The search "
            "did not run correctly - refusing to write a report that would read as "
            "'no homology found'. Check the SLURM logs and the container path.")

    best = (hits.sort_values("fident", ascending=False)
                .drop_duplicates("query")
                .set_index("query"))
    b["mmseqs_best_identity"] = b["uniprot_id"].map(best["fident"]).fillna(0.0).round(3)
    b["mmseqs_closest_train_chain"] = b["uniprot_id"].map(best["target"])

    def tier(i):
        if i >= 0.90:
            return "same_protein"
        if i >= 0.30:
            return "same_family"
        return "novel_family"

    b["mmseqs_tier"] = b["mmseqs_best_identity"].map(tier)

    t = b.drop_duplicates("uniprot_id")
    log(f"targets: {len(t)}")
    print()
    print("shipped tier vs mmseqs tier (unique targets):")
    print(pd.crosstab(t["novelty_tier"], t["mmseqs_tier"]).to_string())
    print()
    print("by pairs:")
    print(pd.crosstab(b["novelty_tier"], b["mmseqs_tier"]).to_string())

    dis = t[t["novelty_tier"] != t["mmseqs_tier"]]
    print()
    if len(dis):
        log(f"{len(dis)} targets disagree ({dis.shape[0]/len(t)*100:.1f}%)")
        print(dis[["uniprot_id", "gene", "novelty_tier", "mmseqs_tier",
                   "max_train_protein_containment", "mmseqs_best_identity"]]
              .sort_values("mmseqs_best_identity", ascending=False).to_string(index=False))
    else:
        log("no disagreements: local tiers confirmed exactly")

    # The one that matters: anything the local method called novel but mmseqs
    # says is a training protein is a genuine leak.
    leak = t[(t["novelty_tier"] != "same_protein") & (t["mmseqs_best_identity"] >= 0.90)]
    print()
    if len(leak):
        log(f"WARNING: {len(leak)} targets tiered non-same_protein but are >=90% identical "
            "to a training chain — treat these as leaked")
        print(leak[["uniprot_id", "gene", "novelty_tier", "mmseqs_best_identity"]].to_string(index=False))
    else:
        log("no leaks: nothing tiered below same_protein is >=90% identical to training")

    rep = os.path.join(OUT, "mmseqs_tier_reconciliation.csv")
    t[["uniprot_id", "gene", "novelty_tier", "mmseqs_tier",
       "max_train_protein_containment", "mmseqs_best_identity",
       "mmseqs_closest_train_chain"]].to_csv(rep, index=False)
    log(f"wrote {rep}")

    if apply:
        b["novelty_tier_local"] = b["novelty_tier"]
        b["novelty_tier"] = b["mmseqs_tier"]
        b.to_csv(bpath, index=False)
        log(f"applied mmseqs tiers to {bpath} (previous kept as novelty_tier_local)")
    else:
        log("report only; re-run with --apply to adopt the mmseqs tiers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m8", default=os.path.join("..", "work", "mmseqs", "bench_vs_train.m8"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    main(a.m8, a.apply)
