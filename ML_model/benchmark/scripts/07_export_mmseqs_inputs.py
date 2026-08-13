#!/usr/bin/env python
"""
Step 07 - Write the two FASTA files the HPC mmseqs run needs.

  work/mmseqs/train_chains.fasta   all unique training protein chains (target DB)
  work/mmseqs/bench_targets.fasta  the benchmark's unique targets (query)

Run this locally, copy work/mmseqs/ to the cluster, run run_mmseqs_tiers.sbatch
there, copy bench_vs_train.m8 back, then run 08_reconcile_mmseqs.py.
"""
from __future__ import annotations

import os

import pandas as pd

from common import OUT, TRAIN_PROTEINS, WORK, ensure_dirs, log, sanitize_seq

MM = os.path.join(WORK, "mmseqs")


def main() -> None:
    ensure_dirs()
    os.makedirs(MM, exist_ok=True)

    tp = pd.read_csv(TRAIN_PROTEINS)
    train_fa = os.path.join(MM, "train_chains.fasta")
    with open(train_fa, "w") as f:
        for pid, seq in tp[["protein_id", "sequence"]].dropna().values:
            s = sanitize_seq(seq)
            if s:
                f.write(f">{pid}\n{s}\n")
    log(f"wrote {train_fa} ({len(tp)} training chains)")

    b = pd.read_csv(os.path.join(OUT, "savant_benchmark_1000.csv"))
    q = b.drop_duplicates("uniprot_id")[["uniprot_id", "target_sequence"]].dropna()
    bench_fa = os.path.join(MM, "bench_targets.fasta")
    with open(bench_fa, "w") as f:
        for acc, seq in q.values:
            f.write(f">{acc}\n{sanitize_seq(seq)}\n")
    log(f"wrote {bench_fa} ({len(q)} benchmark targets)")

    # tiers as shipped, so the cluster result can be diffed against them
    tiers = os.path.join(MM, "shipped_tiers.csv")
    b.drop_duplicates("uniprot_id")[["uniprot_id", "novelty_tier", "gene",
                                     "max_train_protein_containment"]].to_csv(tiers, index=False)
    log(f"wrote {tiers}")
    log(f"copy {MM} to the cluster and run run_mmseqs_tiers.sbatch there")


if __name__ == "__main__":
    main()
