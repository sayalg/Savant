#!/usr/bin/env python
"""
Step 01 — Attach protein sequences to the training table.

`Input/input.csv` has columns [ID, seq, smiles, ...] where ID == the Job_ID
(dock_N) used in `combined_50k_filtered.csv`. This script joins the `seq` column
onto the training table on that key and writes:

  * <training_with_seq_csv>  : combined_50k_filtered.csv + a `protein_sequence` column
  * <training_fasta>         : every UNIQUE training protein chain, for the mmseqs DB

input.csv is large (~1.3 GB) so it is streamed in chunks; only rows whose ID is
present in the training table are kept.
"""
import argparse
import pandas as pd

from common import load_config, ensure_dirs, sanitize_seq, split_chains, write_fasta, log


def main(config_path):
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    p = cfg["paths"]
    min_len = cfg["filters"]["min_chain_len"]

    log(f"[01] reading training table: {p['training_csv']}")
    train = pd.read_csv(p["training_csv"])
    id_col = "Job_ID" if "Job_ID" in train.columns else train.columns[0]
    wanted = set(train[id_col].astype(str))
    log(f"[01] training rows={len(train)} unique Job_IDs={len(wanted)} (key column='{id_col}')")

    # Stream input.csv, keep ID->seq for wanted IDs only.
    log(f"[01] streaming sequences from: {p['input_csv']}")
    id2seq = {}
    reader = pd.read_csv(p["input_csv"], usecols=["ID", "seq"], chunksize=100_000)
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["ID"].astype(str).isin(wanted)]
        for _id, _seq in zip(chunk["ID"].astype(str), chunk["seq"]):
            if _id not in id2seq and isinstance(_seq, str):
                id2seq[_id] = _seq
        if (i + 1) % 5 == 0:
            log(f"[01]   processed {(i + 1) * 100_000:,} input rows, matched {len(id2seq):,}")
    log(f"[01] matched sequences for {len(id2seq):,}/{len(wanted):,} Job_IDs")

    train["protein_sequence"] = train[id_col].astype(str).map(id2seq)
    n_missing = train["protein_sequence"].isna().sum()
    if n_missing:
        log(f"[01] WARNING: {n_missing} training rows had no sequence match in input.csv")
    train.to_csv(p["training_with_seq_csv"], index=False)
    log(f"[01] wrote {p['training_with_seq_csv']} ({train.shape[0]} rows, +protein_sequence)")

    # Unique training chains -> FASTA (sanitized).
    chains = set()
    for seq in train["protein_sequence"].dropna():
        for c in split_chains(seq, min_len):
            chains.add(sanitize_seq(c))
    write_fasta(((f"t{i}", c) for i, c in enumerate(sorted(chains))), p["training_fasta"])
    log(f"[01] wrote {p['training_fasta']} ({len(chains):,} unique training chains)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
