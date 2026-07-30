#!/usr/bin/env python
"""
Step 01 - Build the unique protein / ligand entity tables + complex map.

Reads:
  * ML_model/combined_50k_filtered.csv   (Job_ID, smiles, features, target)
  * Input/input.csv                      (ID==Job_ID, seq, ...)  ~1.3 GB, streamed

Writes (to taxonomy/output/):
  * unique_proteins.csv   one row per distinct protein sequence
        protein_id, crc64, seq_len, n_complexes, example_job_id, sequence
  * unique_ligands.csv    one row per distinct ligand (by InChIKey)
        ligand_id, inchikey, canonical_smiles, n_complexes, example_job_id, orig_smiles
  * complex_map.csv       one row per Job_ID (the join key back to the model table)
        Job_ID, protein_id, ligand_id, inchikey, crc64, neg_log10_affinity_M

The protein_id / ligand_id are stable surrogate keys so downstream annotation
tables stay small and joinable.
"""
import os
import pandas as pd
from Bio.SeqUtils.CheckSum import crc64
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRAIN_CSV = os.path.join(ROOT, "ML_model", "combined_50k_filtered.csv")
INPUT_CSV = os.path.join(ROOT, "Input", "input.csv")
OUT = os.path.join(HERE, "output")


def log(m):
    print(m, flush=True)


def clean_crc(seq):
    return crc64(seq).replace("CRC-", "")


def main():
    os.makedirs(OUT, exist_ok=True)

    log(f"[01] reading {TRAIN_CSV}")
    train = pd.read_csv(TRAIN_CSV, usecols=["Job_ID", "smiles", "neg_log10_affinity_M"])
    train["Job_ID"] = train["Job_ID"].astype(str)
    wanted = set(train["Job_ID"])
    log(f"[01] {len(train)} complexes, {len(wanted)} Job_IDs")

    # --- stream sequences for wanted Job_IDs ---
    log(f"[01] streaming sequences from {INPUT_CSV}")
    id2seq = {}
    for i, ch in enumerate(pd.read_csv(INPUT_CSV, usecols=["ID", "seq"], chunksize=200_000)):
        ch = ch[ch["ID"].astype(str).isin(wanted)]
        for _id, _seq in zip(ch["ID"].astype(str), ch["seq"]):
            if _id not in id2seq and isinstance(_seq, str):
                id2seq[_id] = _seq.strip()
    log(f"[01] matched sequences for {len(id2seq)}/{len(wanted)} Job_IDs")

    train["sequence"] = train["Job_ID"].map(id2seq)
    n_missing = train["sequence"].isna().sum()
    if n_missing:
        log(f"[01] WARNING {n_missing} rows have no sequence")

    # --- unique proteins ---
    train["crc64"] = train["sequence"].map(lambda s: clean_crc(s) if isinstance(s, str) else None)
    prot = (train.dropna(subset=["sequence"])
                 .groupby("crc64")
                 .agg(seq_len=("sequence", lambda s: len(s.iloc[0])),
                      n_complexes=("Job_ID", "size"),
                      example_job_id=("Job_ID", "first"),
                      sequence=("sequence", "first"))
                 .reset_index())
    prot = prot.sort_values("n_complexes", ascending=False).reset_index(drop=True)
    prot.insert(0, "protein_id", ["P%05d" % (i + 1) for i in range(len(prot))])
    prot.to_csv(os.path.join(OUT, "unique_proteins.csv"), index=False)
    log(f"[01] wrote unique_proteins.csv ({len(prot)} unique proteins)")

    crc2pid = dict(zip(prot["crc64"], prot["protein_id"]))

    # --- unique ligands (canonical SMILES + InChIKey) ---
    log("[01] canonicalizing ligands with RDKit ...")
    uniq_smiles = train["smiles"].dropna().unique()
    rows = []
    bad = 0
    for s in uniq_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            rows.append((s, None, None))
            continue
        rows.append((s, Chem.MolToSmiles(m), Chem.MolToInchiKey(m)))
    sm = pd.DataFrame(rows, columns=["orig_smiles", "canonical_smiles", "inchikey"])
    log(f"[01] {bad} SMILES failed to parse")
    train = train.merge(sm, left_on="smiles", right_on="orig_smiles", how="left")

    # group by inchikey where available, else by orig smiles
    train["lig_key"] = train["inchikey"].fillna("SMI:" + train["smiles"].astype(str))
    lig = (train.groupby("lig_key")
                .agg(inchikey=("inchikey", "first"),
                     canonical_smiles=("canonical_smiles", "first"),
                     orig_smiles=("smiles", "first"),
                     n_complexes=("Job_ID", "size"),
                     example_job_id=("Job_ID", "first"))
                .reset_index())
    lig = lig.sort_values("n_complexes", ascending=False).reset_index(drop=True)
    lig.insert(0, "ligand_id", ["L%05d" % (i + 1) for i in range(len(lig))])
    lig.to_csv(os.path.join(OUT, "unique_ligands.csv"), index=False)
    log(f"[01] wrote unique_ligands.csv ({len(lig)} unique ligands)")

    key2lid = dict(zip(lig["lig_key"], lig["ligand_id"]))

    # --- complex map ---
    train["protein_id"] = train["crc64"].map(crc2pid)
    train["ligand_id"] = train["lig_key"].map(key2lid)
    cmap = train[["Job_ID", "protein_id", "ligand_id", "inchikey", "crc64",
                  "neg_log10_affinity_M"]]
    cmap.to_csv(os.path.join(OUT, "complex_map.csv"), index=False)
    log(f"[01] wrote complex_map.csv ({len(cmap)} rows)")


if __name__ == "__main__":
    main()
