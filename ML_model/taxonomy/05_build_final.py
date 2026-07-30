#!/usr/bin/env python
"""
Step 05 - Assemble the final taxonomy deliverables.

Protein side (joins steps 01/03/04):
  output/protein_taxonomy.csv   one row per unique protein, with every tree level
      as its own column so you can group at any depth:
        identity/mapping: protein_id, n_complexes, seq_len, pdb_entity, identity,
                          uniprot_id, protein_name, organism, gene
        ChEMBL target class tree: chembl_l1..l6, chembl_class_path
        EC tree:                  ec, ec_l1_name (top class), ec_class..sub_subclass
        Pfam/InterPro (domains):  pfam, pfam_names, interpro, interpro_names
        GO molecular function:    go_mf
        UniProt family:           protein_families, family_l1, family_l2

Ligand side is produced by step 02 (ligand_taxonomy.csv).

Also writes complexes_annotated.csv: complex_map joined to the protein + ligand
top-level labels, so per-complex model performance can be grouped by any level.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

EC_L1 = {"1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
         "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases"}


def _first_ec(ec):
    if not isinstance(ec, str) or not ec.strip():
        return None
    return ec.split(";")[0].strip()


def build_protein():
    prot = pd.read_csv(os.path.join(OUT, "unique_proteins.csv"))[
        ["protein_id", "crc64", "seq_len", "n_complexes"]]
    pm = pd.read_csv(os.path.join(OUT, "protein_pdb_map.csv"))[
        ["protein_id", "pdb_entity", "identity", "cutoff", "status"]]
    df = prot.merge(pm, on="protein_id", how="left")

    if os.path.exists(os.path.join(OUT, "entity_annotation.csv")):
        ea = pd.read_csv(os.path.join(OUT, "entity_annotation.csv"))
        df = df.merge(ea, on="pdb_entity", how="left", suffixes=("", "_ent"))
    if os.path.exists(os.path.join(OUT, "uniprot_annotation.csv")):
        ua = pd.read_csv(os.path.join(OUT, "uniprot_annotation.csv"))
        df = df.merge(ua, on="uniprot_id", how="left", suffixes=("", "_up"))
    if os.path.exists(os.path.join(OUT, "chembl_class.csv")):
        cc = pd.read_csv(os.path.join(OUT, "chembl_class.csv"))
        df = df.merge(cc, on="uniprot_id", how="left", suffixes=("", "_ch"))

    # --- ChEMBL class levels ---
    def split_ch(p, i):
        if not isinstance(p, str):
            return None
        parts = p.split(" > ")
        return parts[i] if i < len(parts) else None
    for i in range(6):
        df[f"chembl_l{i+1}"] = df["chembl_class_path"].map(lambda p: split_ch(p, i))

    # --- EC levels ---
    ec1 = df["ec"].map(_first_ec)
    df["ec_primary"] = ec1
    df["ec_l1_name"] = ec1.map(lambda e: EC_L1.get(e.split(".")[0]) if isinstance(e, str) else None)
    for lvl, nm in [(1, "ec_class"), (2, "ec_subclass"), (3, "ec_sub_subclass")]:
        df[nm] = ec1.map(lambda e, L=lvl: ".".join(e.split(".")[:L]) if isinstance(e, str) else None)

    # --- UniProt family levels ---
    def fam(p, i):
        if not isinstance(p, str) or not p.strip():
            return None
        parts = [x.strip() for x in p.split(",")]
        return parts[i] if i < len(parts) else None
    df["family_l1"] = df["protein_families"].map(lambda p: fam(p, 0))
    df["family_l2"] = df["protein_families"].map(lambda p: fam(p, 1))

    cols = ["protein_id", "n_complexes", "seq_len", "pdb_entity", "identity", "cutoff",
            "status", "uniprot_id", "protein_name", "organism", "gene",
            "chembl_class_path", "chembl_l1", "chembl_l2", "chembl_l3", "chembl_l4",
            "chembl_l5", "chembl_l6",
            "ec_primary", "ec_l1_name", "ec_class", "ec_subclass", "ec_sub_subclass",
            "pfam", "pfam_names", "interpro", "interpro_names",
            "go_mf", "protein_families", "family_l1", "family_l2", "description"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("n_complexes", ascending=False)
    df.to_csv(os.path.join(OUT, "protein_taxonomy.csv"), index=False)
    print(f"[05] wrote protein_taxonomy.csv ({len(df)} proteins)")
    mapped = df["uniprot_id"].notna().sum()
    print(f"[05]   mapped to UniProt: {mapped}/{len(df)} "
          f"({100*mapped/len(df):.1f}%)")
    return df


def build_complexes(prot):
    cmap = pd.read_csv(os.path.join(OUT, "complex_map.csv"))
    pcols = ["protein_id", "uniprot_id", "protein_name", "organism",
             "chembl_l1", "chembl_l2", "ec_l1_name", "family_l1"]
    pcols = [c for c in pcols if c in prot.columns]
    df = cmap.merge(prot[pcols], on="protein_id", how="left")
    if os.path.exists(os.path.join(OUT, "ligand_taxonomy.csv")):
        lt = pd.read_csv(os.path.join(OUT, "ligand_taxonomy.csv"))
        lcols = [c for c in ["ligand_id", "kingdom", "superclass", "cf_class", "subclass"]
                 if c in lt.columns]
        df = df.merge(lt[lcols], on="ligand_id", how="left")
    df.to_csv(os.path.join(OUT, "complexes_annotated.csv"), index=False)
    print(f"[05] wrote complexes_annotated.csv ({len(df)} complexes)")


if __name__ == "__main__":
    prot = build_protein()
    build_complexes(prot)
