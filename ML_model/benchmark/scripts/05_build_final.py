#!/usr/bin/env python
"""
Step 05 - Assemble the deliverable and run the QC gate.

Writes output/savant_benchmark_1000.csv plus a FASTA / SMILES sidecar and a
QC report. The QC assertions are hard: if a ligand that overlaps training ever
reaches the output, or the pKd convention drifts, this step fails loudly rather
than shipping a quietly broken benchmark (which is what happened to the
previous build).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from common import (LIG_ANNOT, MAX_LIGAND_TANIMOTO, OUT, WORK, ensure_dirs, log,
                    pkd_from_kd_nm)

ATC_L1 = {
    "A": "Alimentary tract & metabolism", "B": "Blood & blood-forming organs",
    "C": "Cardiovascular system", "D": "Dermatologicals",
    "G": "Genito-urinary & sex hormones", "H": "Systemic hormonal preparations",
    "J": "Anti-infectives (systemic)", "L": "Antineoplastic & immunomodulating",
    "M": "Musculo-skeletal system", "N": "Nervous system",
    "P": "Antiparasitic", "R": "Respiratory system",
    "S": "Sensory organs", "V": "Various",
}

COLUMNS = [
    # identity
    "benchmark_id", "uniprot_id", "gene", "protein_name", "organism",
    "target_sequence", "ligand_smiles", "inchikey",
    # measured affinity (the label)
    "kd_nM", "neg_log10_affinity_M", "deltaG_kcal_per_mol",
    # disjointness from training
    "novelty_tier", "max_train_protein_containment", "closest_train_protein",
    "max_train_ligand_tanimoto",
    # protein taxonomy
    "protein_class", "chembl_l1", "chembl_l2", "family_l1", "ec_primary", "ec_l1_name",
    "chembl_target_id",
    # ligand taxonomy
    "cf_kingdom", "cf_superclass", "cf_class", "cf_subclass", "cf_direct_parent",
    "murcko_scaffold",
    # drug annotation
    "molecule_chembl_id", "ligand_chembl_name", "max_phase", "atc_code",
    "atc_l1", "drug_class_atc",
    # ligand physchem
    "molecular_weight", "logP", "num_h_donors", "num_h_acceptors",
    "num_rotatable_bonds", "tpsa", "num_rings", "num_aromatic_rings",
    "num_heavy_atoms", "formal_charge",
    # provenance
    "source",
]


def main() -> None:
    ensure_dirs()
    sel = pd.read_csv(os.path.join(WORK, "selected.csv"))

    if os.path.exists(LIG_ANNOT):
        lig = pd.read_csv(LIG_ANNOT)
        # `chembl_pref_name` exists on both sides (target name vs molecule name);
        # disambiguate before merging or the protein name silently wins.
        lig = lig.rename(columns={"chembl_pref_name": "ligand_chembl_name"})
        sel = sel.merge(lig, on="canonical_smiles", how="left", suffixes=("", "_lig"))
        log(f"merged ligand taxonomy ({(sel.cf_status == 'ok').sum()}/{len(sel)} ClassyFire ok)")
    else:
        log("WARNING: ligand annotation not found; ChemOnt columns will be empty")
        for c in ("inchikey", "cf_status", "kingdom", "superclass", "cf_class", "subclass",
                  "direct_parent", "molecule_chembl_id", "ligand_chembl_name", "max_phase",
                  "atc_code", "atc_l1"):
            sel[c] = None

    out = sel.rename(columns={
        "acc": "uniprot_id_clean",
        "target_seq": "target_sequence",
        "canonical_smiles": "ligand_smiles",
        "max_train_kmer_containment": "max_train_protein_containment",
        "kingdom": "cf_kingdom",
        "superclass": "cf_superclass",
        "subclass": "cf_subclass",
        "direct_parent": "cf_direct_parent",
    })
    out["uniprot_id"] = out["uniprot_id_clean"]
    out["deltaG_kcal_per_mol"] = (-1.364 * out["neg_log10_affinity_M"]).round(3)
    out["drug_class_atc"] = out["atc_l1"].map(lambda a: ATC_L1.get(a) if isinstance(a, str) else None)
    out["source"] = "BindingDB (TDC Kd table, Harvard Dataverse 4291555)"

    for c in COLUMNS:
        if c not in out.columns:
            out[c] = None
    out = out[COLUMNS].sort_values(["novelty_tier", "protein_class", "neg_log10_affinity_M"])
    out["benchmark_id"] = [f"bm_{i+1:04d}" for i in range(len(out))]

    # ---------------- QC gate ----------------
    fail = []
    n = len(out)
    if n < 950:
        fail.append(f"only {n} pairs")

    bad_pkd = (~np.isclose(out["neg_log10_affinity_M"],
                           pkd_from_kd_nm(out["kd_nM"]).round(3), atol=0.011)).sum()
    if bad_pkd:
        fail.append(f"{bad_pkd} rows where neg_log10_affinity_M != 9 - log10(kd_nM)")

    over = (out["max_train_ligand_tanimoto"] >= MAX_LIGAND_TANIMOTO).sum()
    if over:
        fail.append(f"{over} ligands at/above Tanimoto {MAX_LIGAND_TANIMOTO} to training")

    if out["ligand_smiles"].isna().any() or out["target_sequence"].isna().any():
        fail.append("null sequence or SMILES")
    dupes = out.duplicated(subset=["uniprot_id", "ligand_smiles"]).sum()
    if dupes:
        fail.append(f"{dupes} duplicate (target, ligand) pairs")

    dest = os.path.join(OUT, "savant_benchmark_1000.csv")
    out.to_csv(dest, index=False)
    log(f"wrote {dest} ({n} rows, {len(COLUMNS)} cols)")

    # sidecars for structure prediction / docking tools
    fa = os.path.join(OUT, "savant_benchmark_targets.fasta")
    with open(fa, "w") as f:
        for u, s in out.drop_duplicates("uniprot_id")[["uniprot_id", "target_sequence"]].values:
            f.write(f">{u}\n{s}\n")
    sm = os.path.join(OUT, "savant_benchmark_ligands.smi")
    with open(sm, "w") as f:
        for s, i in out.drop_duplicates("ligand_smiles")[["ligand_smiles", "inchikey"]].values:
            f.write(f"{s}\t{i}\n")
    log(f"wrote {fa} and {sm}")

    write_report(out, fail)
    if fail:
        log("QC FAILED: " + "; ".join(fail))
        sys.exit(1)
    log("QC passed.")


def write_report(out: pd.DataFrame, fail: list) -> None:
    p = os.path.join(OUT, "savant_benchmark_1000_summary.md")
    tier_counts = out["novelty_tier"].value_counts()

    def tbl(s, name, top=None):
        vc = s.value_counts(dropna=False)
        if top:
            vc = vc.head(top)
        lines = [f"| {name} | pairs | share |", "|---|---:|---:|"]
        for k, v in vc.items():
            lines.append(f"| {k} | {v} | {v/len(out)*100:.1f}% |")
        return "\n".join(lines)

    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Savant benchmark - {len(out)} protein-ligand Kd pairs\n\n")
        f.write(f"**QC: {'FAILED - ' + '; '.join(fail) if fail else 'PASSED'}**\n\n")
        f.write(f"- unique targets: {out.uniprot_id.nunique()}\n")
        f.write(f"- unique ligands: {out.ligand_smiles.nunique()} "
                f"({out.murcko_scaffold.nunique()} Bemis-Murcko scaffolds)\n")
        f.write(f"- protein classes: {out.protein_class.nunique()}; "
                f"ChemOnt classes: {out.cf_class.nunique()}\n")
        f.write(f"- pKd: {out.neg_log10_affinity_M.min():.2f} to "
                f"{out.neg_log10_affinity_M.max():.2f} "
                f"(mean {out.neg_log10_affinity_M.mean():.2f}, "
                f"sd {out.neg_log10_affinity_M.std():.2f})\n")
        f.write(f"- source: {out.source.iloc[0]}\n\n")

        f.write("## Disjointness from training\n\n")
        f.write(f"Every ligand is below ECFP4 Tanimoto {MAX_LIGAND_TANIMOTO} to *every* "
                f"training ligand (max observed "
                f"{out.max_train_ligand_tanimoto.max():.3f}, median "
                f"{out.max_train_ligand_tanimoto.median():.3f}), and no exact structure or "
                "InChIKey-skeleton match survives. The novelty tier records how far the "
                "*protein* sits from training:\n\n")
        f.write("| tier | meaning | pairs | targets |\n|---|---|---:|---:|\n")
        meaning = {
            "same_protein": "target in training; ligand is new chemistry",
            "same_family": "new protein, UniProt family seen in training",
            "novel_family": "protein family absent from training",
        }
        for t in ("same_protein", "same_family", "novel_family"):
            sub = out[out.novelty_tier == t]
            f.write(f"| `{t}` | {meaning[t]} | {tier_counts.get(t, 0)} | {sub.uniprot_id.nunique()} |\n")
        f.write("\n")

        f.write("## Protein classes (ChEMBL level 2)\n\n")
        f.write(tbl(out.protein_class, "protein class") + "\n\n")
        f.write("## Ligand structural classes (ClassyFire / ChemOnt)\n\n")
        f.write(tbl(out.cf_class, "ChemOnt class", top=20) + "\n\n")
        f.write("## Drug classes (ATC level 1, where the ligand is a known drug)\n\n")
        f.write(tbl(out.drug_class_atc.dropna(), "ATC class") + "\n\n")
        f.write("## Affinity distribution\n\n")
        bins = pd.cut(out.neg_log10_affinity_M, [0, 5, 6, 7, 8, 9, 20])
        f.write(tbl(bins.astype(str), "pKd range") + "\n")
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
