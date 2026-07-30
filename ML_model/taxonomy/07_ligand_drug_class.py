#!/usr/bin/env python
"""
Step 07 - Derive a target-based "drug class" (mechanistic class) per complex.

ClassyFire (ligand_taxonomy.csv) is purely STRUCTURAL and has no
inhibitor/agonist information. This step derives a functional drug-class label
from the *target* the ligand is bound to, because this is a binding (Kd) dataset:
a binder of an enzyme active site is, in practice, an inhibitor.

  Enzyme targets           -> "<L2> inhibitor"   (Kinase inhibitor, Protease inhibitor, ...)
  Nuclear receptor / TF    -> "Nuclear receptor modulator"
  Membrane receptor (GPCR) -> "Receptor ligand (agonist/antagonist)"
  Ion channel              -> "Ion channel modulator"
  Epigenetic regulator     -> "Epigenetic modulator"
  Transporter              -> "Transporter inhibitor/modulator"
  (other)                  -> "<L1> ligand"

This is an INFERENCE (binding != proven inhibition; receptor direction unknown),
so the column is named `target_based_drug_class` and marked putative. For the
subset with curated data, ChEMBL action_type is the ground truth (~15-20%
coverage) - run step 08 if you want that overlay.

Adds the column to complexes_annotated.csv and writes a distribution summary.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

ENZYME_L2_INHIBITOR = {
    "Kinase": "Kinase inhibitor",
    "Protease": "Protease inhibitor",
    "Phosphodiesterase": "Phosphodiesterase inhibitor",
    "Phosphatase": "Phosphatase inhibitor",
    "Cytochrome P450": "Cytochrome P450 inhibitor",
    "Lyase": "Lyase inhibitor",
    "Hydrolase": "Hydrolase inhibitor",
    "Transferase": "Transferase inhibitor",
    "Oxidoreductase": "Oxidoreductase inhibitor",
    "Isomerase": "Isomerase inhibitor",
    "Ligase": "Ligase inhibitor",
}


def derive(l1, l2):
    if not isinstance(l1, str):
        return None
    if l1 == "Enzyme":
        if isinstance(l2, str) and l2 in ENZYME_L2_INHIBITOR:
            return ENZYME_L2_INHIBITOR[l2]
        return "Enzyme inhibitor"
    if l1 in ("Transcription factor", "Other nuclear protein"):
        return "Nuclear receptor / TF modulator"
    if l1 == "Membrane receptor":
        return "Receptor ligand (agonist/antagonist)"
    if l1 == "Ion channel":
        return "Ion channel modulator"
    if l1 == "Epigenetic regulator":
        return "Epigenetic modulator"
    if l1 == "Transporter":
        return "Transporter inhibitor/modulator"
    return f"{l1} ligand"


def main():
    c = pd.read_csv(os.path.join(OUT, "complexes_annotated.csv"))
    c["target_based_drug_class"] = [derive(a, b) for a, b in zip(c["chembl_l1"], c["chembl_l2"])]
    c.to_csv(os.path.join(OUT, "complexes_annotated.csv"), index=False)

    lab = c["target_based_drug_class"]
    n = len(c)
    cov = lab.notna().sum()
    print(f"[07] added target_based_drug_class  coverage {cov}/{n} ({100*cov/n:.1f}%)")
    summ = (c.groupby("target_based_drug_class").size()
              .sort_values(ascending=False).rename("n_complexes").reset_index())
    summ.to_csv(os.path.join(OUT, "drug_class_summary.csv"), index=False)
    print("\n=== target-based drug class (complexes) ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
