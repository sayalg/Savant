"""Savant v2: CPU-only protein-ligand affinity prediction from a docked complex.

What changed from v1, and why:

  HADDOCK3 removed.  v1 ran topology generation plus energy minimisation per
  complex to produce six energy scalars. Ablated on the 1000-pair benchmark,
  those six were worth +0.012 Pearson r over a plain distance-based contact
  count. They cost minutes per complex and a Linux-only dependency, so they are
  gone. v2 needs numpy, scipy, RDKit and a pickled model.

  Interactions enumerated, not summed.  In their place, a typed interaction
  fingerprint: hydrophobic contacts, hydrogen bonds split by ligand role, ionic
  pairs, pi-stacking split by ring geometry, cation-pi, halogen bonds, metal
  coordination, contacts broken out per residue type, and element-pair counts at
  two distance shells. An interface is hundreds of atom-atom contacts; v1
  compressed that to six numbers and the ablation shows what survived.

  The protein finally enters the feature vector.  v1's training table had no
  protein column at all - only ligand descriptors and interaction energies - so
  two different targets with similar contact energetics were the same input.
  Per-residue-type contacts and pocket composition fix that without needing a
  protein identifier.

  Ligand structure, not just properties.  ECFP4 alongside the eleven
  physicochemical descriptors, measured at +0.035 Pearson r.

  Calibrated uncertainty.  Split-conformal prediction intervals with
  distribution-free coverage, so a prediction arrives with an interval and an
  applicability-domain flag rather than as a bare point estimate.

The binder/nonbinder pre-filter is retained from v1.
"""

__version__ = "2.0.0-dev"

PROTEIN_CHAIN = "A"
LIGAND_CHAIN = "B"
LIGAND_RESNAME = "LIG"
