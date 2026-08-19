"""Train/test splits, including the double-disjoint one.

The benchmark run exposed why this module exists. Both model families were
selected on a protein-grouped held-out split, where they scored r 0.580 (trees)
and 0.598 (attention) - effectively tied, with the transformer marginally ahead.
On the external benchmark, which is disjoint on BOTH axes, they scored 0.327 and
0.269. The internal split did not merely overestimate; it ranked the two models
the wrong way round.

The cause is that grouping by protein alone leaves the ligand axis free. A model
can meet a genuinely new protein in the held-out fold while scoring a ligand
whose near-twin it trained on, and remembered chemistry carries it. The
benchmark allows neither: its ligands are gated at ECFP4 Tanimoto <= 0.50
against every training ligand.

``double_disjoint_split`` reproduces those conditions internally. Hold out a
fraction of proteins AND a fraction of Bemis-Murcko scaffolds, then keep only:

    train  protein NOT held out  AND  scaffold NOT held out
    test   protein IS  held out  AND  scaffold IS  held out

Rows that straddle - new protein but familiar scaffold, or the reverse - are
discarded, because keeping them on either side reintroduces exactly the leak
being measured. That discard is expensive (typically a third of the data) and is
the honest price of the question.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def murcko_scaffolds(smiles: Sequence[str]) -> np.ndarray:
    """Bemis-Murcko scaffold per ligand, as a grouping key.

    Acyclic ligands have no Murcko scaffold; they are pooled under one key
    rather than each becoming its own group. Pooling is the conservative choice
    for a disjointness split - it sends them all to one side, where a per-ligand
    key would let chemically similar acyclic molecules sit on both.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smiles:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=str(smi))
        except Exception:
            scaffold = ""
        out.append(scaffold if scaffold else "ACYCLIC")
    return np.asarray(out, dtype=object)


def protein_split(
    groups: Sequence, test_size: float = 0.25, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Group split on the protein axis only - what v2 shipped with."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    held = set(unique[: max(1, int(len(unique) * test_size))].tolist())

    test = np.array([g in held for g in groups])
    train_idx = np.where(~test)[0]
    test_idx = np.where(test)[0]
    stats = {
        "mode": "protein",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_discarded": 0,
        "proteins_held_out": len(held),
        "scaffolds_held_out": 0,
    }
    return train_idx, test_idx, stats


def double_disjoint_split(
    protein_groups: Sequence,
    scaffolds: Sequence,
    protein_test_frac: float = 0.30,
    scaffold_test_frac: float = 0.30,
    seed: int = 0,
    min_test: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Split disjoint on protein AND ligand scaffold simultaneously.

    Fractions default to 0.30 rather than 0.25 because the test fold is the
    intersection of two held-out sets: at 0.25/0.25 only ~6% of rows qualify,
    and on 47,619 complexes that is a noisy evaluation. At 0.30/0.30 it is ~9%.

    Raises if the test fold ends up smaller than ``min_test``, since a split too
    small to measure anything is worse than no split at all.
    """
    protein_groups = np.asarray(protein_groups)
    scaffolds = np.asarray(scaffolds)
    rng = np.random.RandomState(seed)

    prot_unique = np.unique(protein_groups)
    rng.shuffle(prot_unique)
    prot_held = set(prot_unique[: max(1, int(len(prot_unique) * protein_test_frac))].tolist())

    scaf_unique = np.unique(scaffolds)
    rng.shuffle(scaf_unique)
    scaf_held = set(scaf_unique[: max(1, int(len(scaf_unique) * scaffold_test_frac))].tolist())

    prot_out = np.array([p in prot_held for p in protein_groups])
    scaf_out = np.array([s in scaf_held for s in scaffolds])

    train_idx = np.where(~prot_out & ~scaf_out)[0]
    test_idx = np.where(prot_out & scaf_out)[0]
    discarded = len(protein_groups) - len(train_idx) - len(test_idx)

    if len(test_idx) < min_test:
        raise ValueError(
            f"double-disjoint test fold has only {len(test_idx)} rows "
            f"(minimum {min_test}); raise protein_test_frac / scaffold_test_frac"
        )

    stats = {
        "mode": "protein+scaffold",
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_discarded": int(discarded),
        "pct_discarded": round(100.0 * discarded / len(protein_groups), 1),
        "proteins_held_out": len(prot_held),
        "scaffolds_held_out": len(scaf_held),
        "total_proteins": len(prot_unique),
        "total_scaffolds": len(scaf_unique),
    }
    return train_idx, test_idx, stats


def describe(stats: Dict) -> str:
    line = (f"{stats['mode']} split: {stats['n_train']} train / {stats['n_test']} test")
    if stats.get("n_discarded"):
        line += (f", {stats['n_discarded']} discarded ({stats['pct_discarded']}%) "
                 "as they straddle the held-out sets")
    line += (f"\n  held out {stats['proteins_held_out']}"
             f"{'/' + str(stats['total_proteins']) if 'total_proteins' in stats else ''} proteins")
    if stats.get("scaffolds_held_out"):
        line += (f", {stats['scaffolds_held_out']}/{stats['total_scaffolds']} scaffolds")
    return line
