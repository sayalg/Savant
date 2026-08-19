"""Distance-biased cross-attention between ligand atoms and pocket residues.

The gradient-boosted models see the interface as a bag of counts. This sees it
as what it is: a set of ligand atoms and a set of pocket residues, with every
atom free to attend to every residue, biased by how far apart they actually are.

    ligand atoms  (M, atom features)          RDKit typing + geometry
    pocket residues (K, ESM-2 + residue type) evolutionary context
    attention bias  -d(atom, residue) / tau   geometry, not learned from scratch

Two properties earn its place over the GBDT:

  It can express cooperativity. "This carboxylate against that arginine, given
  that this ring is stacked over there" is a product of interactions; a bag of
  counts can only sum them.

  The attention matrix is the explanation. One weight per (ligand atom, pocket
  residue) pair, directly readable as a binding map and checkable against known
  pharmacophores - which is a validation, not a vanity plot, and something
  neither Boltz-2 nor PRODIGY-LIG can produce.

The same architecture serves both stages of the v1 flow: one instance trained
with a binary head is the binder/nonbinder classifier, another with a scalar
head is the pKd regressor. They are separate models with separate weights,
saved as separate .pkl files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from savant2.structure import AMINO_ACIDS

# Per-atom ligand features, all cheap and all from RDKit.
ATOM_ELEMENTS = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P")
ATOM_FEATURE_DIM = len(ATOM_ELEMENTS) + 1 + 6      # onehot + other, then 6 scalars
MAX_LIGAND_ATOMS = 96
TAU_ANGSTROM = 4.0        # attention-bias length scale


def ligand_atom_features(mol, n_atoms: int) -> np.ndarray:
    """(n_atoms, ATOM_FEATURE_DIM) descriptors for the ligand's heavy atoms."""
    features = np.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    if mol is None or mol.GetNumAtoms() != n_atoms:
        return features
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol in ATOM_ELEMENTS:
            features[i, ATOM_ELEMENTS.index(symbol)] = 1.0
        else:
            features[i, len(ATOM_ELEMENTS)] = 1.0
        offset = len(ATOM_ELEMENTS) + 1
        features[i, offset + 0] = float(atom.GetIsAromatic())
        features[i, offset + 1] = float(atom.IsInRing())
        features[i, offset + 2] = float(atom.GetFormalCharge())
        features[i, offset + 3] = float(atom.GetDegree()) / 4.0
        features[i, offset + 4] = float(atom.GetTotalNumHs()) / 4.0
        features[i, offset + 5] = float(atom.GetMass()) / 100.0
    return features


def residue_type_onehot(names: np.ndarray) -> np.ndarray:
    out = np.zeros((len(names), len(AMINO_ACIDS)), dtype=np.float32)
    for i, name in enumerate(names):
        if name in AMINO_ACIDS:
            out[i, AMINO_ACIDS.index(name)] = 1.0
    return out


def pairwise_distances(ligand_xyz: np.ndarray, residue_xyz: np.ndarray) -> np.ndarray:
    if len(ligand_xyz) == 0 or len(residue_xyz) == 0:
        return np.zeros((len(ligand_xyz), len(residue_xyz)), np.float32)
    return np.linalg.norm(
        ligand_xyz[:, None, :] - residue_xyz[None, :, :], axis=2
    ).astype(np.float32)


# --------------------------------------------------------------------------
# torch model. Imported lazily so the rest of savant2 runs without torch.
# --------------------------------------------------------------------------
def build_module(
    esm_dim: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_outputs: int = 1,
):
    """Construct the torch module. Kept in a factory so importing this file
    never requires torch."""
    import torch
    import torch.nn as nn

    class DistanceBiasedCrossAttention(nn.Module):
        """One block: ligand atoms attend to pocket residues, biased by distance."""

        def __init__(self, d_model: int, n_heads: int, dropout: float):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.q = nn.Linear(d_model, d_model)
            self.k = nn.Linear(d_model, d_model)
            self.v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            # Learned per-head sensitivity to distance: some interaction types
            # reach further than others, so the model tunes its own length scale.
            self.distance_scale = nn.Parameter(torch.ones(n_heads))

        def forward(self, ligand, pocket, dist, pocket_mask, need_weights=False):
            B, M, _ = ligand.shape
            K = pocket.shape[1]
            q = self.q(self.norm_q(ligand)).view(B, M, self.n_heads, self.d_head).transpose(1, 2)
            kv = self.norm_kv(pocket)
            k = self.k(kv).view(B, K, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v(kv).view(B, K, self.n_heads, self.d_head).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
            # Closer pairs get a higher prior score. This is the inductive bias
            # that lets the model learn from thousands rather than millions of
            # complexes: it starts out knowing that contact requires proximity.
            bias = -dist.unsqueeze(1) / (TAU_ANGSTROM * self.distance_scale.view(1, -1, 1, 1))
            scores = scores + bias
            scores = scores.masked_fill(~pocket_mask[:, None, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights)

            attended = (self.dropout(weights) @ v).transpose(1, 2).reshape(B, M, -1)
            updated = ligand + self.out(attended)
            return (updated, weights) if need_weights else (updated, None)

    class CrossAttentionScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ligand_in = nn.Sequential(
                nn.Linear(ATOM_FEATURE_DIM, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
            self.pocket_in = nn.Sequential(
                nn.Linear(esm_dim + len(AMINO_ACIDS) + 1, d_model),
                nn.GELU(), nn.LayerNorm(d_model),
            )
            self.blocks = nn.ModuleList(
                [DistanceBiasedCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)]
            )
            self.ffn = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                              nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
                for _ in range(n_layers)
            ])
            self.head = nn.Sequential(
                nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_model, n_outputs),
            )

        def forward(self, batch, need_weights: bool = False):
            ligand = self.ligand_in(batch["ligand_features"])
            pocket = self.pocket_in(
                torch.cat([batch["pocket_embedding"], batch["pocket_type"],
                           batch["pocket_distance"].unsqueeze(-1)], dim=-1)
            )
            attn = None
            for block, ffn in zip(self.blocks, self.ffn):
                ligand, attn = block(ligand, pocket, batch["distance"],
                                     batch["pocket_mask"], need_weights)
                ligand = ligand + ffn(ligand)

            mask = batch["ligand_mask"].unsqueeze(-1).float()
            summed = (ligand * mask).sum(1)
            counts = mask.sum(1).clamp(min=1.0)
            pooled = torch.cat([summed / counts, (ligand * mask).max(1).values], dim=-1)
            return self.head(pooled), attn

    return CrossAttentionScorer()


@dataclass
class AttentionModel:
    """Picklable wrapper: config plus weights, saved as a .pkl like the GBDTs."""

    esm_dim: int
    task: str = "regression"          # or "classification"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    state_dict: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    _module: object = field(default=None, repr=False, compare=False)

    def module(self):
        """Instantiate (once) and return the torch module on CPU."""
        if self._module is None:
            self._module = build_module(
                self.esm_dim, self.d_model, self.n_heads, self.n_layers,
                self.dropout, n_outputs=1,
            )
            if self.state_dict is not None:
                self._module.load_state_dict(self.state_dict)
        return self._module

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_module"] = None          # weights travel in state_dict
        return state

    def save(self, path) -> None:
        import joblib
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "AttentionModel":
        import joblib
        return joblib.load(path)


def collate(examples: List[Dict], esm_dim: int, device: str = "cpu"):
    """Pad a list of per-complex examples into a batch of torch tensors."""
    import torch

    B = len(examples)
    M = max(1, max(len(e["ligand_features"]) for e in examples))
    K = max(1, max(len(e["pocket_embedding"]) for e in examples))

    batch = {
        "ligand_features": torch.zeros(B, M, ATOM_FEATURE_DIM),
        "pocket_embedding": torch.zeros(B, K, esm_dim),
        "pocket_type": torch.zeros(B, K, len(AMINO_ACIDS)),
        "pocket_distance": torch.zeros(B, K),
        "distance": torch.full((B, M, K), 30.0),
        "ligand_mask": torch.zeros(B, M, dtype=torch.bool),
        "pocket_mask": torch.zeros(B, K, dtype=torch.bool),
    }
    for i, e in enumerate(examples):
        m, k = len(e["ligand_features"]), len(e["pocket_embedding"])
        batch["ligand_features"][i, :m] = torch.from_numpy(np.asarray(e["ligand_features"], np.float32))
        if k:
            batch["pocket_embedding"][i, :k] = torch.from_numpy(np.asarray(e["pocket_embedding"], np.float32))
            batch["pocket_type"][i, :k] = torch.from_numpy(np.asarray(e["pocket_type"], np.float32))
            batch["pocket_distance"][i, :k] = torch.from_numpy(np.asarray(e["pocket_distance"], np.float32))
            batch["distance"][i, :m, :k] = torch.from_numpy(np.asarray(e["distance"], np.float32))
        batch["ligand_mask"][i, :m] = True
        batch["pocket_mask"][i, :k] = True
    return {k: v.to(device) for k, v in batch.items()}
