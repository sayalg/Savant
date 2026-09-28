"""Protein and ligand representations, including the PLM-mode ablation.

PLAN.md Section 6.7 is explicit that the PLM is an *empirical question*, not a
given: frozen ESM-2 features reduced Pearson on every strict LP-PDBBind tier
(-0.024 / -0.042 / -0.064). The last sprint reproduced that result the hard way -
its best model with 7,680 ESM dimensions scored per-target rho 0.158 against 0.325
for a ligand-only control on 613 held-out protein clusters. Protein features that
carry family identity are worse than no protein features at all when the split
removes the families.

So every mode here is a config variant that Phase 9 runs on strict splits, and the
MVP default is whichever wins, not `esm2_650m_full`.

    none                residue-type embedding only; no PLM tensor is read at all
    esm2_650m_full      frozen 1280-d per residue
    esm2_bottleneck     1280 -> 64 with dropout and train-time noise
    esm2_pocket_window  full width, but zeroed outside the predicted pocket window
    esmc_300m           frozen 960-d
    esm2_3b             frozen 2560-d (compute ablation)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..physics.sites import AA, UNK

PLM_DIM = {
    "none": 0,
    "esm2_650m_full": 1280,
    "esm2_bottleneck": 1280,
    "esm2_pocket_window": 1280,
    "esmc_300m": 960,
    "esm2_3b": 2560,
}
PLM_MODES = tuple(PLM_DIM)


class ProteinTrunk(nn.Module):
    """(residue indices, optional PLM embeddings) -> (B, L, d_model) per residue."""

    def __init__(self, mode: str = "esm2_bottleneck", d_model: int = 256,
                 bottleneck: int = 64, dropout: float = 0.1,
                 noise_std: float = 0.1, aa_dim: int = 64):
        super().__init__()
        if mode not in PLM_DIM:
            raise SystemExit(f"plm.mode={mode!r} unknown; have {PLM_MODES}")
        self.mode = mode
        self.noise_std = noise_std
        self.aa = nn.Embedding(len(AA) + 1, aa_dim, padding_idx=UNK)

        plm_in = PLM_DIM[mode]
        if mode == "esm2_bottleneck":
            self.proj = nn.Sequential(nn.LayerNorm(plm_in),
                                      nn.Linear(plm_in, bottleneck), nn.GELU(),
                                      nn.Dropout(dropout))
            plm_out = bottleneck
        elif plm_in:
            self.proj = nn.Sequential(nn.LayerNorm(plm_in),
                                      nn.Linear(plm_in, d_model // 2), nn.GELU(),
                                      nn.Dropout(dropout))
            plm_out = d_model // 2
        else:
            self.proj = None
            plm_out = 0

        self.mix = nn.Sequential(
            nn.Linear(aa_dim + plm_out, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.d_model = d_model

    @property
    def needs_plm(self) -> bool:
        return self.mode != "none"

    @property
    def base_dim(self) -> int:
        return self.aa.embedding_dim

    def base(self, res_idx: torch.Tensor) -> torch.Tensor:
        """PLM-free residue representation, for the pocket head (see pocket.py)."""
        return self.aa(res_idx.clamp(0, len(AA)))

    def forward(self, res_idx: torch.Tensor, plm: torch.Tensor | None = None,
                pocket_window: torch.Tensor | None = None) -> torch.Tensor:
        x = self.aa(res_idx.clamp(0, len(AA)))
        if self.proj is not None:
            if plm is None:
                raise SystemExit(
                    f"plm.mode={self.mode!r} needs cached per-residue embeddings, but the "
                    "batch carries none. Run scripts/phase4_embed.py for this mode, or "
                    "set plm.mode=none.")
            p = plm.to(x.dtype)
            if self.mode == "esm2_pocket_window" and pocket_window is not None:
                ## Keep the width but delete everything outside the predicted pocket,
                ## so the model cannot read family identity off distant residues.
                p = p * pocket_window.unsqueeze(-1).to(p.dtype)
            if self.training and self.noise_std > 0:
                p = p + torch.randn_like(p) * self.noise_std
            x = torch.cat([x, self.proj(p)], dim=-1)
        return self.mix(x)


class LigandTrunk(nn.Module):
    """Atom features (+ optional frozen CLM embeddings) -> (B, A, d_model).

    A small message-passing stack over the covalent graph. Uni-Mol atom embeddings
    are accepted through `clm` when cached (Section 3), but the graph encoder is not
    optional: ring centroids are pseudo-atoms with no CLM vector of their own and
    still need a representation.
    """

    def __init__(self, atom_dim: int, bond_dim: int, d_model: int = 256,
                 clm_dim: int = 0, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(atom_dim + clm_dim, d_model), nn.GELU(),
                                   nn.LayerNorm(d_model))
        self.clm_dim = clm_dim
        self.bond = nn.Sequential(nn.Linear(bond_dim, d_model), nn.GELU())
        self.msg = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout))
            for _ in range(layers)])
        self.upd = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.d_model = d_model

    def forward(self, atom_feat: torch.Tensor, adj: torch.Tensor,
                bond_feat: torch.Tensor, mask: torch.Tensor,
                clm: torch.Tensor | None = None) -> torch.Tensor:
        """atom_feat (B,A,F), adj (B,A,A) float, bond_feat (B,A,A,Fb), mask (B,A)."""
        if self.clm_dim:
            if clm is None:
                raise SystemExit("ligand.clm_dim>0 but no cached CLM embeddings in the batch")
            atom_feat = torch.cat([atom_feat, clm.to(atom_feat.dtype)], dim=-1)
        h = self.embed(atom_feat) * mask.unsqueeze(-1)
        e = self.bond(bond_feat)                                  # (B,A,A,d)
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        for msg, norm in zip(self.msg, self.upd):
            nbr = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)   # (B,A,A,d)
            m = torch.cat([nbr, e], dim=-1)
            m = (msg(m) * adj.unsqueeze(-1)).sum(2) / deg
            h = norm(h + m) * mask.unsqueeze(-1)
        return h
