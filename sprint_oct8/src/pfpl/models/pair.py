"""Pair module: cross-attention between pocket residues and ligand atoms.

Produces the pair representation z_ij that the distogram head, the coordinate head
(Variant B) and the no-physics control (Variant C) all consume. Keeping one pair
module behind all three is what makes the Phase 9 comparison a comparison of the
*energy function*, not of three differently-sized encoders (PLAN.md Section 8,
"parameters matched to the physics model").

The outer-product update is the piece that matters. The previous sprint's ablation
put an explicit interaction head at rho 0.120 against 0.070 for plain concatenation
on the same folds - nearly double, on 613 targets - which is the one positive
mechanistic result it produced and the reason this module exists rather than an MLP
over concatenated pooled vectors.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """One round of ligand->protein and protein->ligand attention, plus an FFN."""

    def __init__(self, d_model: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.p2l = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.l2p = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.np1, self.np2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.nl1, self.nl2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.ffl = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, p, l, p_mask, l_mask):
        pk = ~p_mask.bool()
        lk = ~l_mask.bool()
        a, _ = self.p2l(self.np1(p), self.nl1(l), self.nl1(l), key_padding_mask=lk,
                        need_weights=False)
        p = p + a
        b, _ = self.l2p(self.nl1(l), self.np1(p), self.np1(p), key_padding_mask=pk,
                        need_weights=False)
        l = l + b
        p = p + self.ffp(self.np2(p))
        l = l + self.ffl(self.nl2(l))
        return p * p_mask.unsqueeze(-1), l * l_mask.unsqueeze(-1)


class PairModule(nn.Module):
    """(pocket residues, ligand atoms) -> z_ij (B, K, A, pair_dim) plus updated towers.

    `geom_prior` carries the ligand conformer-ensemble internal-distance summary and
    the protein's own contact-derived features, so the pair rep starts with the
    geometry that *is* knowable without a pose (Section 6.3 Tier A).
    """

    def __init__(self, d_model: int = 256, pair_dim: int = 96, blocks: int = 3,
                 heads: int = 8, dropout: float = 0.1, geom_prior_dim: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, heads, dropout) for _ in range(blocks)])
        self.qp = nn.Linear(d_model, pair_dim)
        self.ql = nn.Linear(d_model, pair_dim)
        self.gate = nn.Linear(d_model, pair_dim)
        self.geom = nn.Linear(geom_prior_dim, pair_dim) if geom_prior_dim else None
        self.out = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(pair_dim))
        self.pair_dim = pair_dim
        self.d_model = d_model

    def forward(self, p, l, p_mask, l_mask, geom_prior=None):
        for blk in self.blocks:
            p, l = blk(p, l, p_mask, l_mask)
        ## Multiplicative outer product, not concatenation: the whole hypothesis is
        ## that affinity is an interaction, and a sum of two marginals cannot encode
        ## one. `gate` lets a residue switch the interaction off entirely.
        z = self.qp(p).unsqueeze(2) * self.ql(l).unsqueeze(1)          # (B,K,A,pair)
        z = z + torch.sigmoid(self.gate(p)).unsqueeze(2)
        if self.geom is not None and geom_prior is not None:
            z = z + self.geom(geom_prior)
        z = self.out(z)
        m = (p_mask.unsqueeze(-1) * l_mask.unsqueeze(-2)).unsqueeze(-1)
        return z * m, p, l


def pool_pair(z: torch.Tensor, p_mask: torch.Tensor, l_mask: torch.Tensor
              ) -> torch.Tensor:
    """Masked mean + max over (K, A) -> (B, 2*pair_dim) context for the residual head."""
    m = (p_mask.unsqueeze(-1) * l_mask.unsqueeze(-2)).unsqueeze(-1)
    s = (z * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1.0)
    mx = z.masked_fill(m == 0, -1e4).amax((1, 2))
    return torch.cat([s, mx], dim=-1)
