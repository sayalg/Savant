"""Distogram head (Variant A) - one distance distribution per (protein site, ligand site).

The pair module works at residue x atom resolution; the physics works at
pseudo-site x atom resolution (4 sites per residue, Section 6.1). Rather than
inflating the attention to 4L tokens, this head expands the residue-level pair
representation into the four site kinds with a per-kind projection. That keeps the
expensive part at K x A and the cheap part at 4K x A.

Section 2.1: the output is the thing most at risk of being gamed, because nothing
forces independently predicted P_ij to describe a geometry that exists. The
supervised cross-entropy anchor, the triangle penalty and the contact prior are all
applied to this head's output, and its calibration is monitored in every later
stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..physics import terms as T
from ..physics.sites import SITE_KINDS


class DistogramHead(nn.Module):
    def __init__(self, pair_dim: int, hidden: int = 128, dropout: float = 0.1,
                 n_kinds: int = len(SITE_KINDS), init_bias_far: float = 8.0):
        super().__init__()
        self.n_kinds = n_kinds
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_kinds * T.N_BINS))
        nn.init.zeros_(self.net[-1].weight)
        b = torch.zeros(n_kinds, T.N_BINS)
        ## Start with almost all mass in the far bin, where every term is exactly
        ## zero. The default is 8.0 because the arithmetic matters: softmax puts
        ## e^b / (e^b + N_DIST_BINS) on "far", so b=2 leaves 27% of the mass spread
        ## over the real bins and b=8 leaves 0.7%. Most protein-site/ligand-atom
        ## pairs in a real complex ARE far apart, so a near-uniform initial
        ## distribution is not neutral - it is a specific, wrong claim that every one
        ## of the ~2,500 pairs is in contact.
        ##
        ## Measured on the Phase 7 synthetic batch: b=2 starts the model at
        ## pK ~ -2230 (dG ~ +3000 kcal/mol of smeared r^-12 wall), b=8 starts it
        ## near zero. Both converge, but the first spends its early epochs undoing
        ## an initialisation artifact, and on a deadline that is real time.
        b[:, T.FAR_BIN] = init_bias_far
        self.net[-1].bias.data.copy_(b.reshape(-1))

    def forward(self, z: torch.Tensor, site_mask: torch.Tensor,
                lig_mask: torch.Tensor):
        """z (B,K,A,pair) -> logits (B, K*n_kinds, A, N_BINS), mask (B, K*n_kinds).

        Site order is (residue0.BB, residue0.SC, residue0.TIP, residue0.RING,
        residue1.BB, ...), matching `ProteinSiteParams` flattened the same way.
        """
        b, k, a, _ = z.shape
        logits = self.net(z).view(b, k, a, self.n_kinds, T.N_BINS)
        logits = logits.permute(0, 1, 3, 2, 4).reshape(b, k * self.n_kinds, a, T.N_BINS)
        flat_mask = site_mask.reshape(b, k * self.n_kinds)
        ## Sites that do not exist (GLY side chain) and padded atoms are forced into
        ## the far bin so they contribute exactly zero energy.
        dead = (flat_mask.unsqueeze(-1) * lig_mask.unsqueeze(-2)) == 0
        logits = logits.masked_fill(dead.unsqueeze(-1), -1e4)
        far = torch.zeros_like(logits)
        far[..., T.FAR_BIN] = 1e4
        logits = torch.where(dead.unsqueeze(-1), far, logits)
        return logits, flat_mask

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)


def expected_distance(P: torch.Tensor) -> torch.Tensor:
    """E[d] with the far bin held at D_MAX - conservative, never vacuous."""
    d = T.bin_centers(P.device, P.dtype)
    dfull = torch.cat([d, d.new_tensor([T.D_MAX])])
    return (P * dfull).sum(-1)
