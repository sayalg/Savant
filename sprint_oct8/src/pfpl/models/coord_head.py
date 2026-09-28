"""Variant B: geometry-consistent by construction (PLAN.md Section 2.2).

Variant A predicts each P_ij independently, so nothing stops the model from
claiming a set of distances no arrangement of atoms could produce - E3Bind
documents exactly these "knotted" maps from TankBind-style heads. Variant A patches
that with penalties; Variant B removes the possibility.

A light coordinate head emits `n_hypotheses` pocket-frame placements: coordinates
for the K x 4 protein sites and the A ligand sites. Distances - and therefore every
energy - are computed FROM those coordinates, so the triangle inequality holds by
arithmetic rather than by penalty. The energy is the mixture-weighted average over
hypotheses.

The interface is identical to Variant A on purpose: both hand the energy module a
(B, n_i, n_j, N_BINS) tensor, so `physics/energy.py` never learns which variant
produced it. Variant B's distributions are soft-binned rather than one-hot, keeping
the whole path differentiable.

Trained winner-take-all (gradient to the best hypothesis only) plus soft mixture
weights, which is the standard recipe for multi-hypothesis heads and avoids the
mode collapse that a plain mixture average produces.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics import terms as T


def soft_bin(d: torch.Tensor, sharpness: float = 4.0) -> torch.Tensor:
    """Distances -> a differentiable distribution over the N_BINS grid.

    A hard bin index has zero gradient with respect to the coordinates, which would
    disconnect the coordinate head from the affinity loss entirely.
    """
    centers = T.bin_centers(d.device, d.dtype)
    logits = -sharpness * (d.unsqueeze(-1) - centers).abs()
    far = sharpness * (d - T.D_MAX)                       # wins once d > D_MAX
    return torch.softmax(torch.cat([logits, far.unsqueeze(-1)], -1), dim=-1)


class CoordinateHypothesisHead(nn.Module):
    """z_ij and the two towers -> K x 4 protein site coords and A ligand coords.

    Coordinates live in an arbitrary pocket frame. Nothing downstream is rotation
    sensitive (only distances are used), so no equivariance machinery is needed -
    which is the entire reason this variant is affordable inside the sprint.
    """

    def __init__(self, d_model: int, pair_dim: int, n_hypotheses: int = 4,
                 n_kinds: int = 4, hidden: int = 128, dropout: float = 0.1,
                 init_spread: float = 1.2, seed: int = 0):
        super().__init__()
        self.h = n_hypotheses
        self.n_kinds = n_kinds
        self.prot = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_hypotheses * n_kinds * 3))
        self.lig = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_hypotheses * 3))
        self.mix = nn.Sequential(
            nn.LayerNorm(2 * pair_dim), nn.Linear(2 * pair_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_hypotheses))
        ## Start from a diffuse, plausible pocket rather than from the origin.
        ##
        ## With the default init the last layer emits small values, tanh is near
        ## linear there, and every site lands within a fraction of an Angstrom of
        ## every other - so the whole complex begins as one enormous clash and the
        ## model starts at pK ~ -7600. Zeroing the weight and giving the bias a fixed
        ## random cloud makes the untrained head predict a ~10 A pocket, and the
        ## network then learns displacements from it. Same idea as the parameter
        ## heads starting at the tabulated force field.
        g = torch.Generator().manual_seed(seed)
        for head, n_out in ((self.prot, n_hypotheses * n_kinds * 3),
                            (self.lig, n_hypotheses * 3)):
            nn.init.zeros_(head[-1].weight)
            head[-1].bias.data.copy_(torch.randn(n_out, generator=g) * init_spread)

    def forward(self, prot: torch.Tensor, lig: torch.Tensor, ctx: torch.Tensor):
        """prot (B,K,d), lig (B,A,d), ctx (B,2*pair) -> coords and mixture weights."""
        b, k, _ = prot.shape
        a = lig.shape[1]
        px = self.prot(prot).view(b, k * self.n_kinds, self.h, 3).transpose(1, 2)
        lx = self.lig(lig).view(b, a, self.h, 3).transpose(1, 2)
        ## Scale into a plausible pocket box (~20 A across) so an untrained head does
        ## not start with every distance in the far bin and zero gradient.
        px, lx = 10.0 * torch.tanh(px), 10.0 * torch.tanh(lx)
        return px, lx, torch.softmax(self.mix(ctx), dim=-1)      # (B,H,n,3) ... (B,H)

    def distributions(self, prot_xyz, lig_xyz, weights, sharpness: float = 4.0):
        """Mixture-weighted P_ij, consistent by construction. (B, n_i, n_j, N_BINS)."""
        d = torch.cdist(prot_xyz, lig_xyz)                        # (B,H,n_i,n_j)
        P = soft_bin(d, sharpness)                                # (B,H,n_i,n_j,N)
        return (P * weights.view(*weights.shape, 1, 1, 1)).sum(1), d


def wta_coord_loss(pred_d: torch.Tensor, true_d: torch.Tensor,
                   valid: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Winner-take-all distance loss plus a cross-entropy on the mixture weights.

    pred_d (B,H,n_i,n_j), true_d (B,n_i,n_j), valid (B,n_i,n_j).
    """
    v = valid.unsqueeze(1).to(pred_d.dtype)
    err = ((pred_d - true_d.unsqueeze(1)).abs() * v).sum((-2, -1)) / v.sum((-2, -1)).clamp(min=1)
    best = err.argmin(dim=1)
    wta = err.gather(1, best.unsqueeze(1)).squeeze(1).mean()
    return wta + 0.1 * F.cross_entropy(torch.log(weights.clamp_min(1e-9)), best)
