"""Force-field parameter heads: embeddings -> theta (PLAN.md Sections 6.2, 6.4).

The heads predict a *residual* on top of the tabulated prior in `sites.py`, then
squash into the physical range from `PARAM_RANGE`. Both choices are deliberate:

* predicting a residual means an untrained model already evaluates approximately
  correct physics, so Stage 1 does not have to rediscover that oxygen accepts
  hydrogen bonds before it can learn anything about geometry;
* squashing means no amount of label pressure can produce a 40 kcal/mol well or a
  +7 e group charge, which is the standard way these models cheat.

Section 6.4 asks for the drift of the parameter distributions to be logged per
stage. `drift_report()` produces that, and `prior_penalty()` is the annealed L2
that keeps it small.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .sites import (N_SITE_PARAMS, PARAM_RANGE, SITE_PARAMS, ELEM_IDX,
                    RING_CENTROID_PRIOR, ligand_element_priors,
                    residue_site_mask, residue_site_priors)

_LO = torch.tensor([PARAM_RANGE[k][0] for k in SITE_PARAMS], dtype=torch.float32)
_HI = torch.tensor([PARAM_RANGE[k][1] for k in SITE_PARAMS], dtype=torch.float32)


def squash(raw: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def unsquash(value: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Inverse of `squash`, used to turn a tabulated prior into a head bias."""
    p = ((value - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


class SiteParamHead(nn.Module):
    """Per-site parameters from (embedding, prior). Shared form for both molecules.

    `scale` caps how far a head may move from its prior in logit space. With
    scale=2.0 a site can go from the 0.12 quantile of its range to the 0.88
    quantile, which is enough to turn a weak donor into a strong one and not enough
    to invent a charge.
    """

    def __init__(self, in_dim: int, hidden: int = 128, scale: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, N_SITE_PARAMS),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)          # start exactly at the prior
        self.register_buffer("lo", _LO.clone())
        self.register_buffer("hi", _HI.clone())

    def forward(self, emb: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """emb (..., D), prior (..., N_SITE_PARAMS) -> theta (..., N_SITE_PARAMS)."""
        base = unsquash(prior, self.lo, self.hi)
        delta = torch.tanh(self.net(emb)) * self.scale
        return squash(base + delta, self.lo, self.hi)


class ProteinSiteParams(nn.Module):
    """Residue embeddings -> (n_res, 4 pseudo-sites, N_SITE_PARAMS) + a validity mask.

    Each pseudo-site gets its own learned "kind" vector concatenated to the residue
    embedding, so BB / SC / TIP / RING share the trunk but not their behaviour.
    """

    def __init__(self, emb_dim: int, hidden: int = 128, kind_dim: int = 16, **kw):
        super().__init__()
        self.kind = nn.Embedding(4, kind_dim)
        self.head = SiteParamHead(emb_dim + kind_dim, hidden, **kw)
        self.register_buffer("prior", torch.from_numpy(residue_site_priors()))
        self.register_buffer("mask", torch.from_numpy(residue_site_mask()))

    def forward(self, emb: torch.Tensor, res_idx: torch.Tensor):
        """emb (B, L, D), res_idx (B, L) int -> theta (B, L, 4, P), mask (B, L, 4)."""
        b, ll, _ = emb.shape
        prior = self.prior[res_idx]                                   # (B, L, 4, P)
        mask = self.mask[res_idx]                                     # (B, L, 4)
        kinds = self.kind.weight.view(1, 1, 4, -1).expand(b, ll, -1, -1)
        x = torch.cat([emb.unsqueeze(2).expand(-1, -1, 4, -1), kinds], dim=-1)
        return self.head(x, prior), mask


class LigandSiteParams(nn.Module):
    """Atom embeddings -> (n_atom, N_SITE_PARAMS).

    The element prior is overridden per atom by RDKit-derived donor/acceptor/charge
    when `feat_prior` is supplied by `features/embed_ligand.py`: RDKit knows the
    protonation state and formal charge of *this* molecule, which an element table
    cannot.
    """

    def __init__(self, emb_dim: int, hidden: int = 128, **kw):
        super().__init__()
        self.head = SiteParamHead(emb_dim, hidden, **kw)
        self.register_buffer("prior", torch.from_numpy(ligand_element_priors()))
        self.register_buffer("ring_prior", torch.from_numpy(RING_CENTROID_PRIOR.copy()))

    def forward(self, emb: torch.Tensor, elem_idx: torch.Tensor,
                is_ring_centroid: torch.Tensor | None = None,
                feat_prior: torch.Tensor | None = None) -> torch.Tensor:
        prior = self.prior[elem_idx]                                  # (B, A, P)
        if is_ring_centroid is not None:
            rp = self.ring_prior.view(1, 1, -1).expand_as(prior)
            prior = torch.where(is_ring_centroid.unsqueeze(-1).bool(), rp, prior)
        if feat_prior is not None:
            ## RDKit-derived channels win where they are defined (finite).
            prior = torch.where(torch.isfinite(feat_prior), feat_prior, prior)
        return self.head(emb, prior)


def prior_penalty(theta: torch.Tensor, prior: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
    """Annealed L2 toward the tabulated values, in normalised parameter units.

    Normalising by each parameter's range stops the penalty from being dominated by
    `charge` (span 2.0) and ignoring `eps` (span 0.58).
    """
    span = (_HI - _LO).to(theta.device)
    d = ((theta - prior) / span) ** 2
    if mask is not None:
        d = d * mask.unsqueeze(-1)
        denom = mask.sum().clamp(min=1.0) * theta.shape[-1]
        return d.sum() / denom
    return d.mean()


@torch.no_grad()
def drift_report(theta: torch.Tensor, prior: torch.Tensor,
                 mask: torch.Tensor | None = None) -> dict:
    """Per-parameter mean/std drift from the prior - the Section 6.4 log line."""
    t, p = theta.detach().float(), prior.detach().float()
    if mask is not None:
        m = mask.bool().unsqueeze(-1).expand_as(t)
        t, p = t[m].view(-1, t.shape[-1]), p[m].view(-1, p.shape[-1])
    else:
        t, p = t.reshape(-1, t.shape[-1]), p.reshape(-1, p.shape[-1])
    if t.numel() == 0:
        return {}
    d = (t - p)
    return {f"drift/{k}_mean": float(d[:, i].mean()) for i, k in enumerate(SITE_PARAMS)} | \
           {f"drift/{k}_std": float(t[:, i].std()) for i, k in enumerate(SITE_PARAMS)}


def elements_to_idx(symbols) -> np.ndarray:
    return np.array([ELEM_IDX.get(s, ELEM_IDX["*"]) for s in symbols], np.int16)
