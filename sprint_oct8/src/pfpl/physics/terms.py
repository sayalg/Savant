"""The physical functional forms. One registry entry per term (PLAN.md Section 6.2).

Every term is a pure function of a distance grid and the two sites' parameters:

    e_t(d; theta_i, theta_j) -> (..., n_bins)   kcal/mol, attractive = negative

They are evaluated once at the bin centres and then averaged under the predicted
distance distribution in `energy.py`. Nothing here knows about batches, models, or
labels - which is what makes `tests/test_physics.py` able to check them directly.

Two properties are enforced for every term, because Section 6.5 tests for them:

* **finite and differentiable everywhere, including d -> 0.** Every 1/d is softened
  as 1/sqrt(d^2 + SOFT^2). A raw Lennard-Jones is +inf at the origin, and one NaN in
  a batch kills a multi-day training run.
* **exactly zero beyond the cutoff.** A cosine switch over [D_SWITCH, D_CUT]
  multiplies every pairwise term, so "decays to zero at long range" is true by
  construction rather than approximately true. Without it a 10^-4 kcal/mol tail
  summed over 64 residues x 40 atoms x 20 bins is a real, meaningless offset.

`vdw_form` is an ablation, not a setting: PLAN.md Section 6.2 asks for
Lennard-Jones vs PIGNet2's Morse-type well, and Phase 9 reports both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .sites import P_ACC, P_AROM, P_CAT, P_DONOR, P_EPS, P_HAL, P_PHOB, P_Q, P_R0

## Distogram grid (Section 6.1): 2.0-12.0 A in 0.5 A bins, plus a "far" bin that
## contributes exactly zero to every term.
D_MIN, D_MAX, D_STEP = 2.0, 12.0, 0.5
N_DIST_BINS = int(round((D_MAX - D_MIN) / D_STEP))      # 20
N_BINS = N_DIST_BINS + 1                                 # + far/no-contact
FAR_BIN = N_DIST_BINS

D_SWITCH, D_CUT = 9.0, 11.0     # cosine switch window, inside the last real bin
SOFT = 0.8                      # A, softening length for every 1/d
COULOMB_K = 332.0637            # kcal*A/(mol*e^2)


def bin_edges(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.linspace(D_MIN, D_MAX, N_DIST_BINS + 1, device=device, dtype=dtype)


def bin_centers(device=None, dtype=torch.float32) -> torch.Tensor:
    """(N_DIST_BINS,) centres of the real distance bins. The far bin has no centre."""
    e = bin_edges(device, dtype)
    return 0.5 * (e[:-1] + e[1:])


def distance_to_onehot(d: torch.Tensor) -> torch.Tensor:
    """Real distances -> one-hot over N_BINS. This is what oracle mode feeds in.

    Anything at or beyond D_MAX lands in the far bin; anything below D_MIN lands in
    the first bin, so a clash is represented rather than dropped.
    """
    idx = torch.floor((d - D_MIN) / D_STEP).long()
    idx = torch.where(d >= D_MAX, torch.full_like(idx, FAR_BIN), idx.clamp(0, N_DIST_BINS - 1))
    return torch.nn.functional.one_hot(idx, N_BINS).to(d.dtype)


def switch(d: torch.Tensor) -> torch.Tensor:
    """Smooth 1 -> 0 cosine taper over [D_SWITCH, D_CUT]; C1-continuous at both ends."""
    x = ((d - D_SWITCH) / (D_CUT - D_SWITCH)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.pi * x))


def _soft_d(d: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(d * d + SOFT * SOFT)


def _gauss(d: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((d - mu) / sigma) ** 2)


# --------------------------------------------------------------------- the terms
# Signature: f(d, pi, pj) -> (..., n_d)
#   d  : (n_d,)            bin centres
#   pi : (..., 1, P)       protein-site parameters, broadcast over the grid
#   pj : (..., 1, P)       ligand-site parameters
# Return is in kcal/mol and is NOT yet switched; `evaluate` applies the taper.

def vdw_lj(d, pi, pj):
    """Soft-core 12-6: eps*[(r0/d)^12 - 2(r0/d)^6]. Minimum -eps at d = r0."""
    eps = torch.sqrt(pi[..., P_EPS] * pj[..., P_EPS])          # geometric (Berthelot)
    r0 = 0.5 * (pi[..., P_R0] + pj[..., P_R0])                 # arithmetic (Lorentz)
    x = (r0 * r0 / (d * d + SOFT * SOFT)) ** 3                 # (r0/d_soft)^6
    return eps * (x * x - 2.0 * x)


def vdw_morse(d, pi, pj, alpha: float = 1.8):
    """PIGNet2-style Morse well: eps*[(1 - e^{-a(d-r0)})^2 - 1].

    Softer than 12-6 at short range, which matters here: averaging a steep r^-12
    wall under a broad predicted distribution smears it into a meaningless constant,
    so the LJ-vs-Morse ablation is really a test of how much the repulsive wall
    survives pose uncertainty.
    """
    eps = torch.sqrt(pi[..., P_EPS] * pj[..., P_EPS])
    r0 = 0.5 * (pi[..., P_R0] + pj[..., P_R0])
    t = 1.0 - torch.exp(-alpha * (d - r0))
    return eps * (t * t - 1.0)


def hbond(d, pi, pj, mu: float = 2.95, sigma: float = 0.40):
    """Gaussian well at donor-acceptor heavy-atom distance, both directions.

    Distance-only, so it cannot see the D-H...A angle. Section 6.1 flags this as a
    known limitation of residue-level coarse-graining, and Section 9 asks for the
    failure cases to be reported by chemistry rather than hidden.
    """
    pair = (pi[..., P_DONOR] * pj[..., P_ACC] + pi[..., P_ACC] * pj[..., P_DONOR])
    return -pair * _gauss(d, mu, sigma)


def hydrophobic(d, pi, pj, good: float = 0.5, bad: float = 1.5):
    """Vina/ChemScore-style piecewise-linear contact between apolar sites.

    1 at surface separation <= `good`, ramping linearly to 0 at `bad`, where
    separation is d - (r0_i + r0_j)/2 so it scales with site size.
    """
    r0 = 0.5 * (pi[..., P_R0] + pj[..., P_R0])
    s = d - r0
    ramp = ((bad - s) / (bad - good)).clamp(0.0, 1.0)
    return -(pi[..., P_PHOB] * pj[..., P_PHOB]) * ramp


def coulomb(d, pi, pj):
    """Screened Coulomb with a distance-dependent dielectric, eps(d) = 4d.

    Section 0.4 rule 7: PIGNet2 omits explicit electrostatics and still performs, so
    this is Wave-3 and gated. A directionless, desolvation-free Coulomb on
    residue-level group charges may well add noise; the ablation decides.
    """
    ds = _soft_d(d)
    return COULOMB_K * pi[..., P_Q] * pj[..., P_Q] / (4.0 * ds * ds)


def aromatic(d, pi, pj, mu: float = 4.40, sigma: float = 0.90):
    """pi-pi stacking between ring centroids, plus cation-pi, as distance wells."""
    pp = pi[..., P_AROM] * pj[..., P_AROM]
    cp = pi[..., P_CAT] * pj[..., P_AROM] + pi[..., P_AROM] * pj[..., P_CAT]
    return -(pp * _gauss(d, mu, sigma) + cp * _gauss(d, 4.2, 1.0))


def halogen(d, pi, pj, mu: float = 3.30, sigma: float = 0.35):
    """Halogen bond: ligand sigma-hole donor vs protein acceptor.

    Strongly directional in reality (C-X...A near 165-180 deg). With no angle
    available this is an upper bound on the interaction and is expected to be the
    weakest of the gated terms.
    """
    return -(pj[..., P_HAL] * pi[..., P_ACC] + pi[..., P_HAL] * pj[..., P_ACC]) \
        * _gauss(d, mu, sigma)


@dataclass(frozen=True)
class Term:
    name: str
    fn: Callable
    wave: int               # 2 = MVP, 3 = gated expansion
    attractive: bool        # True -> weight constrained positive (Section 6.2)
    needs: tuple = ()       # parameter channels it reads, for the priors report


REGISTRY: dict[str, Term] = {
    "vdw": Term("vdw", vdw_lj, 2, True, ("eps", "r0")),
    "vdw_morse": Term("vdw_morse", vdw_morse, 2, True, ("eps", "r0")),
    "hbond": Term("hbond", hbond, 2, True, ("donor", "acceptor")),
    "hydrophobic": Term("hydrophobic", hydrophobic, 2, True, ("hydrophobic", "r0")),
    "coulomb": Term("coulomb", coulomb, 3, False, ("charge",)),
    "aromatic": Term("aromatic", aromatic, 3, True, ("aromatic", "cation")),
    "halogen": Term("halogen", halogen, 3, True, ("halogen", "acceptor")),
}

MVP_TERMS = ("vdw", "hbond", "hydrophobic")          # + the rotor term in energy.py
ALL_PAIR_TERMS = tuple(REGISTRY)


def resolve_terms(names, vdw_form: str = "lj") -> list[Term]:
    """Map config names to registry entries, honouring the vdw_form ablation."""
    out = []
    for n in names:
        if n == "vdw":
            n = "vdw_morse" if vdw_form == "morse" else "vdw"
        if n not in REGISTRY:
            raise SystemExit(f"unknown physics term {n!r}; have {sorted(REGISTRY)}")
        out.append(REGISTRY[n])
    return out


def evaluate(term: Term, pi: torch.Tensor, pj: torch.Tensor,
             d: torch.Tensor | None = None, apply_switch: bool = True) -> torch.Tensor:
    """e_t at every bin centre, tapered, with the far bin appended as exact zero.

        pi : (..., n_i, P)  protein-site parameters
        pj : (..., n_j, P)  ligand-site parameters
        ->   (..., n_i, n_j, N_BINS)

    The broadcast is done here, once, so a term function only ever writes plain
    elementwise arithmetic and cannot get the axis order wrong.
    """
    d = bin_centers(pi.device, pi.dtype) if d is None else d.to(pi.dtype)
    pi_ = pi.unsqueeze(-2).unsqueeze(-2)          # (..., n_i,   1, 1, P)
    pj_ = pj.unsqueeze(-3).unsqueeze(-2)          # (...,   1, n_j, 1, P)
    e = term.fn(d, pi_, pj_)                      # (..., n_i, n_j, n_d)
    if apply_switch:
        e = e * switch(d)
    zero = torch.zeros(*e.shape[:-1], 1, device=e.device, dtype=e.dtype)
    return torch.cat([e, zero], dim=-1)
