"""Expected-energy assembly: the module that makes this project different.

There is no pose at inference, so every physical term is evaluated as an
expectation under the predicted distance distribution (PLAN.md Section 2):

    <E_t> = sum_{i,j} sum_b  P_ij(b) * e_t(d_b; theta_i, theta_j)

and the free energy is assembled as

    dG = sum_t softplus(w_t) <E_t> + desolvation + conformational + b [+ residual]
    pK = scale * (-dG / (RT ln10)) + offset

Three things in here exist only because of the failure mode in Section 2.1 - the
affinity loss can push every pair toward its energy minimum and produce
over-attractive, geometrically impossible predictions:

* `contact_count` and `burial` are returned so the training loop can apply the
  contact-count prior;
* `entropy` is computed but is off by default and always ablated, because rewarding
  a broad P_ij is the single most gameable term in the design;
* the module returns per-term, per-residue and per-ligand-atom decompositions, so
  Section 9's diagnostics can check that the energy is distributed over plausible
  residues rather than concentrated in one pathological pair.

`mode="oracle"` swaps predicted distributions for one-hot distributions built from
real coordinates. Same heads, same terms, same weights - the gap between the two is
the price of dropping the pose (Section 6.6), and `tests/test_physics.py` asserts
that the expected-energy path reproduces a direct coordinate evaluation exactly.

Sign convention throughout: attractive = negative energy, and a more negative dG
means tighter binding, so pK rises.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import terms as T

RT_LN10 = 1.98720425e-3 * 298.15 * 2.302585092994046   # 1.3642 kcal/mol per pK unit
CONTACT_BIN_CUTOFF = 5      # bins with centre < 4.5 A count as a contact


@dataclass
class EnergyOut:
    pK: torch.Tensor                     # (B,)
    dG: torch.Tensor                     # (B,)
    per_term: dict                       # name -> (B,) expected energy, unweighted
    weighted: dict                       # name -> (B,) after softplus(w_t)
    per_residue: torch.Tensor            # (B, n_i) total energy attributed to residue i
    per_atom: torch.Tensor               # (B, n_j) total energy attributed to atom j
    burial: torch.Tensor                 # (B, n_j) expected protein contacts per atom
    contact_count: torch.Tensor          # (B,) total expected contacts
    entropy: torch.Tensor                # (B,) mean H[P_ij] over valid pairs, nats
    residual: torch.Tensor               # (B,) residual-head contribution to dG
    extras: dict = field(default_factory=dict)


class EnergyModule(nn.Module):
    """Assembles dG from distance distributions and site parameters.

    Parameters
    ----------
    term_names   which pairwise terms are enabled (Section 6.2 registry names)
    vdw_form     "lj" or "morse" - the Section 6.2 ablation
    residual_dim >0 enables the residual head on a pooled context vector
    """

    def __init__(self, term_names=T.MVP_TERMS, vdw_form: str = "lj",
                 use_rotor: bool = True, use_desolvation: bool = False,
                 use_entropy: bool = False, residual_dim: int = 0,
                 residual_clamp: float = 2.0, residual_dropout: float = 0.2,
                 temperature_K: float = 298.15):
        super().__init__()
        self.terms = T.resolve_terms(term_names, vdw_form)
        self.term_names = [t.name for t in self.terms]
        self.use_rotor = use_rotor
        self.use_desolvation = use_desolvation
        self.use_entropy = use_entropy
        self.residual_clamp = residual_clamp
        self.T = temperature_K

        ## softplus(0.5413) == 1.0, so every term starts at unit weight.
        self.w = nn.Parameter(torch.full((len(self.terms),), 0.5413))
        ## Terms whose sign is not fixed (Coulomb) get a free weight instead of a
        ## positive-constrained one; Section 6.2 constrains only attractive terms.
        self.register_buffer("w_positive",
                             torch.tensor([t.attractive for t in self.terms]))

        self.w_rotor = nn.Parameter(torch.tensor(-1.0))       # softplus -> ~0.31
        self.w_desolv = nn.Parameter(torch.tensor(-1.0))
        self.w_entropy = nn.Parameter(torch.tensor(-2.0))     # sigmoid -> ~0.12
        self.bias = nn.Parameter(torch.zeros(()))             # GLOBAL only (Sec. 0.4.5)
        self.scale = nn.Parameter(torch.ones(()))
        self.offset = nn.Parameter(torch.zeros(()))

        self.residual = None
        if residual_dim > 0:
            self.residual = nn.Sequential(
                nn.LayerNorm(residual_dim),
                nn.Linear(residual_dim, 128), nn.GELU(), nn.Dropout(residual_dropout),
                nn.Linear(128, 1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    # ------------------------------------------------------------------ helpers
    def _weights(self) -> torch.Tensor:
        pos = F.softplus(self.w)
        return torch.where(self.w_positive, pos, self.w)

    @staticmethod
    def pair_mask(mask_i: torch.Tensor, mask_j: torch.Tensor) -> torch.Tensor:
        """(B, n_i) x (B, n_j) -> (B, n_i, n_j) float."""
        return (mask_i.unsqueeze(-1) * mask_j.unsqueeze(-2)).to(mask_i.dtype)

    # -------------------------------------------------------------------- main
    def forward(self, P: torch.Tensor, theta_i: torch.Tensor, theta_j: torch.Tensor,
                mask_i: torch.Tensor, mask_j: torch.Tensor,
                n_rot: torch.Tensor | None = None,
                dG_solv: torch.Tensor | None = None,
                context: torch.Tensor | None = None) -> EnergyOut:
        """
        P        (B, n_i, n_j, N_BINS)  distance distribution, sums to 1 over bins
        theta_i  (B, n_i, P)            protein-site parameters
        theta_j  (B, n_j, P)            ligand-site parameters
        mask_i   (B, n_i)  mask_j (B, n_j)   1 where the site exists
        n_rot    (B,)  rotatable-bond count       dG_solv (B,) kcal/mol, may be NaN
        """
        pm = self.pair_mask(mask_i, mask_j)                       # (B, n_i, n_j)
        Pm = P * pm.unsqueeze(-1)

        per_term, weighted = {}, {}
        contrib = torch.zeros_like(pm)                            # (B, n_i, n_j)
        wts = self._weights()
        for k, term in enumerate(self.terms):
            e = T.evaluate(term, theta_i, theta_j)                # (B,n_i,n_j,N_BINS)
            eij = (Pm * e).sum(-1)                                # expectation over d
            per_term[term.name] = eij.sum((-2, -1))
            weighted[term.name] = wts[k] * per_term[term.name]
            contrib = contrib + wts[k] * eij

        dG = torch.stack(list(weighted.values()), 0).sum(0) if weighted \
            else torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)

        ## ---- burial and the contact-count prior (Section 2.1) ------------------
        p_contact = Pm[..., :CONTACT_BIN_CUTOFF].sum(-1)          # (B, n_i, n_j)
        burial = p_contact.sum(-2)                                # per ligand atom
        contact_count = p_contact.sum((-2, -1))

        extras: dict = {}
        ## ---- conformational / rotor penalty -----------------------------------
        if self.use_rotor and n_rot is not None:
            nb = mask_j.sum(-1).clamp(min=1.0)
            rot = F.softplus(self.w_rotor) * n_rot * (burial.sum(-1) / nb)
            dG = dG + rot                                          # positive = penalty
            extras["rotor"] = rot

        ## ---- desolvation --------------------------------------------------------
        if self.use_desolvation and dG_solv is not None:
            ## Burying a well-solvated ligand costs its solvation free energy. NaN
            ## (Tier C not computed for this ligand) contributes exactly zero rather
            ## than poisoning the batch.
            frac = (burial.sum(-1) / mask_j.sum(-1).clamp(min=1.0)).clamp(0, 1)
            s = torch.nan_to_num(dG_solv, nan=0.0)
            desolv = F.softplus(self.w_desolv) * (-s) * frac
            dG = dG + desolv
            extras["desolvation"] = desolv

        ## ---- geometric entropy (off by default; Section 6.2 "always ablate") ----
        logP = torch.log(P.clamp_min(1e-9))
        H = (-(P * logP).sum(-1) * pm).sum((-2, -1)) / pm.sum((-2, -1)).clamp(min=1.0)
        if self.use_entropy:
            dG = dG - torch.sigmoid(self.w_entropy) * self.T * 1.98720425e-3 * H
            extras["entropy_term"] = H

        ## ---- residual head ------------------------------------------------------
        res = torch.zeros_like(dG)
        if self.residual is not None and context is not None:
            res = self.residual(context).squeeze(-1).clamp(-self.residual_clamp,
                                                           self.residual_clamp)
            dG = dG + res

        dG = dG + self.bias
        pK = self.scale * (-dG / RT_LN10) + self.offset

        return EnergyOut(
            pK=pK, dG=dG, per_term=per_term, weighted=weighted,
            per_residue=contrib.sum(-1), per_atom=contrib.sum(-2),
            burial=burial, contact_count=contact_count, entropy=H,
            residual=res, extras=extras)

    # ----------------------------------------------------------------- reporting
    @torch.no_grad()
    def weight_report(self) -> dict:
        w = self._weights().detach()
        d = {f"w/{n}": float(w[i]) for i, n in enumerate(self.term_names)}
        d["w/rotor"] = float(F.softplus(self.w_rotor))
        d["w/desolvation"] = float(F.softplus(self.w_desolv))
        d["w/entropy"] = float(torch.sigmoid(self.w_entropy))
        d["w/bias"] = float(self.bias)
        d["w/scale"] = float(self.scale)
        d["w/offset"] = float(self.offset)
        return d


# --------------------------------------------------------------- oracle geometry

def distributions_from_coords(prot_xyz: torch.Tensor, lig_xyz: torch.Tensor
                              ) -> torch.Tensor:
    """Real coordinates -> one-hot P_ij. This is `energy.mode: oracle`.

    prot_xyz (B, n_i, 3), lig_xyz (B, n_j, 3) -> (B, n_i, n_j, N_BINS)
    """
    d = torch.cdist(prot_xyz, lig_xyz)
    return T.distance_to_onehot(d)


def direct_energy_from_coords(module: EnergyModule, prot_xyz, lig_xyz,
                              theta_i, theta_j, mask_i, mask_j, **kw) -> EnergyOut:
    """Brute-force coordinate evaluation, for the Section 6.5 equivalence test.

    Deliberately NOT the same code path as `forward`: it bins the real distances
    and evaluates the terms at those bin centres directly, so agreeing with the
    expected-energy path is evidence rather than a tautology.
    """
    P = distributions_from_coords(prot_xyz, lig_xyz)
    return module(P, theta_i, theta_j, mask_i, mask_j, **kw)


# --------------------------------------------------------------- physics losses

def triangle_penalty(P: torch.Tensor, lig_dist: torch.Tensor,
                     mask_i: torch.Tensor, mask_j: torch.Tensor,
                     n_sample: int = 64, generator=None) -> torch.Tensor:
    """|E[d_ij] - E[d_ik]| must not exceed the ligand's own d_jk (Section 2.1).

    Independent per-pair distributions have no reason to be geometrically
    consistent; E3Bind documents "knotted", invalid distance maps from exactly this
    design. The ligand is rigid enough for its internal distances to be known from
    the conformer ensemble, so this is a real constraint and not a regulariser.

    Sampling `n_sample` (j, k) pairs rather than forming the full n_j^2 tensor keeps
    it affordable at every step; over an epoch every pair is visited many times.
    """
    b, ni, nj, _ = P.shape
    d = T.bin_centers(P.device, P.dtype)
    ## Expected distance: the far bin is treated as D_MAX, its lower bound, which
    ## makes the constraint conservative rather than vacuous.
    dfull = torch.cat([d, d.new_tensor([T.D_MAX])])
    Ed = (P * dfull).sum(-1)                                   # (B, n_i, n_j)

    g = generator
    jj = torch.randint(0, nj, (n_sample,), device=P.device, generator=g)
    kk = torch.randint(0, nj, (n_sample,), device=P.device, generator=g)
    lhs = (Ed[:, :, jj] - Ed[:, :, kk]).abs()                  # (B, n_i, S)
    rhs = lig_dist[:, jj, kk].unsqueeze(1)                     # (B, 1, S)
    viol = F.relu(lhs - rhs)
    m = mask_i.unsqueeze(-1) * (mask_j[:, jj] * mask_j[:, kk]).unsqueeze(1)
    return (viol * m).sum() / m.sum().clamp(min=1.0)


def contact_prior_loss(burial: torch.Tensor, mask_j: torch.Tensor,
                       target: float = 6.0, tol: float = 3.0) -> torch.Tensor:
    """Expected contacts per ligand heavy atom should match crystal statistics.

    ~6 protein pseudo-sites within 4.5 A of a buried ligand atom is the typical
    value for the crystal complexes used in Stage 1; `tol` makes this a soft band
    rather than a point target, since surface atoms legitimately have very few.
    """
    per_atom = burial * mask_j
    mean = per_atom.sum(-1) / mask_j.sum(-1).clamp(min=1.0)
    return F.relu((mean - target).abs() - tol).mean()


def distogram_ce(logits: torch.Tensor, target_bins: torch.Tensor,
                 valid: torch.Tensor, label_smoothing: float = 0.02) -> torch.Tensor:
    """The supervised anchor that must stay on through EVERY stage (Section 0.3.4).

    logits (B, n_i, n_j, N_BINS), target_bins (B, n_i, n_j) long, valid (B, n_i, n_j).
    """
    b, ni, nj, nb = logits.shape
    ls = F.cross_entropy(logits.reshape(-1, nb), target_bins.reshape(-1),
                         reduction="none", label_smoothing=label_smoothing)
    v = valid.reshape(-1).to(ls.dtype)
    return (ls * v).sum() / v.sum().clamp(min=1.0)


@torch.no_grad()
def distogram_calibration(P: torch.Tensor, target_bins: torch.Tensor,
                          valid: torch.Tensor, n_buckets: int = 10) -> dict:
    """ECE plus contact recovery - the Section 0.3.4 "is the physics being gamed?" check.

    If calibration degrades while affinity improves, the model is buying pK by
    distorting geometry and the result is not a physics result.
    """
    v = valid.bool()
    if v.sum() == 0:
        return {}
    p = P[v]                                                 # (N, N_BINS)
    t = target_bins[v]                                       # (N,)
    conf, pred = p.max(-1)
    correct = (pred == t).float()
    edges = torch.linspace(0, 1, n_buckets + 1, device=p.device)
    ece = torch.zeros((), device=p.device)
    for a, bnd in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= bnd)
        if m.any():
            ece = ece + m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    true_contact = t < CONTACT_BIN_CUTOFF
    pred_contact = p[:, :CONTACT_BIN_CUTOFF].sum(-1) > 0.5
    tp = (true_contact & pred_contact).sum().float()
    return {
        "disto/ece": float(ece),
        "disto/top1": float(correct.mean()),
        "disto/nll": float(F.nll_loss(torch.log(p.clamp_min(1e-9)), t)),
        "disto/contact_recall": float(tp / true_contact.sum().clamp(min=1)),
        "disto/contact_precision": float(tp / pred_contact.sum().clamp(min=1)),
        "disto/mean_entropy": float((-(p * torch.log(p.clamp_min(1e-9))).sum(-1)).mean()),
    }
