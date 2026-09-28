"""The assembled model, and the matched no-physics control.

Four variants share one trunk, one pocket head and one pair module, so that Phase 9
compares the *energy function* and nothing else (PLAN.md Sections 2.2, 8):

    A  independent distance distributions + penalties      model.variant=A
    B  coordinate hypotheses, consistent by construction   model.variant=B
    C  no physics: pooled pair rep -> MLP -> pK            model.variant=C
    D  oracle: distributions from real coordinates         energy.mode=oracle

Variant C is the control every physics claim is measured against, and Section 9's
comparison rule is explicit: physics must beat it by dPearson >= 0.02 on the
protein-cluster split with a bootstrap CI excluding zero. It is built here rather
than in a separate file precisely so it cannot drift into being a different model
with a different capacity.

The model also emits a heteroscedastic log-variance (Section 1: "predicted pKd,
uncertainty, and an interpretable decomposition") and the per-term / per-residue /
per-atom decompositions from the energy module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..physics import terms as T
from ..physics.energy import EnergyModule, EnergyOut
from ..physics.params import LigandSiteParams, ProteinSiteParams
from .coord_head import CoordinateHypothesisHead
from .distogram import DistogramHead
from .pair import PairModule, pool_pair
from .pocket import PocketHead
from .trunk import LigandTrunk, ProteinTrunk


@dataclass
class ModelOut:
    pK: torch.Tensor
    logvar: torch.Tensor
    pocket_logits: torch.Tensor
    pocket_idx: torch.Tensor
    disto_logits: torch.Tensor | None
    P: torch.Tensor | None
    energy: EnergyOut | None
    theta_i: torch.Tensor | None = None
    theta_j: torch.Tensor | None = None
    prior_i: torch.Tensor | None = None
    prior_j: torch.Tensor | None = None
    site_mask: torch.Tensor | None = None
    coord_d: torch.Tensor | None = None


class PFPL(nn.Module):
    def __init__(self, cfg, atom_dim: int, bond_dim: int, geom_prior_dim: int = 0,
                 pocket_extra_dim: int = 0):
        super().__init__()
        m, p = cfg.model, cfg.plm
        self.variant = str(m.get("variant", "A")).upper()
        self.energy_mode = str(cfg.get_path("model.energy.mode", "predicted"))
        self.topk = int(m.get("topk", 64))

        d = int(m.get("d_model", 256))
        pair_dim = int(m.get("pair_dim", 96))
        drop = float(m.get("dropout", 0.1))

        self.prot = ProteinTrunk(mode=str(p.mode), d_model=d,
                                 bottleneck=int(p.get("bottleneck", 64)),
                                 dropout=drop, noise_std=float(p.get("noise_std", 0.1)))
        self.lig = LigandTrunk(atom_dim, bond_dim, d_model=d,
                               clm_dim=int(m.get("clm_dim", 0)),
                               layers=int(m.get("ligand_layers", 3)), dropout=drop)
        self.pocket = PocketHead(self.prot.base_dim, d, pocket_extra_dim,
                                 hidden=int(m.get("pocket_hidden", 128)),
                                 dropout=drop, topk=self.topk)
        self.pair = PairModule(d, pair_dim, blocks=int(m.get("pair_blocks", 3)),
                               heads=int(m.get("heads", 8)), dropout=drop,
                               geom_prior_dim=geom_prior_dim)

        self.theta_p = ProteinSiteParams(d, hidden=int(m.get("param_hidden", 128)),
                                         scale=float(m.get("param_scale", 2.0)),
                                         dropout=drop)
        self.theta_l = LigandSiteParams(d, hidden=int(m.get("param_hidden", 128)),
                                        scale=float(m.get("param_scale", 2.0)),
                                        dropout=drop)

        ctx_dim = 2 * pair_dim
        self.logvar = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, 64),
                                    nn.GELU(), nn.Linear(64, 1))

        if self.variant == "C":
            ## Matched control. Same trunk, same pocket head, same pair module; the
            ## ONLY difference is that a generic MLP replaces the energy function.
            ## Sized to land within ~10% of the physics head's parameter count.
            self.head = nn.Sequential(
                nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, 256), nn.GELU(),
                nn.Dropout(drop), nn.Linear(256, 128), nn.GELU(),
                nn.Dropout(drop), nn.Linear(128, 1))
            self.disto = self.coord = self.energy = None
        else:
            e = cfg.get_path("model.energy", {}) or {}
            self.energy = EnergyModule(
                term_names=tuple(e.get("terms", T.MVP_TERMS)),
                vdw_form=str(e.get("vdw_form", "lj")),
                use_rotor=bool(e.get("rotor", True)),
                use_desolvation=bool(e.get("desolvation", False)),
                use_entropy=bool(e.get("entropy", False)),
                residual_dim=ctx_dim if bool(e.get("residual", False)) else 0,
                residual_clamp=float(e.get("residual_clamp", 2.0)))
            self.disto = DistogramHead(pair_dim, dropout=drop)
            self.coord = (CoordinateHypothesisHead(
                d, pair_dim, n_hypotheses=int(m.get("n_hypotheses", 4)), dropout=drop)
                if self.variant == "B" else None)
            self.head = None

    # ----------------------------------------------------------------- forward
    def forward(self, b: dict) -> ModelOut:
        res_idx, prot_mask = b["res_idx"], b["prot_mask"]
        lig_mask = b["lig_mask"]

        lig = self.lig(b["atom_feat"], b["adj"], b["bond_feat"], lig_mask,
                       b.get("lig_clm"))
        base = self.prot.base(res_idx)
        pl = self.pocket(base, lig, lig_mask, prot_mask, b.get("pocket_extra"))
        idx, pk_mask, window = self.pocket.select(pl, prot_mask, self.topk)

        prot = self.prot(res_idx, b.get("plm"), window)
        gi = idx.unsqueeze(-1).expand(-1, -1, prot.shape[-1])
        pocket_emb = torch.gather(prot, 1, gi)                       # (B,K,d)
        pocket_res = torch.gather(res_idx, 1, idx)                   # (B,K)

        z, p2, l2 = self.pair(pocket_emb, lig, pk_mask, lig_mask, b.get("geom_prior"))
        ctx = pool_pair(z, pk_mask, lig_mask)
        logvar = self.logvar(ctx).squeeze(-1).clamp(-6.0, 3.0)

        if self.variant == "C":
            return ModelOut(pK=self.head(ctx).squeeze(-1), logvar=logvar,
                            pocket_logits=pl, pocket_idx=idx,
                            disto_logits=None, P=None, energy=None)

        theta_i, kind_mask = self.theta_p(p2, pocket_res)            # (B,K,4,P),(B,K,4)
        kind_mask = kind_mask * pk_mask.unsqueeze(-1)
        bsz, k, nk, npar = theta_i.shape
        theta_i_f = theta_i.reshape(bsz, k * nk, npar)
        mask_i = kind_mask.reshape(bsz, k * nk).to(theta_i.dtype)
        prior_i = self.theta_p.prior[pocket_res].reshape(bsz, k * nk, npar)

        theta_j = self.theta_l(l2, b["elem_idx"], b.get("is_ring_centroid"),
                               b.get("atom_param_prior"))
        prior_j = self.theta_l.prior[b["elem_idx"]]

        disto_logits, P, coord_d = None, None, None
        if self.energy_mode == "oracle":
            ## Section 6.6: same heads, same terms, real geometry. The gap to
            ## `predicted` is the price of dropping the pose.
            if "true_dist" not in b:
                raise SystemExit("energy.mode=oracle needs `true_dist` in the batch; "
                                 "only structure-bearing splits can run oracle mode")
            P = T.distance_to_onehot(b["true_dist"])
        elif self.variant == "B":
            px, lx, w = self.coord(p2, l2, ctx)
            P, dh = self.coord.distributions(px, lx, w)
            coord_d = dh
            ## Variant B still emits logits so the SAME distogram anchor, the same
            ## calibration monitor and the same reporting apply to both variants.
            disto_logits = torch.log(P.clamp_min(1e-9))
        else:
            disto_logits, flat_mask = self.disto(z, kind_mask, lig_mask)
            P = DistogramHead.probabilities(disto_logits)

        e = self.energy(P, theta_i_f, theta_j, mask_i, lig_mask,
                        n_rot=b.get("n_rot"), dG_solv=b.get("dG_solv"), context=ctx)
        return ModelOut(pK=e.pK, logvar=logvar, pocket_logits=pl, pocket_idx=idx,
                        disto_logits=disto_logits, P=P, energy=e,
                        theta_i=theta_i_f, theta_j=theta_j,
                        prior_i=prior_i, prior_j=prior_j,
                        site_mask=mask_i, coord_d=coord_d)

    # --------------------------------------------------------------- bookkeeping
    def n_params(self) -> dict:
        def c(m):
            return 0 if m is None else sum(p.numel() for p in m.parameters())
        return {"total": c(self), "protein_trunk": c(self.prot), "ligand_trunk": c(self.lig),
                "pocket": c(self.pocket), "pair": c(self.pair),
                "energy_or_head": c(self.energy) + c(self.head) + c(self.disto)
                + c(self.coord) + c(self.theta_p) + c(self.theta_l)}
