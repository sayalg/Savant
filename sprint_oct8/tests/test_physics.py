"""Physics unit tests (PLAN.md Section 6.5), one test per listed requirement.

These are cheap, they run on CPU, and every one of them corresponds to a way the
energy module can be subtly wrong while still training to a plausible number. The
oracle-equivalence test in particular is the only thing that establishes that the
expected-energy path means what the paper will say it means.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
import torch

from pfpl.physics import terms as T
from pfpl.physics.energy import (EnergyModule, RT_LN10, contact_prior_loss,
                                 distributions_from_coords, triangle_penalty)
from pfpl.physics.params import LigandSiteParams, ProteinSiteParams, prior_penalty
from pfpl.physics.sites import (N_SITE_PARAMS, PARAM_RANGE, P_ACC, P_DONOR, P_EPS,
                                P_Q, P_R0, SITE_PARAMS, ligand_element_priors,
                                residue_site_mask, residue_site_priors)

ALL_TERMS = list(T.REGISTRY)
D_FINE = torch.linspace(0.0, 15.0, 600)


def params(**kw) -> torch.Tensor:
    p = torch.tensor([0.2, 3.6, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5])
    for k, v in kw.items():
        p[SITE_PARAMS.index(k)] = v
    return p.view(1, 1, -1)


@pytest.mark.parametrize("name", ALL_TERMS)
def test_finite_and_differentiable_everywhere(name):
    """Including d -> 0, where a raw 12-6 is +inf and one NaN kills a multi-day run."""
    e = T.evaluate(T.REGISTRY[name], params(), params(), D_FINE)
    assert torch.isfinite(e).all(), f"{name} is not finite over 0-15 A"
    d0 = torch.zeros(3, requires_grad=True)
    g = torch.autograd.grad(T.evaluate(T.REGISTRY[name], params(), params(), d0).sum(), d0)[0]
    assert torch.isfinite(g).all(), f"{name} has a non-finite gradient at d=0"


@pytest.mark.parametrize("name", ALL_TERMS)
def test_exactly_zero_beyond_the_cutoff(name):
    """Not "small" - zero. A 1e-4 tail summed over 256 sites x 64 atoms x 21 bins is
    a real, meaningless offset that the global bias would then absorb."""
    ## evaluate() appends the far bin, so the real grid is the first len(D_FINE).
    e = T.evaluate(T.REGISTRY[name], params(), params(), D_FINE)[0, 0, 0, :len(D_FINE)]
    assert e[D_FINE >= T.D_CUT].abs().max() < 1e-7


@pytest.mark.parametrize("name", ALL_TERMS)
def test_far_bin_contributes_nothing(name):
    e = T.evaluate(T.REGISTRY[name], params(), params())
    assert float(e[..., T.FAR_BIN].abs().max()) == 0.0


@pytest.mark.parametrize("vdw", ["vdw", "vdw_morse"])
def test_vdw_is_repulsive_close_and_attractive_at_r0(vdw):
    p = params(eps=0.3, r0=3.6)
    e = T.evaluate(T.REGISTRY[vdw], p, p, D_FINE)[0, 0, 0, :len(D_FINE)]
    assert e[D_FINE < 2.0].max() > 0, f"{vdw} is not repulsive at short range"
    assert e.min() < -0.05, f"{vdw} has no attractive well"
    dmin = float(D_FINE[e.argmin()])
    assert 2.5 < dmin < 5.5, f"{vdw} minimum at {dmin:.2f} A is not near r0=3.6"


def test_coulomb_sign_law():
    d = torch.tensor([4.0])
    f = lambda a, b: float(T.evaluate(  # noqa: E731
        T.REGISTRY["coulomb"], params(charge=a), params(charge=b), d)[0, 0, 0, 0])
    assert f(1.0, -1.0) < 0, "opposite charges must attract"
    assert f(-1.0, 1.0) < 0
    assert f(1.0, 1.0) > 0, "like charges must repel"
    assert f(-1.0, -1.0) > 0
    assert abs(f(1.0, -1.0) - 2 * f(0.5, -1.0)) < 1e-5, "must scale with the product"
    assert abs(f(0.0, 1.0)) < 1e-9


def test_hbond_needs_a_donor_and_an_acceptor():
    d = torch.tensor([2.95])
    both = float(T.evaluate(T.REGISTRY["hbond"], params(donor=1, acceptor=0),
                            params(donor=0, acceptor=1), d)[0, 0, 0, 0])
    neither = float(T.evaluate(T.REGISTRY["hbond"], params(donor=1, acceptor=0),
                               params(donor=1, acceptor=0), d)[0, 0, 0, 0])
    assert both < -0.5, "donor-acceptor must be attractive at 2.95 A"
    assert abs(neither) < 1e-6, "donor-donor must not bond"


def test_expected_energy_equals_brute_force():
    torch.manual_seed(0)
    em = EnergyModule(("vdw", "hbond", "hydrophobic"))
    P = torch.softmax(torch.randn(3, 5, 4, T.N_BINS), -1)
    ti = torch.rand(3, 5, N_SITE_PARAMS) * 0.4 + 0.3
    tj = torch.rand(3, 4, N_SITE_PARAMS) * 0.4 + 0.3
    out = em(P, ti, tj, torch.ones(3, 5), torch.ones(3, 4))
    w = em._weights()
    brute = torch.zeros(3)
    for k, term in enumerate(em.terms):
        e = T.evaluate(term, ti, tj)
        for b in range(3):
            for i in range(5):
                for j in range(4):
                    brute[b] += w[k] * float((P[b, i, j] * e[b, i, j]).sum())
    assert torch.allclose(brute, out.dG - em.bias, atol=1e-4)


def test_oracle_mode_equals_direct_coordinate_evaluation():
    """Section 6.5's last physics requirement, and the basis for the Section 6.6
    cost-of-pose-free ablation being a like-for-like comparison."""
    torch.manual_seed(1)
    em = EnergyModule(("vdw", "hbond", "hydrophobic"))
    px, lx = torch.randn(2, 6, 3) * 4, torch.randn(2, 4, 3) * 4
    ti = torch.rand(2, 6, N_SITE_PARAMS) * 0.4 + 0.3
    tj = torch.rand(2, 4, N_SITE_PARAMS) * 0.4 + 0.3
    out = em(distributions_from_coords(px, lx), ti, tj, torch.ones(2, 6), torch.ones(2, 4))

    dist = torch.cdist(px, lx)
    idx = torch.where(dist >= T.D_MAX,
                      torch.full_like(dist, T.FAR_BIN),
                      torch.floor((dist - T.D_MIN) / T.D_STEP).clamp(0, T.N_DIST_BINS - 1)
                      ).long()
    w = em._weights()
    direct = torch.zeros(2)
    for k, term in enumerate(em.terms):
        e = T.evaluate(term, ti, tj)
        direct += w[k] * torch.gather(e, -1, idx.unsqueeze(-1)).squeeze(-1).sum((-2, -1))
    assert torch.allclose(direct, out.dG - em.bias, atol=1e-4)


def test_moving_mass_closer_lowers_an_attractive_energy_monotonically():
    em = EnergyModule(("hbond",))
    ti = params(donor=1.0, acceptor=1.0).view(1, 1, -1)
    vals = []
    for b in range(T.N_DIST_BINS):
        P = torch.zeros(1, 1, 1, T.N_BINS)
        P[0, 0, 0, b] = 1.0
        vals.append(float(em(P, ti, ti, torch.ones(1, 1), torch.ones(1, 1)).dG))
    lo = int(np.argmin(vals))
    assert all(vals[i] >= vals[i + 1] - 1e-6 for i in range(lo)), \
        "energy must fall monotonically as mass moves toward the optimum"
    assert all(vals[i] <= vals[i + 1] + 1e-6 for i in range(lo, len(vals) - 1))


def test_masked_sites_contribute_nothing():
    torch.manual_seed(2)
    em = EnergyModule(("vdw", "hbond"))
    P = torch.softmax(torch.randn(1, 4, 3, T.N_BINS), -1)
    ti = torch.rand(1, 4, N_SITE_PARAMS) * 0.4 + 0.3
    tj = torch.rand(1, 3, N_SITE_PARAMS) * 0.4 + 0.3
    full = em(P, ti, tj, torch.ones(1, 4), torch.ones(1, 3)).dG
    m = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    masked = em(P, ti, tj, m, torch.ones(1, 3)).dG
    sub = em(P[:, :3], ti[:, :3], tj, torch.ones(1, 3), torch.ones(1, 3)).dG
    assert torch.allclose(masked, sub, atol=1e-5)
    assert not torch.allclose(masked, full, atol=1e-5)


def test_pK_sign_convention():
    """A more negative dG must mean a HIGHER pK, or every gate reads backwards."""
    em = EnergyModule(("hbond",))
    strong = params(donor=1.0, acceptor=1.0).view(1, 1, -1)
    weak = params(donor=0.0, acceptor=0.0).view(1, 1, -1)
    P = torch.zeros(1, 1, 1, T.N_BINS)
    P[0, 0, 0, 1] = 1.0                      # ~2.75 A, in the H-bond well
    a = em(P, strong, strong, torch.ones(1, 1), torch.ones(1, 1))
    b = em(P, weak, weak, torch.ones(1, 1), torch.ones(1, 1))
    assert float(a.dG) < float(b.dG)
    assert float(a.pK) > float(b.pK)
    assert abs(RT_LN10 - 1.3642) < 1e-3, "RT ln10 must be 1.364 kcal/mol per pK unit"


def test_parameter_heads_start_at_the_prior_and_stay_in_range():
    torch.manual_seed(3)
    head = ProteinSiteParams(32)
    head.eval()
    res = torch.randint(0, 20, (2, 7))
    theta, mask = head(torch.randn(2, 7, 32), res)
    prior = torch.from_numpy(residue_site_priors())[res]
    ## atol=1e-3, not 0: unsquash() clamps the logit to [1e-4, 1-1e-4] so it stays
    ## finite, which costs at most 1e-4 * range on a prior sitting exactly on a range
    ## boundary (donor=0, acceptor=1). Everything strictly inside is exact.
    assert torch.allclose(theta, prior, atol=1e-3), \
        "an untrained head must evaluate the tabulated force field"
    for i, k in enumerate(SITE_PARAMS):
        lo, hi = PARAM_RANGE[k]
        assert float(theta[..., i].min()) >= lo - 1e-4
        assert float(theta[..., i].max()) <= hi + 1e-4
    ## GLY has no side chain and ALA no distinct terminal atom; emitting sites for
    ## them would put H-bond capacity on groups that do not exist.
    m = residue_site_mask()
    assert not m[0][2], "ALA must have no TIP site"
    assert not m[5][1] and not m[5][2], "GLY must have no SC or TIP site"


def test_parameter_heads_cannot_be_pushed_out_of_range():
    head = ProteinSiteParams(8, scale=2.0)
    with torch.no_grad():
        for p in head.head.net[-1].parameters():
            p.fill_(50.0)          # absurd activation
    theta, _ = head(torch.randn(1, 3, 8) * 100, torch.randint(0, 20, (1, 3)))
    for i, k in enumerate(SITE_PARAMS):
        lo, hi = PARAM_RANGE[k]
        assert lo - 1e-4 <= float(theta[..., i].min())
        assert float(theta[..., i].max()) <= hi + 1e-4


def test_ligand_feature_prior_overrides_the_element_table():
    head = LigandSiteParams(16)
    head.eval()
    elem = torch.zeros(1, 2, dtype=torch.long)          # carbon
    feat = torch.full((1, 2, N_SITE_PARAMS), float("nan"))
    feat[0, 0, P_ACC] = 1.0
    theta = head(torch.randn(1, 2, 16), elem, feat_prior=feat)
    assert float(theta[0, 0, P_ACC]) > float(theta[0, 1, P_ACC]), \
        "an RDKit-derived acceptor flag must beat the element table"
    assert abs(float(theta[0, 1, P_EPS]) - float(ligand_element_priors()[0, P_EPS])) < 1e-4


def test_triangle_penalty_fires_on_an_impossible_geometry():
    """|E[d_ij] - E[d_ik]| <= d_jk. The Section 2.1 mitigation for Variant A."""
    B, ni, nj = 1, 2, 2
    P = torch.zeros(B, ni, nj, T.N_BINS)
    P[:, :, 0, 0] = 1.0                    # atom 0 very close
    P[:, :, 1, T.N_DIST_BINS - 1] = 1.0    # atom 1 very far
    tight = torch.full((B, nj, nj), 1.0)   # but the ligand says they are 1 A apart
    loose = torch.full((B, nj, nj), 12.0)
    mi, mj = torch.ones(B, ni), torch.ones(B, nj)
    g = torch.Generator().manual_seed(0)
    v_bad = float(triangle_penalty(P, tight, mi, mj, 64, g))
    v_ok = float(triangle_penalty(P, loose, mi, mj, 64,
                                  torch.Generator().manual_seed(0)))
    assert v_bad > 1.0, "an impossible geometry must be penalised"
    assert v_ok < 1e-6, "a consistent geometry must not be"


def test_contact_prior_bands_rather_than_pins():
    burial = torch.full((2, 10), 6.0)
    assert float(contact_prior_loss(burial, torch.ones(2, 10))) == 0.0
    assert float(contact_prior_loss(torch.full((2, 10), 40.0), torch.ones(2, 10))) > 0


def test_prior_penalty_is_range_normalised():
    """Unnormalised, `charge` (span 2.0) dominates and `eps` (span 0.58) is ignored."""
    def cost(idx: int, key: str) -> float:
        lo, hi = PARAM_RANGE[key]
        prior = torch.zeros(1, 4, N_SITE_PARAMS)
        theta = torch.zeros(1, 4, N_SITE_PARAMS)
        prior[..., idx] = lo
        theta[..., idx] = hi          # a full-range move on exactly one parameter
        return float(prior_penalty(theta, prior))

    a, b = cost(P_Q, "charge"), cost(P_EPS, "eps")
    assert abs(a - b) < 0.05 * max(a, b), \
        f"a full-range move costs {a:.4f} on charge but {b:.4f} on eps"


def test_distance_to_onehot_round_trips():
    d = torch.tensor([[[0.5, 2.0, 2.4, 6.0, 11.9, 12.0, 50.0]]])
    oh = T.distance_to_onehot(d)
    assert torch.allclose(oh.sum(-1), torch.ones_like(d))
    idx = oh.argmax(-1)[0, 0]
    assert idx[0] == 0, "a clash below D_MIN must land in the first bin, not be dropped"
    assert idx[1] == 0 and idx[2] == 0
    assert idx[3] == int((6.0 - T.D_MIN) / T.D_STEP)
    assert idx[4] == T.N_DIST_BINS - 1
    assert idx[5] == T.FAR_BIN and idx[6] == T.FAR_BIN


def test_switch_is_continuous_at_both_ends():
    assert abs(float(T.switch(torch.tensor([T.D_SWITCH - 1e-4]))) - 1.0) < 1e-5
    assert abs(float(T.switch(torch.tensor([T.D_CUT + 1e-4])))) < 1e-9
    d = torch.linspace(T.D_SWITCH - 1, T.D_CUT + 1, 300)
    s = T.switch(d)
    assert (s.diff() <= 1e-9).all(), "the switch must be monotone decreasing"


def test_attractive_weights_stay_positive_under_pressure():
    """Section 6.2 constrains attractive terms. Without it the model can flip the
    sign of the vdW term to fit a label and call the result physics."""
    em = EnergyModule(("vdw", "hbond", "hydrophobic"))
    with torch.no_grad():
        em.w.fill_(-50.0)
    w = em._weights()
    assert (w > 0).all(), f"attractive weights went non-positive: {w}"


def test_coulomb_weight_is_free():
    """Coulomb's sign is not fixed, so Section 6.2 does not constrain its weight."""
    em = EnergyModule(("coulomb",))
    with torch.no_grad():
        em.w.fill_(-2.0)
    assert float(em._weights()[0]) < 0


def test_resolve_terms_honours_the_vdw_form_ablation():
    assert T.resolve_terms(["vdw"], "lj")[0].name == "vdw"
    assert T.resolve_terms(["vdw"], "morse")[0].name == "vdw_morse"
    with pytest.raises(SystemExit):
        T.resolve_terms(["nonsense"], "lj")


def test_ligand_site_distances_index_like_the_graph():
    """The triangle penalty reads lig_dist[j, k] for graph sites j, k. If the
    conformer cache and build_graph ever disagree on site order, the penalty
    constrains the wrong pairs and nothing downstream notices."""
    pytest.importorskip("rdkit")
    from pfpl.features import ligand_phys as LP
    from pfpl.features.embed_ligand import build_graph
    ## written non-canonically on purpose, with a 3-ring, a 6-ring and a fused system
    for smi in ("OC(=O)c1ccccc1OC(C)=O", "C1CC1C(=O)Nc1ccc2ccccc2c1", "c1ccccc1CCN"):
        d = LP.tier_a(smi, n_conf=4, threads=1)
        g = build_graph(smi, max_atoms=999)
        n = d["_n_sites"]
        assert n == g["n_sites"]
        m = np.frombuffer(d["_site_dist"], np.float16).reshape(n, n).astype(float)
        h = g["n_heavy"]
        bonded = m[:h, :h][g["adj"][:h, :h] > 0]
        assert 1.0 < bonded.min() and bonded.max() < 2.1, "heavy-atom order drifted"
        for k in np.where(g["ring_flag"] > 0)[0]:
            members = m[k, :h][g["adj"][k, :h] > 0]
            assert 0.8 < members.min() and members.max() < 1.9, "ring order drifted"
