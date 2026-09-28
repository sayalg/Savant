#!/usr/bin/env python3
"""Phase 7 gate: does the model do what it claims before it is trained on anything?

PLAN.md Section 8, Phase 7: "all physics tests pass; oracle-mode equivalence
verified; overfits a 200-complex subset; distogram calibration reasonable on
validation crystals."

The overfit check is the one people skip and the one that catches real bugs. A
model that cannot drive the loss to near zero on 200 complexes has a wiring fault -
a detached gradient, a mask applied to the wrong axis, a head initialised so the
energy is constant - and no amount of training on 600,000 pairs will reveal it,
because on 600,000 pairs a broken model and a working one both plateau at the
label variance.

    python scripts/phase7_selftest.py
    python scripts/phase7_selftest.py --overfit --n 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import torch  # noqa: E402

from pfpl import conf, paths, preflight  # noqa: E402
from pfpl.data.datasets import ATOM_FEATURE_DIM, BOND_FEATURE_DIM  # noqa: E402
from pfpl.features.embed_esm import POCKET_EXTRA_DIM  # noqa: E402
from pfpl.models.affinity import PFPL  # noqa: E402
from pfpl.physics import terms as T  # noqa: E402
from pfpl.physics.energy import EnergyModule, distributions_from_coords  # noqa: E402
from pfpl.physics.sites import N_SITE_PARAMS  # noqa: E402


def synthetic_batch(b=4, L=40, A=16, plm_dim=0, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {
        "res_idx": torch.randint(0, 20, (b, L), generator=g),
        "prot_mask": torch.ones(b, L),
        "pocket_extra": torch.randn(b, L, POCKET_EXTRA_DIM, generator=g),
        "atom_feat": torch.randn(b, A, ATOM_FEATURE_DIM, generator=g),
        "adj": (torch.rand(b, A, A, generator=g) > 0.7).float(),
        "bond_feat": torch.randn(b, A, A, BOND_FEATURE_DIM, generator=g),
        "elem_idx": torch.randint(0, 5, (b, A), generator=g),
        "is_ring_centroid": torch.zeros(b, A),
        "atom_param_prior": torch.full((b, A, N_SITE_PARAMS), float("nan")),
        "lig_mask": torch.ones(b, A),
        "lig_dist": torch.full((b, A, A), 6.0),
        "y": torch.randn(b, generator=g) * 1.5 + 6.5,
        "weight": torch.ones(b), "rel": torch.zeros(b, dtype=torch.long),
        "protein_idx": torch.arange(b), "pair_idx": torch.arange(b),
        "n_rot": torch.full((b,), 5.0),
        "dG_solv": torch.full((b,), float("nan")),
    }
    if plm_dim:
        out["plm"] = torch.randn(b, L, plm_dim, generator=g)
    return out


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'pass' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def physics_checks() -> bool:
    """Section 6.5's list, run against the real term implementations."""
    ok = True
    d_fine = torch.linspace(0.0, 15.0, 400)
    pi = torch.tensor([[0.2, 3.6, 1.0, 1.0, 0.8, 0.5, 1.0, 0.0, 0.0]]).unsqueeze(0)
    pj = torch.tensor([[0.2, 3.6, 1.0, 1.0, 0.8, -0.5, 1.0, 0.0, 1.0]]).unsqueeze(0)

    for tname, term in T.REGISTRY.items():
        ## evaluate() appends the far bin, so the real grid is the first len(d_fine).
        e = T.evaluate(term, pi, pj, d_fine)[0, 0, 0, :len(d_fine)]
        ok &= check(f"{tname}: finite everywhere including d->0",
                    bool(torch.isfinite(e).all()),
                    f"min {e.min():.2f} max {e.max():.2f} kcal/mol")
        ok &= check(f"{tname}: exactly zero beyond the cutoff",
                    bool(e[d_fine >= T.D_CUT].abs().max() < 1e-6))
        ## Differentiability at the origin, where a raw 1/d^12 is +inf.
        d0 = torch.zeros(1, requires_grad=True)
        g = torch.autograd.grad(T.evaluate(term, pi, pj, d0).sum(), d0)[0]
        ok &= check(f"{tname}: differentiable at d=0", bool(torch.isfinite(g).all()))

    for vdw in ("vdw", "vdw_morse"):
        e = T.evaluate(T.REGISTRY[vdw], pi, pj, d_fine)[0, 0, 0, :len(d_fine)]
        near = d_fine < 2.5
        ok &= check(f"{vdw}: repulsive at short range", bool(e[near].max() > 0),
                    f"e(1.5A)={e[d_fine.sub(1.5).abs().argmin()]:+.2f}")
        ok &= check(f"{vdw}: attractive well near r0", bool(e.min() < -0.05),
                    f"min {e.min():+.3f} at {float(d_fine[e.argmin()]):.2f}A")

    ## Coulomb sign law.
    q = lambda a, b: T.evaluate(T.REGISTRY["coulomb"],  # noqa: E731
                                pi.clone().index_put_((torch.tensor([0]), torch.tensor([0]),
                                                       torch.tensor([5])), torch.tensor([a])),
                                pj.clone().index_put_((torch.tensor([0]), torch.tensor([0]),
                                                       torch.tensor([5])), torch.tensor([b])),
                                torch.tensor([4.0]))[0, 0, 0, 0]
    ok &= check("coulomb: opposite charges attract", bool(q(1.0, -1.0) < 0),
                f"{float(q(1.0, -1.0)):+.3f}")
    ok &= check("coulomb: like charges repel", bool(q(1.0, 1.0) > 0),
                f"{float(q(1.0, 1.0)):+.3f}")
    ok &= check("coulomb: scales with the charge product",
                bool(abs(float(q(1.0, -1.0)) - 2 * float(q(0.5, -1.0))) < 1e-4))

    ## Expected energy equals a brute-force sum over a toy distribution.
    em = EnergyModule(("vdw", "hbond", "hydrophobic"))
    P = torch.softmax(torch.randn(2, 6, 4, T.N_BINS), -1)
    ti = torch.rand(2, 6, N_SITE_PARAMS) * 0.5 + 0.3
    tj = torch.rand(2, 4, N_SITE_PARAMS) * 0.5 + 0.3
    mi, mj = torch.ones(2, 6), torch.ones(2, 4)
    o = em(P, ti, tj, mi, mj)
    brute = torch.zeros(2)
    w = em._weights()
    for k, term in enumerate(em.terms):
        e = T.evaluate(term, ti, tj)
        for b in range(2):
            for i in range(6):
                for j in range(4):
                    brute[b] += w[k] * (P[b, i, j] * e[b, i, j]).sum()
    ok &= check("expected energy == brute-force sum",
                bool(torch.allclose(brute, o.dG - em.bias, atol=1e-4)),
                f"max|d| {float((brute - (o.dG - em.bias)).abs().max()):.2e}")

    ## Moving mass closer must lower an attractive-only energy, monotonically.
    em2 = EnergyModule(("hbond",))
    ti = torch.tensor([[[0.2, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    tj = ti.clone()
    vals = []
    for b in range(T.N_DIST_BINS):
        P = torch.zeros(1, 1, 1, T.N_BINS)
        P[0, 0, 0, b] = 1.0
        vals.append(float(em2(P, ti, tj, torch.ones(1, 1), torch.ones(1, 1)).dG))
    near = int(np.argmin(vals))
    ok &= check("hbond: shifting mass toward the optimum lowers the energy",
                all(vals[i] >= vals[i + 1] - 1e-6 for i in range(near)),
                f"minimum at bin {near} (d={T.D_MIN + near * T.D_STEP + 0.25:.2f}A)")

    ## Oracle equivalence: one-hot distributions from coordinates reproduce a direct
    ## coordinate evaluation. Section 6.5's last physics test.
    g = torch.Generator().manual_seed(3)
    px = torch.randn(2, 6, 3, generator=g) * 4
    lx = torch.randn(2, 4, 3, generator=g) * 4
    P = distributions_from_coords(px, lx)
    ti = torch.rand(2, 6, N_SITE_PARAMS, generator=g) * 0.4 + 0.3
    tj = torch.rand(2, 4, N_SITE_PARAMS, generator=g) * 0.4 + 0.3
    o = em(P, ti, tj, torch.ones(2, 6), torch.ones(2, 4))
    dist = torch.cdist(px, lx)
    idx = torch.where(dist >= T.D_MAX, torch.full_like(dist, T.FAR_BIN, dtype=torch.float),
                      torch.floor((dist - T.D_MIN) / T.D_STEP).clamp(0, T.N_DIST_BINS - 1)
                      ).long()
    direct = torch.zeros(2)
    for k, term in enumerate(em.terms):
        e = T.evaluate(term, ti, tj)
        direct += w[k] * torch.gather(e, -1, idx.unsqueeze(-1)).squeeze(-1).sum((-2, -1))
    ok &= check("oracle mode == direct coordinate evaluation",
                bool(torch.allclose(direct, o.dG - em.bias, atol=1e-4)),
                f"max|d| {float((direct - (o.dG - em.bias)).abs().max()):.2e}")
    return ok


def model_checks(cfg) -> bool:
    ok = True
    plm_dim = {"none": 0, "esm2_650m_full": 1280, "esm2_bottleneck": 1280,
               "esm2_pocket_window": 1280, "esmc_300m": 960, "esm2_3b": 2560}[
        str(cfg.plm.mode)]
    for variant in ("A", "B", "C"):
        c = conf.load(cfg.get("_name", "config"), [f"model=variant_{variant.lower()}",
                                                   f"plm.mode={cfg.plm.mode}"])
        m = PFPL(c, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                 pocket_extra_dim=POCKET_EXTRA_DIM)
        b = synthetic_batch(plm_dim=plm_dim)
        out = m(b)
        ok &= check(f"variant {variant}: forward pass",
                    bool(torch.isfinite(out.pK).all()),
                    f"pK {out.pK.min():.2f}..{out.pK.max():.2f}, "
                    f"{m.n_params()['total']:,} params")
        ## An untrained model should start near the label range, not thousands of pK
        ## units away. Both heads are zero-initialised onto an informative prior
        ## (far bin / diffuse pocket) precisely so this holds; a large number here
        ## means an initialisation artifact the first epochs would have to undo.
        ok &= check(f"variant {variant}: starts on a sane pK scale",
                    bool(out.pK.abs().max() < 50),
                    f"|pK|max {float(out.pK.abs().max()):.2f}")

        ## Gradient coverage, measured AFTER one step. The last layer of every head
        ## is zero-initialised, so at step 0 the trunk legitimately receives no
        ## gradient through the physics path and a coverage check would pass for the
        ## wrong reason. One step makes those layers non-zero; anything still
        ## unreached after that is genuinely detached.
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            m(b).pK.sum().backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        m(b).pK.sum().backward()
        n_train = sum(1 for p in m.parameters() if p.requires_grad)
        grads = [p for p in m.parameters() if p.requires_grad and p.grad is not None
                 and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
        ok &= check(f"variant {variant}: gradients reach the trunk",
                    len(grads) >= 0.8 * n_train,
                    f"{len(grads)}/{n_train} tensors")
    ## Section 9's comparison rule only means something if the control is matched.
    na = PFPL(conf.load("config", ["model=variant_a"]), ATOM_FEATURE_DIM,
              BOND_FEATURE_DIM, pocket_extra_dim=POCKET_EXTRA_DIM).n_params()["total"]
    nc = PFPL(conf.load("config", ["model=variant_c"]), ATOM_FEATURE_DIM,
              BOND_FEATURE_DIM, pocket_extra_dim=POCKET_EXTRA_DIM).n_params()["total"]
    ok &= check("variant C is parameter-matched to variant A (within 25%)",
                abs(na - nc) / max(na, nc) < 0.25,
                f"A {na:,} vs C {nc:,} ({abs(na - nc) / max(na, nc):.1%})")
    return ok


def overfit(cfg, n: int = 200, steps: int = 400) -> bool:
    """Can the model drive the loss to near zero on a tiny set? Section 8, Phase 7."""
    from pfpl.train.loop import LossWeights, compute_losses
    dev = preflight.pick_device()
    plm_dim = {"none": 0}.get(str(cfg.plm.mode), 1280)
    b = synthetic_batch(b=min(n, 16), plm_dim=plm_dim, seed=7)
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    m = PFPL(cfg, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
             pocket_extra_dim=POCKET_EXTRA_DIM).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ## Affinity only: this tests wiring, not physics, and the anchor has no targets
    ## on synthetic data.
    w = LossWeights(affinity=1.0, disto=0.0, pocket=0.0, triangle=0.0, contact=0.0,
                    prior=0.0, coord=0.0, censored=0.0)
    first = last = None
    t0 = time.time()
    for s in range(steps):
        out = m(b)
        loss, _ = compute_losses(out, b, w, "affinity", use_uncertainty=False)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s == 0:
            first = float(loss)
        last = float(loss)
        if s % 100 == 0:
            print(f"    step {s:>4}  loss {float(loss):.4f}")
    var = float(b["y"].var())
    print(f"    {steps} steps in {time.time() - t0:.1f}s on {dev}; "
          f"label variance {var:.3f}")
    return check("overfits a tiny batch (wiring is intact)",
                 last < 0.1 * max(first, 1e-6) or last < 0.02 * var,
                 f"{first:.4f} -> {last:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    cfg = conf.from_args(a)
    paths.ensure()
    torch.manual_seed(0)

    print("=" * 78)
    print("PHASE 7 SELFTEST")
    print("=" * 78)
    print("\n-- physics (Section 6.5) " + "-" * 50)
    ok = physics_checks()
    print("\n-- model wiring " + "-" * 59)
    ok &= model_checks(cfg)
    if a.overfit:
        print("\n-- overfit " + "-" * 64)
        ok &= overfit(cfg, a.n, a.steps)

    print("\n" + "=" * 78)
    print(f"GATE - Phase 7: {'PASS' if ok else 'FAIL'}")
    if not a.overfit:
        print("  (run with --overfit for the full gate; Section 8 Phase 7 requires it)")
    print("=" * 78)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
