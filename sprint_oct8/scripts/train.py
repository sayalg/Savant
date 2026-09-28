#!/usr/bin/env python3
"""Train one configuration (PLAN.md Section 8, Phases 7-8).

One entry point for every variant and every stage; the config decides which.
Everything it needs already exists by Phase 5, so this script does no featurisation
and no splitting - it loads, trains, scores on VAL, and writes a run directory.

    python scripts/train.py -o model=variant_a -o plm=bottleneck
    python scripts/train.py -o model=variant_c              # the matched control
    python scripts/train.py -o train=stage1_geometry        # geometry only
    python scripts/train.py -o model=oracle --split double  # cost-of-pose-free

Model selection uses the VALIDATION fold of the configured split and nothing else
(Section 0.4.3). The test fold is scored here for the ablation tables; the SEALED
benchmark is scored only by `scripts/phase10_benchmark.py`, once per model version.

Every run appends a row to `reports/compute_ledger.csv` with its config hash, so
Section 7.4's multiplicity question - "how many configurations produced this
number?" - has an answer.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
import pandas as pd  # noqa: E402

from pfpl import conf, ledger, paths, preflight  # noqa: E402
from pfpl.data import splits as SP  # noqa: E402
from pfpl.data.datasets import (ATOM_FEATURE_DIM, BOND_FEATURE_DIM,  # noqa: E402
                                PairDataset, ProteinStore, make_loader)
from pfpl.eval import metrics as M  # noqa: E402
from pfpl.features.embed_esm import POCKET_EXTRA_DIM  # noqa: E402
from pfpl.models.affinity import PFPL  # noqa: E402
from pfpl.train.loop import Trainer  # noqa: E402


def run_name(cfg) -> str:
    return cfg.get("run_name") or (
        f"{cfg.model.variant}_{cfg.plm.mode}_{cfg.data.split}"
        f"_{cfg.train.get('stage', 'affinity')}_s{cfg.seed}_{cfg.hash}")


def load_geometry(pairs: pd.DataFrame) -> dict:
    """Map crystal geometry onto pair rows by (sequence_hash, ligand) where possible.

    Only rows that carry it get an anchor gradient; Section 8's Stage 2 trains on
    mixed batches for exactly this reason.
    """
    p = paths.WORK / "geometry" / "geometry.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        labels = pickle.load(fh)
    by_hash: dict[str, list] = {}
    for L in labels:
        by_hash.setdefault(L["sequence_hash"], []).append(L)
    out = {}
    for i, h in enumerate(pairs.sequence_hash.astype(str)):
        if h in by_hash:
            out[i] = by_hash[h][0]
    print(f"  geometry: {len(labels):,} complexes, attached to {len(out):,} pairs "
          f"({len(out) / max(len(pairs), 1):.1%})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--split", default=None, help="override data.split")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap the pair count")
    ap.add_argument("--allow-unanchored", action="store_true",
                    help="deliberate ablation: run an affinity stage with the "
                         "distogram anchor off. Recorded in run.json.")
    ap.add_argument("--no-geometry", action="store_true")
    a = ap.parse_args()

    cfg = conf.from_args(a)
    if a.split:
        cfg["data"]["split"] = a.split
        cfg["hash"] = conf.config_hash(cfg)
    paths.ensure()
    import torch
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    name = run_name(cfg)
    out = paths.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"TRAIN  {name}")
    print("=" * 78)
    conf.dump(cfg, out / "config.json")

    stage = str(cfg.train.get("stage", "affinity"))
    if stage == "physics":
        ## NOT IMPLEMENTED, and saying so beats a silent no-op: with stage="physics"
        ## the shared loop computes no affinity loss and no geometry loss, so the run
        ## would train on nothing and still write a plausible-looking run directory.
        raise SystemExit(
            "Stage 0 (QM dimer pretraining) is NOT implemented in this codebase.\n\n"
            "PLAN.md Section 8 lists it as an ABLATION (on/off), not a prerequisite,\n"
            "and it needs a SAPT-decomposed dimer set that is not staged here:\n"
            "  * Splinter    ~1.7M configs of >9,000 protein-ligand-relevant dimers\n"
            "  * DES370K     SAPT0/aVTZ components\n"
            "  * SPICE       off-equilibrium energies and forces\n"
            "Appendix A also flags that the component sets must be VERIFIED before\n"
            "the Stage 0 losses are designed against them.\n\n"
            "To add it: implement src/pfpl/physics/qm_pretrain.py to regress the SAPT\n"
            "components onto the site-parameter heads (electrostatics <-> Coulomb,\n"
            "exchange <-> repulsion, dispersion <-> attraction), then teach\n"
            "train/loop.py's compute_losses about the `physics` stage.\n\n"
            "Everything else runs without it. Report the ablation as not-run rather\n"
            "than as no-effect.")

    ## ---- data ---------------------------------------------------------------
    pairs = pd.read_parquet(paths.WORK / "pairs.parquet").reset_index(drop=True)
    prot = pd.read_parquet(paths.WORK / "proteins.parquet").sort_values("protein_idx")
    phys_p = paths.WORK / "ligand_phys.parquet"
    if phys_p.exists():
        lp = pd.read_parquet(phys_p)
        keep = [c for c in ("ligand_idx", "n_rot", "dG_solv") if c in lp.columns]
        pairs = pairs.merge(lp[keep], on="ligand_idx", how="left")

    tr, va, te = SP.load(paths.SPLITS, str(cfg.data.split))
    if a.limit:
        ## Keep the fold structure while shrinking: a debug run that silently
        ## reshuffles the split is worse than no debug run.
        rng = np.random.default_rng(0)
        for m in (tr, va, te):
            idx = np.where(m)[0]
            if len(idx) > a.limit:
                m[rng.choice(idx, len(idx) - a.limit, replace=False)] = False

    geometry = {} if a.no_geometry else load_geometry(pairs)
    if stage == "geometry":
        keep = np.zeros(len(pairs), bool)
        keep[list(geometry)] = True
        if not keep.any():
            raise SystemExit(
                "train=stage1_geometry needs structural data, and none is attached.\n"
                "  Run scripts/phase2_geometry.py --pdbbind <dir> then --merge.\n\n"
                "  Without it there is no distogram anchor, and PLAN.md Section 2.1\n"
                "  lists the anchor as a REQUIRED mitigation: with only an affinity\n"
                "  loss the model learns to move probability mass toward each pair's\n"
                "  energy minimum and converges on an over-attractive, geometrically\n"
                "  impossible solution. A physics claim without it is not supported.")
        tr, va, te = tr & keep, va & keep, te & keep
        print(f"  geometry stage: {tr.sum():,}/{va.sum():,}/{te.sum():,} structural rows")

    need_plm = str(cfg.plm.mode) != "none"
    store = None
    if need_plm:
        ptr = paths.WORK / f"plm_{cfg.plm.mode}.path"
        if not ptr.exists():
            raise SystemExit(
                f"no embedding cache for plm.mode={cfg.plm.mode}.\n"
                f"    sbatch --array=0-11 slurm/40_embed_protein.sbatch\n"
                f"    python scripts/phase4_embed.py --merge -o plm.mode={cfg.plm.mode}\n"
                "  Or train without a PLM at all: -o plm=none, which Section 6.7 asks\n"
                "  you to treat as a serious candidate rather than a control.")
        store = ProteinStore(Path(ptr.read_text().strip()))
        print(f"  PLM cache {store.root.name}  dim {store.dim}")

    max_res = int(cfg.data.get("max_sequence_length", 1022))
    max_atoms = int(cfg.data.get("max_atoms", 64))
    mk = lambda m, sh: make_loader(  # noqa: E731
        PairDataset(pairs[m], prot, store, max_res, max_atoms,
                    {i: g for i, g in ((j, geometry.get(k))
                                       for j, k in enumerate(np.where(m)[0]))
                     if g is not None},
                    need_plm=need_plm),
        int(cfg.train.batch_size), sh, int(cfg.compute.get("workers", 8)),
        max_res, max_atoms, int(cfg.seed))
    train_loader, val_loader, test_loader = mk(tr, True), mk(va, False), mk(te, False)

    ## ---- model ---------------------------------------------------------------
    model = PFPL(cfg, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
                 pocket_extra_dim=POCKET_EXTRA_DIM)
    npar = model.n_params()
    print(f"  variant {model.variant}  device {preflight.pick_device()}  "
          f"params {npar['total']:,}")
    print(f"    {json.dumps(npar)}")

    ## ---- train ---------------------------------------------------------------
    t0 = time.time()
    trainer = Trainer(model, cfg, out, stage=stage)
    fit = trainer.fit(train_loader, val_loader,
                      select_on=str(cfg.train.get("select_on", "val_pearson")),
                      allow_unanchored=a.allow_unanchored)
    elapsed = time.time() - t0

    ## ---- score ---------------------------------------------------------------
    import torch
    best = out / "best.pt"
    if best.exists():
        trainer.model.load_state_dict(
            torch.load(best, map_location=trainer.device, weights_only=False)["model"])
    reports = {}
    preds = trainer.predict(test_loader)
    if len(preds):
        preds.to_csv(out / "predictions.csv", index=False)
        train_mean = float(pairs.pK.to_numpy()[tr].mean())
        reports["model"] = M.evaluate(preds, name, logvar="logvar",
                                      train_mean=train_mean,
                                      min_n=int(cfg.eval.min_target_n))
        te_df = preds[["protein_idx", "y"]].copy()
        reports |= M.blind_baselines(pd.DataFrame({"y": pairs.pK.to_numpy()[tr]}),
                                     te_df, y="y")
        M.print_table(reports, f"{name} - test fold of split {cfg.data.split}")

    decomposition = trainer.decompose(test_loader)
    if decomposition:
        print("  per-term energies (test): "
              + json.dumps({k: round(v, 3) for k, v in decomposition.items()
                            if not k.startswith("drift/")}))

    prereg, why = ledger.check_preregistered(cfg)
    (out / "run.json").write_text(json.dumps({
        "name": name, "stage": stage, "variant": model.variant,
        "plm_mode": str(cfg.plm.mode), "split": str(cfg.data.split),
        "seed": int(cfg.seed), "config_hash": cfg.hash,
        "terms": list(cfg.get_path("model.energy.terms", []) or []),
        "vdw_form": cfg.get_path("model.energy.vdw_form"),
        "residual": cfg.get_path("model.energy.residual"),
        "n_params": npar, "fit": fit, "elapsed_s": round(elapsed, 1),
        "preregistered": prereg, "preregistration_note": why,
        "allow_unanchored": bool(a.allow_unanchored),
        "decomposition": decomposition,
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "geometry_rows": len(geometry),
    }, indent=1, default=float))
    if reports:
        M.write(reports, out / "reports.json")

    ledger.record(cfg, stage=stage, elapsed_s=elapsed, run_name=name,
                  preregistered=str(prereg), n_params=npar["total"],
                  select_metric=str(cfg.train.get("select_on", "val_pearson")),
                  select_value=fit.get("best"),
                  test_pearson=reports["model"].pearson if reports else "",
                  test_rho=reports["model"].per_target_spearman_median if reports else "",
                  notes=why)
    print(f"\n  artifacts in {out}")
    print(f"  ledger: {json.dumps(ledger.summary())}")
    if not prereg:
        print(f"\n  *** EXPLORATORY RUN: {why}")
        print("      Section 7.4: a headline claim resting on an exploratory run must")
        print("      say so.")


if __name__ == "__main__":
    main()
