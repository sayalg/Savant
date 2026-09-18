"""Learning curve for the docked (structure-based) model on combined_50k_filtered.csv.

    python learning_curve_structure_based.py --out work_curve

Answers two questions the report needs and nothing else currently measures:

  1. **Is the structure-based model still improving at 47k pairs?** If the curve
     is still rising at the largest training size, folding more of input.csv is
     expected to help, and the slope says roughly how much. If it has flattened,
     it has not, and the report should say so instead.

  2. **What is a docked structure worth, at every training size?** Each size is
     trained twice - once with the seven HADDOCK3 columns and once without -
     on identical rows. The gap between the two curves is the value of folding,
     measured rather than assumed.

Those are different claims and both are needed: a rising structure-based curve
alone would not show that *folding* is what pays, only that more data does.

The test set is fixed across every size and every repeat: one protein-grouped
split is taken first, and the training fractions are drawn from the training
portion only. Re-splitting per size would confound the curve with test-set noise.

Repeats use different subsamples of the same training pool, so the error bars
describe subsample variability, not label noise.

Cost: the embedding stage dominates and is cached to --out. ESM2-8M and
ChemBERTa-77M are both small enough to run on CPU in well under an hour;
a GPU makes it minutes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"
ESM = "facebook/esm2_t6_8M_UR50D"          # 320-dim, matches the structure-free tier
AA = "ACDEFGHIKLMNPQRSTVWY"

HADDOCK_COLS = ["score", "total", "vdw", "elec", "desolv", "bsa",
                "num_interacting_residues"]
## iptm is excluded on purpose: it comes from the AlphaFold3 runs and would not
## be available for a structure a user supplies.


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def encode(texts, model_name, batch_size, max_length=None):
    """Mean-pooled last hidden state, padding masked out."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"{model_name} on {device}, {len(texts)} items")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    order = np.argsort([len(t) for t in texts])       # batch like with like
    out = None
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            sel = order[start:start + batch_size]
            batch = tok([texts[i] for i in sel], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            hidden = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), pooled.shape[1]), dtype=np.float32)
            out[sel] = pooled
    return out


def ligand_blocks(smiles):
    """(11 AddHs descriptors, full descriptor frame, ECFP4 counts)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import (Crippen, Descriptors, Lipinski,
                            rdFingerprintGenerator, rdMolDescriptors)
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    d11, full, fps = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        mh = Chem.AddHs(m)
        d11.append([Descriptors.MolWt(mh), Crippen.MolLogP(mh),
                    Lipinski.NumHDonors(mh), Lipinski.NumHAcceptors(mh),
                    Lipinski.NumRotatableBonds(mh), rdMolDescriptors.CalcTPSA(mh),
                    rdMolDescriptors.CalcNumRings(mh),
                    rdMolDescriptors.CalcNumAromaticRings(mh), mh.GetNumAtoms(),
                    mh.GetNumHeavyAtoms(), Chem.GetFormalCharge(mh)])
        full.append(Descriptors.CalcMolDescriptors(m))
        fps.append(gen.GetCountFingerprintAsNumPy(m))

    ## cast first, then clean: Ipc and Kappa2 only overflow to inf on the cast,
    ## so cleaning beforehand leaves infinities behind
    fd = pd.DataFrame(full).astype(np.float32)
    fd = fd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fd = fd.loc[:, fd.max() != fd.min()]                  # drop constant columns
    return (np.asarray(d11, np.float32), fd.to_numpy(np.float32),
            np.asarray(fps, np.float32), list(fd.columns))


def aa_composition(seqs):
    return np.array([[s.count(a) / len(s) for a in AA] + [len(s)] for s in seqs],
                    dtype=np.float32)


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "features.npz"
    if cache.exists():
        log(f"reusing {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X_free"], z["X_dock"], z["y"], z["groups"]

    df = pd.read_csv(args.csv)
    n0 = len(df)
    df = df[df.neg_log10_affinity_M >= args.min_pkd]
    ## vdw > 1000 is a clashing pose, not an energy; training dropped these
    df = df[~(df["vdw"] > 1000)]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    df = df[[Chem.MolFromSmiles(s) is not None for s in df.smiles]]
    df = df.reset_index(drop=True)
    log(f"rows {n0} -> {len(df)} after pKd >= {args.min_pkd}, failed dockings, "
        f"unparseable SMILES")

    ligands = pd.Index(df.smiles.unique())
    proteins = pd.Index(df.seq.unique())
    log(f"{len(ligands)} unique ligands, {len(proteins)} unique proteins")

    d11_u, full_u, fp_u, cols = ligand_blocks(ligands.tolist())
    cb_u = encode(ligands.tolist(), CHEMBERTA, args.ligand_batch, 256)
    esm_u = encode(proteins.tolist(), ESM, args.protein_batch, 1024)
    aa_u = aa_composition(proteins.tolist())

    li = ligands.get_indexer(df.smiles)
    pi = proteins.get_indexer(df.seq)
    ## block order mirrors the structure-free tier in savant_structure_based.py
    X_free = np.hstack([cb_u[li], esm_u[pi], d11_u[li], full_u[li],
                        aa_u[pi], fp_u[li]]).astype(np.float32)
    X_dock = np.hstack([X_free, df[HADDOCK_COLS].to_numpy(np.float32)])
    y = df.neg_log10_affinity_M.to_numpy(np.float32)
    groups = pi

    np.savez_compressed(cache, X_free=X_free, X_dock=X_dock, y=y, groups=groups)
    json.dump({"rdkit_columns": cols, "n_rows": int(len(df)),
               "free_width": int(X_free.shape[1]),
               "dock_width": int(X_dock.shape[1])},
              open(out / "feature_spec.json", "w"), indent=1)
    log(f"free {X_free.shape}  dock {X_dock.shape}  cached to {cache}")
    return X_free, X_dock, y, groups


# --------------------------------------------------------------------------- #
# the curve
# --------------------------------------------------------------------------- #

def run(args):
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit

    X_free, X_dock, y, groups = build(args)
    out = Path(args.out)

    ## ONE protein-grouped split; the test set never changes so the curve
    ## measures training size and nothing else
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=args.seed)
                  .split(X_free, y, groups))
    log(f"fixed test set: {len(te)} rows, {len(set(groups[te]))} proteins; "
        f"training pool {len(tr)} rows, {len(set(groups[tr]))} proteins")

    fracs = [float(f) for f in args.fractions.split(",")]
    rows = []
    for frac in fracs:
        n = int(round(frac * len(tr)))
        for rep in range(args.repeats):
            rng = np.random.default_rng(args.seed + rep)
            sub = rng.choice(tr, size=n, replace=False)
            ## inner split for early stopping, grouped on the same key
            gi, vi = next(GroupShuffleSplit(1, test_size=0.2,
                                            random_state=args.seed + rep)
                          .split(sub, y[sub], groups=groups[sub]))
            fit, val = sub[gi], sub[vi]

            for name, X in (("structure-free", X_free), ("structure-based", X_dock)):
                params = {"objective": "reg:squarederror", "eval_metric": "rmse",
                          "max_depth": 7, "eta": 0.05, "subsample": 0.8,
                          "colsample_bytree": 0.5, "min_child_weight": 5,
                          "tree_method": "hist", "nthread": args.threads,
                          "seed": args.seed + rep}
                ## the eval matrix must reference the training one, or XGBoost
                ## refuses: its bin edges have to come from the training data
                dfit = xgb.QuantileDMatrix(X[fit], label=y[fit])
                dval = xgb.QuantileDMatrix(X[val], label=y[val], ref=dfit)
                b = xgb.train(params, dfit, num_boost_round=args.rounds,
                              evals=[(dval, "val")],
                              early_stopping_rounds=40, verbose_eval=False)
                p = b.inplace_predict(X[te])
                rows.append({
                    "features": name, "frac": frac, "n_train": len(fit),
                    "n_proteins": len(set(groups[fit])), "repeat": rep,
                    "best_iter": int(b.best_iteration),
                    "R2": float(r2_score(y[te], p)),
                    "Pearson": float(np.corrcoef(y[te], p)[0, 1]),
                    "Spearman": float(spearmanr(y[te], p).correlation),
                    "RMSE": float(np.sqrt(mean_squared_error(y[te], p))),
                    "pred_std_ratio": float(p.std() / y[te].std()),
                })
                log(f"  {name:16s} frac {frac:.2f} n={len(fit):6d} rep {rep} "
                    f"R2 {rows[-1]['R2']:.4f} Pearson {rows[-1]['Pearson']:.4f}")
            pd.DataFrame(rows).to_csv(out / "learning_curve.csv", index=False)

    lc = pd.DataFrame(rows)
    ## aggregate on frac, not n_train: the inner grouped split lands a few rows
    ## differently per repeat, so n_train varies slightly and would fragment
    ## every group into singletons
    summary = (lc.groupby(["features", "frac"])
                 .agg(n_train=("n_train", "mean"), n_proteins=("n_proteins", "mean"),
                      R2=("R2", "mean"), R2_sd=("R2", "std"),
                      Pearson=("Pearson", "mean"), RMSE=("RMSE", "mean"))
                 .round(4))
    print("\n" + summary.to_string())
    summary.to_csv(out / "learning_curve_summary.csv")

    ## the two claims the report makes, stated as numbers
    sb = summary.loc["structure-based"]
    if len(sb) >= 2:
        (f1, r1), (f2, r2) = list(sb.R2.items())[-2:]
        n1, n2 = sb.n_train[f1], sb.n_train[f2]
        print(f"\nlast segment of the structure-based curve: "
              f"R2 {r1:.4f} -> {r2:.4f} as n_train {n1:.0f} -> {n2:.0f}")
        if n2 > n1:
            print(f"  slope per doubling of training data: "
                  f"{(r2 - r1) / np.log2(n2 / n1):+.4f} R2")
        print("  a clearly positive slope is the evidence that folding more of "
              "input.csv would help; a flat one is evidence it would not")

    gap = summary.R2.unstack(0)
    if {"structure-based", "structure-free"} <= set(gap.columns):
        d = (gap["structure-based"] - gap["structure-free"]).round(4)
        print("\nvalue of the HADDOCK3 block (R2, structure-based minus free), by frac:")
        print(d.to_string())
        print(f"  mean over all sizes: {d.mean():+.4f} R2")

    lc.to_csv(out / "learning_curve.csv", index=False)
    log(f"wrote {out / 'learning_curve.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", default=str(HERE / "combined_50k_filtered.csv"))
    p.add_argument("--out", default=str(HERE / "work_curve"))
    p.add_argument("--fractions", default="0.1,0.2,0.3,0.4,0.6,0.8,1.0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-pkd", type=float, default=4.0)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--ligand-batch", type=int, default=256)
    p.add_argument("--protein-batch", type=int, default=32)
    p.add_argument("--threads", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
