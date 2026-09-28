"""Phase 6 baselines. These must exist before any physics claim is made.

PLAN.md Section 0.3.1 and Gate G1: "Build honest splits and all baselines first.
Every later claim is relative to them. The physics work may not be judged against
numbers that do not exist yet."

This is not procedural caution. On the previous corpus a ligand-only model - 20
RDKit descriptors, no protein input of any kind - reached median per-target
Spearman +0.325 on 613 held-out protein clusters, while the best model with 7,680
ESM dimensions reached 0.158. Every protein-aware model lost to a protein-blind
control by a factor of two. If that baseline had been run first, several days of
architecture work would have been aimed somewhere else.

Four baselines, all on the frozen splits:

    target_mean   ligand-blind pooled ceiling (in eval/metrics.py)
    const         RMSE floor                  (in eval/metrics.py)
    ligand_only   ECFP + descriptors -> GBDT. The memorisation floor.
    protein_only  PLM mean-pooled -> GBDT. Shows how much is protein identity.

Section 6.2's "GEMS ligand-only RMSE 1.424 beat Vina on original PDBbind" is the
same phenomenon in the literature; it collapses on CleanSplit. Expect these to be
strong, and treat a small margin over them as the warning it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval import metrics as M
from ..features.ligand_phys import TIER_A
from ..preflight import require_rdkit

GBDT = dict(n_estimators=1200, max_depth=7, learning_rate=0.04, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=2.0, min_child_weight=8,
            tree_method="hist")


def ecfp_matrix(smiles, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    require_rdkit("ECFP for the ligand-only baseline")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = np.zeros((len(smiles), nbits), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[i] = np.frombuffer(gen.GetFingerprintAsNumPy(m).tobytes(), np.uint8)[:nbits]
    return out


def fit_gbdt(Xtr, ytr, Xva, yva, seed: int = 1337, params: dict | None = None,
             wtr=None):
    import xgboost as xgb
    p = dict(GBDT, **(params or {}))
    m = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed,
                         early_stopping_rounds=60, n_jobs=-1, **p)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
    return m


def ligand_only(pairs: pd.DataFrame, train, val, test, seed: int = 1337,
                use_ecfp: bool = True, phys: pd.DataFrame | None = None) -> pd.DataFrame:
    """The memorisation floor. No protein information of any kind enters."""
    parts = []
    if use_ecfp:
        parts.append(ecfp_matrix(pairs.ligand_smiles_std.tolist()).astype(np.float32))
    if phys is not None:
        cols = [c for c in TIER_A if c in phys.columns]
        parts.append(np.nan_to_num(phys[cols].to_numpy(np.float32)))
    if not parts:
        raise SystemExit("ligand_only needs ECFP or ligand-physics features")
    X = np.hstack(parts)
    print(f"  ligand_only X {X.shape}"
          + ("  (ECFP + descriptors)" if len(parts) == 2 else ""))
    y = pairs.pK.to_numpy(np.float32)
    m = fit_gbdt(X[train], y[train], X[val], y[val], seed,
                 wtr=pairs.weight.to_numpy()[train] if "weight" in pairs else None)
    return pd.DataFrame(dict(pair_idx=np.where(test)[0],
                             protein_idx=pairs.protein_idx.to_numpy()[test],
                             y=y[test], pred=m.predict(X[test])))


def protein_only(pairs: pd.DataFrame, prot_features: np.ndarray, train, val, test,
                 seed: int = 1337) -> pd.DataFrame:
    """No ligand information. Its per-target rho is undefined by construction -
    the prediction is constant within a target - so it is scored pooled only."""
    X = prot_features[pairs.protein_idx.to_numpy()]
    y = pairs.pK.to_numpy(np.float32)
    m = fit_gbdt(X[train], y[train], X[val], y[val], seed)
    return pd.DataFrame(dict(pair_idx=np.where(test)[0],
                             protein_idx=pairs.protein_idx.to_numpy()[test],
                             y=y[test], pred=m.predict(X[test])))


def fusion_gbdt(pairs: pd.DataFrame, prot_features: np.ndarray, train, val, test,
                seed: int = 1337, phys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Concatenated protein + ligand features. The cheap no-physics reference.

    NOT the Section 9 control - that is Variant C in `models/affinity.py`, which is
    parameter-matched to the physics model. This one is here because it is fast and
    because "a GBDT on concatenated embeddings" is the honest description of the
    previous Savant model, so it gives the sprint a directly comparable before row.
    """
    parts = [prot_features[pairs.protein_idx.to_numpy()],
             ecfp_matrix(pairs.ligand_smiles_std.tolist()).astype(np.float32)]
    if phys is not None:
        cols = [c for c in TIER_A if c in phys.columns]
        parts.append(np.nan_to_num(phys[cols].to_numpy(np.float32)))
    X = np.hstack(parts)
    y = pairs.pK.to_numpy(np.float32)
    m = fit_gbdt(X[train], y[train], X[val], y[val], seed,
                 wtr=pairs.weight.to_numpy()[train] if "weight" in pairs else None)
    return pd.DataFrame(dict(pair_idx=np.where(test)[0],
                             protein_idx=pairs.protein_idx.to_numpy()[test],
                             y=y[test], pred=m.predict(X[test])))


def run_all(pairs: pd.DataFrame, prot_features: np.ndarray | None, train, val, test,
            phys: pd.DataFrame | None = None, seed: int = 1337) -> dict:
    """Every Phase 6 baseline on one split, returned as {name: (Report, predictions)}."""
    tr = pairs[train][["pK"]].rename(columns={"pK": "y"})
    te = pairs[test][["pK", "protein_idx"]].rename(columns={"pK": "y"})
    out: dict[str, tuple] = {}
    for k, r in M.blind_baselines(tr, te, y="y").items():
        out[k] = (r, None)

    print("\n-- ligand_only ------------------------------------------------")
    p = ligand_only(pairs, train, val, test, seed, phys=phys)
    out["ligand_only"] = (M.evaluate(p, "baseline: ligand only (no protein)"), p)

    if prot_features is not None:
        print("\n-- protein_only -----------------------------------------------")
        p = protein_only(pairs, prot_features, train, val, test, seed)
        out["protein_only"] = (M.evaluate(p, "baseline: protein only (no ligand)",
                                          pooled_only=True,
                                          note="constant within a target; pooled only"), p)
        print("\n-- fusion_gbdt ------------------------------------------------")
        p = fusion_gbdt(pairs, prot_features, train, val, test, seed, phys)
        out["fusion_gbdt"] = (M.evaluate(p, "baseline: GBDT fusion (v1-equivalent)"), p)
    return out
