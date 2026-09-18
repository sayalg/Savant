"""Loading a trained bundle, and scoring with it.

Two variants ship, trained by `ML_model/train_savant_full_dedup.ipynb` on the
same ~1.8M gated pairs with the same rows, split, ligand features and
hyperparameters. They differ **only** in the protein encoder, so a difference in
their scores is attributable to the protein representation and nothing else:

    esm650   2935 features   ESM2-650M (1280-dim)   facebook/esm2_t33_650M_UR50D
    esm8     1975 features   ESM2-8M   (320-dim)    facebook/esm2_t6_8M_UR50D

`esm650` is the paper's model of record. Both are XGBoost boosters stored as
`save_model()` JSON inside a joblib bundle, which carries its own RDKit column
list and fitted intercept -- so they load across XGBoost versions and there is
nothing to build before scoring.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from savant.features import build_matrix

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_MODEL = "esm650"

MODELS = {
    "esm650": {"bundle": "savant_full_esm650.pkl", "dim": 1280},
    "esm8":   {"bundle": "savant_full_esm8.pkl",   "dim": 320},
}

## Only used to threshold a predicted pKd into a binder call, and only when the
## bundle does not record its own. A regressor gives every threshold for free at
## inference time; a classifier would fix one at training time.
BINDER_PKD = 6.0

## Searched in order; the first that exists wins. SAVANT_MODEL_DIR overrides the
## lot, which is how an installed copy finds weights outside the repo.
## work_full_v2 is an earlier raw-label build kept only on the cluster.
MODEL_DIRS = [REPO / "ML_model" / "models_full",
              REPO / "ML_model" / "work_full_v2"]


def resolve_model_dir(models_dir=None):
    """Where to look for bundles: explicit argument, then env, then the repo."""
    if models_dir:
        return Path(models_dir)
    env = os.environ.get("SAVANT_MODEL_DIR")
    if env:
        return Path(env)
    return next((d for d in MODEL_DIRS if d.exists()), MODEL_DIRS[0])


# --------------------------------------------------------------------------- #
# booster loading
# --------------------------------------------------------------------------- #

def _parse_base_score(raw):
    """XGBoost reports base_score either as a scalar ('5E-1') or as a
    one-element vector ('[6.670542E0]'), depending on the build. Accept both."""
    return float(str(raw).strip().strip("[]").split(",")[0])


def load_booster(src, quiet=False):
    """A booster from `save_model()` JSON bytes or a path, intercept intact.

    The JSON carries the fitted intercept, but some XGBoost builds report 0.5
    after loading and then predict about 6 pKd low. The right value is in the
    file, so this needs nothing external -- but it does need checking, because
    the failure is a plausible-looking number rather than an error.
    """
    import xgboost as xgb

    raw = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    saved = _parse_base_score(
        json.loads(raw)["learner"]["learner_model_param"]["base_score"])

    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    current = _parse_base_score(
        json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"])

    if abs(current - saved) > 1e-9:
        booster.set_param({"base_score": saved})
        if not quiet:
            print(f"base_score came back as {current:g}, "
                  f"restored to {saved:.6f} from the file")
    return booster


def load_bundle(model=DEFAULT_MODEL, models_dir=None):
    """A bundle dict, from a `.pkl` or from a raw training-output directory."""
    if models_dir and Path(models_dir).is_file():
        return joblib.load(models_dir)
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}, got {model!r}")

    root = resolve_model_dir(models_dir)
    pkl = root / MODELS[model]["bundle"]
    if pkl.exists():
        return joblib.load(pkl)

    ## Fall back to the raw training output, <root>/<variant>/*.json. The
    ## classifier is optional, so it is not in `need`.
    vdir = root / model
    need = [vdir / "xgb_regressor.json", root / "rdkit_columns.json"]
    if not all(p.exists() for p in need):
        raise FileNotFoundError(
            f"no {model!r} model under {root}. Looked for {pkl.name}, then for "
            f"{[p.name for p in need]}.\n"
            f"Default search path: {[str(d) for d in MODEL_DIRS]}\n"
            f"Build the bundles with ML_model/train_savant_full_dedup.ipynb, or "
            f"point at them with --models / SAVANT_MODEL_DIR."
        )

    cols = json.loads((root / "rdkit_columns.json").read_text())
    clf = vdir / "xgb_classifier.json"
    return {"variant": model,
            "esm_model": json.loads((vdir / "metrics.json").read_text())["esm_model"],
            "block_widths": [384, len(cols), 1024, MODELS[model]["dim"], 21],
            "n_features": 384 + len(cols) + 1024 + MODELS[model]["dim"] + 21,
            "rdkit_columns": cols,
            "regressor_json": (vdir / "xgb_regressor.json").read_bytes(),
            "classifier_json": clf.read_bytes() if clf.exists() else None,
            "binder_pkd": BINDER_PKD}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def pkd_to_kd_nM(pkd):
    """pKd -> Kd in nanomolar. pKd = -log10(Kd[M]), so Kd[nM] = 10^(9 - pKd)."""
    return float(10.0 ** (9.0 - float(pkd)))


class Savant:
    """One variant's booster, plus the features it needs.

    >>> sv = Savant()                                  # doctest: +SKIP
    >>> sv.predict(["MTVPDRSEI..."], ["CCO"])          # doctest: +SKIP

    The protein encoder is cached per process, so scoring many ligands against
    one already-encoded target costs only descriptor computation and tree
    inference -- which is the cost structure the method is built around.
    """

    def __init__(self, model=DEFAULT_MODEL, models_dir=None, quiet=False):
        b = load_bundle(model, models_dir)
        self.model = b.get("variant", model)
        self.bundle = b
        self.esm_model = b["esm_model"]
        self.rdkit_columns = b["rdkit_columns"]
        self.block_widths = b["block_widths"]
        self.n_features_in_ = b["n_features"]
        self.binder_pkd = b.get("binder_pkd", BINDER_PKD)
        self.regressor = load_booster(b["regressor_json"], quiet=quiet)
        ## Optional, and absent from the shipped bundles. See the README: on the
        ## benchmark a trained classifier matched the regressor at the pKd 6 cut
        ## and lost at pKd 7, so it was costing half the training budget.
        cj = b.get("classifier_json")
        self.classifier = load_booster(cj, quiet=quiet) if cj else None

    def __repr__(self):
        return (f"Savant(model={self.model!r}, features={self.n_features_in_}, "
                f"esm_model={self.esm_model!r}, "
                f"classifier={'yes' if self.classifier else 'no'})")

    def features(self, seqs, smiles, ligand_batch=64, protein_batch=8):
        """The feature matrix for parallel `seqs` / `smiles`."""
        return build_matrix(seqs, smiles,
                            rdkit_columns=self.rdkit_columns,
                            esm_model=self.esm_model,
                            block_widths=self.block_widths,
                            n_features=self.n_features_in_,
                            ligand_batch=ligand_batch,
                            protein_batch=protein_batch)

    def predict_from(self, X):
        """Score an already-built matrix.

        All three columns are always present. With no classifier in the bundle,
        `p_binder` is NaN and `is_binder` comes from thresholding the predicted
        pKd -- so downstream code that selects on column presence keeps working,
        and an all-NaN probability column is correctly ignored rather than
        silently treated as a real prediction.
        """
        pk = self.regressor.inplace_predict(X)
        if self.classifier is None:
            return pd.DataFrame({"p_binder": np.full(len(pk), np.nan),
                                 "is_binder": pk > self.binder_pkd,
                                 "pKd": pk})
        p = self.classifier.inplace_predict(X)
        return pd.DataFrame({"p_binder": p, "is_binder": p >= 0.5, "pKd": pk})

    def predict(self, seqs, smiles, **kw):
        """Per pair: probability of binding, the call, and predicted pKd."""
        return self.predict_from(self.features(seqs, smiles, **kw))

    def screen(self, sequence, smiles, **kw):
        """One target against many ligands, ranked best-first.

        The frame carries `smiles`, `pKd`, `Kd_nM` and `rank`. The protein is
        encoded once here and reused across every ligand, which is what makes
        screening a large library cheap.
        """
        smiles = list(smiles)
        scored = self.predict([sequence] * len(smiles), smiles, **kw)
        scored.insert(0, "smiles", smiles)
        scored["Kd_nM"] = [pkd_to_kd_nM(v) for v in scored["pKd"]]
        scored["rank"] = scored["pKd"].rank(ascending=False, method="min").astype(int)
        return scored.sort_values("rank", kind="stable").reset_index(drop=True)
