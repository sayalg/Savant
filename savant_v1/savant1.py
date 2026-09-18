"""Savant v1 - binding-affinity prediction from a protein sequence and a ligand SMILES.

Two models, both trained by ML_model/train_savant_full.ipynb on the same ~1.8M
gated pairs. They differ **only** in the protein embedding:

    esm650   2935 features   ESM2-650M (1280-dim)   facebook/esm2_t33_650M_UR50D
    esm8     1975 features   ESM2-8M   (320-dim)    facebook/esm2_t6_8M_UR50D

Everything else - rows, benchmark gate, split, ligand features, hyperparameters -
is identical between them, so a difference in their scores is attributable to the
protein representation and nothing else.

Both are XGBoost. Both ship as `save_model()` JSON inside a joblib bundle, which
carries its own column list and intercept and loads across XGBoost versions.
There is no feature spec to build and no scikit-learn version requirement.

WHAT CHANGED FROM V3
--------------------
v3 carried three models, two of which needed a docking run and a
`feature_spec.json` rebuilt on the training machine. v4 keeps only the two
structure-free full models, so:

  * no HADDOCK columns, no --docking, no --protein-chain / --ligand-chain
  * no feature_spec.json, no --build-spec, no --split
  * a PDB is accepted only as a convenient way to read out a sequence

FEATURE LAYOUT - ORDER IS LOAD-BEARING
--------------------------------------
The width check below cannot catch a wrong order, because two orders can share a
width. This order must match how the training matrix was built:

    ChemBERTa ligand embedding   384
    RDKit named descriptors      226   (columns from the bundle)
    ECFP4 count fingerprint     1024
    ESM2 protein embedding      1280 | 320
    AA composition + length       21

    python savant1.py --seq MTVPDRSEI... --smiles "CCO"
    python savant1.py --model esm8 --csv pairs.csv --out scored.csv
    python savant1.py --pdb complex.pdb --smiles "CCO"
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ML = HERE.parent / "ML_model"

## Newest generation first. work_full_v3 is the averaged-label, regressor-only
## build; work_full_v2 is the earlier raw-label build that also shipped a
## classifier. Whichever exists first wins, and --models overrides either.
MODEL_DIRS = [ML / "work_full_v3", ML / "work_full_v2"]
MODELS_DIR = next((d for d in MODEL_DIRS if d.exists()), MODEL_DIRS[0])

CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"
BINDER_PKD = 6.0
AA = "ACDEFGHIKLMNPQRSTVWY"

MODELS = {
    "esm650": {"bundle": "savant_full_esm650.pkl", "dim": 1280},
    "esm8":   {"bundle": "savant_full_esm8.pkl",   "dim": 320},
}

H11_COLS = ["h_molecular_weight", "h_logP", "h_num_h_donors", "h_num_h_acceptors",
            "h_num_rotatable_bonds", "h_tpsa", "h_num_rings", "h_num_aromatic_rings",
            "h_num_atoms", "h_num_heavy_atoms", "h_formal_charge"]

AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
       "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
       "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
       "TRP": "W", "TYR": "Y"}


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _encoder(name, half=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
    if half and device == "cuda":
        mdl = mdl.half()          # training encoded in fp16 on GPU
    return tok, mdl, device


def _encode(texts, name, batch_size=16, half=False, max_length=None):
    """Mean-pool the last hidden state, ignoring padding - as in training."""
    import torch
    tok, mdl, device = _encoder(name, half)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            hidden = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# ligand and protein features
# --------------------------------------------------------------------------- #

def _mols(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"RDKit could not parse SMILES: {s!r}")
        out.append(m)
    return out


def _descriptors11(mols):
    """The 11 AddHs variants. Training computed these on the hydrogen-added
    molecule; two of them disagree with the bare-molecule values."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    rows = []
    for m in mols:
        m = Chem.AddHs(m)
        rows.append([Descriptors.MolWt(m), Crippen.MolLogP(m),
                     Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                     Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcNumRings(m),
                     rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumAtoms(),
                     m.GetNumHeavyAtoms(), Chem.GetFormalCharge(m)])
    return np.asarray(rows, dtype=np.float32)


def descriptors_named(mols, columns):
    """(n, len(columns)) by name, in the order training used.

    Cast to float32 first, then clean: some descriptors (Ipc, Kappa2) only
    overflow to inf on the cast, so cleaning before it leaves infs behind.
    """
    from rdkit.Chem import Descriptors
    d = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols]).astype(np.float32)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.concat([d, pd.DataFrame(_descriptors11(mols), columns=H11_COLS)], axis=1)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        raise ValueError(
            f"this RDKit build does not produce {len(missing)} descriptor(s) the "
            f"model was trained on, e.g. {missing[:5]}. Score on a machine whose "
            f"RDKit matches the one that trained the bundle."
        )
    return d[columns].to_numpy(np.float32)


def ecfp4(mols, n_bits=1024):
    """(n, 1024) Morgan count fingerprint, radius 2."""
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def aa_composition(seqs):
    """(n, 21): the 20 residue fractions, then sequence length."""
    return np.array([[s.count(a) / len(s) for a in AA] + [len(s)] for s in seqs],
                    dtype=np.float32)


def read_pdb(path, protein_chain="A", ligand_chain="B"):
    """(sequence, smiles) from a structure - a convenience, not a feature source.

    The sequence is exact. The SMILES is inferred from ligand geometry, so bond
    orders are a guess (`C(=O)O` comes back as `C(O)O`) and every ligand feature
    is computed from the SMILES. Pass --smiles whenever you have it.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    seen, seq, het = set(), [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == protein_chain:
            key = line[22:27]
            if key not in seen:
                seen.add(key)
                seq.append(AA3.get(line[17:20].strip(), "X"))
        elif line.startswith("HETATM") and line[21] == ligand_chain:
            het.append(line)

    if not seq:
        raise ValueError(f"no ATOM records on chain {protein_chain!r} in {path}")
    mol = Chem.MolFromPDBBlock("\n".join(het) + "\nEND", sanitize=True) if het else None
    return "".join(seq), (Chem.MolToSmiles(mol) if mol is not None else None)


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #

def _parse_base_score(raw):
    """XGBoost reports base_score either as a scalar ('5E-1') or as a one-element
    vector ('[6.670542E0]'), depending on the build. Accept both."""
    return float(str(raw).strip().strip("[]").split(",")[0])


def _load_booster(src):
    """Load a save_model() JSON booster from bytes or a path, restoring
    base_score if the load drops it.

    The JSON carries the fitted intercept, but some XGBoost builds report 0.5
    after load and then predict about 6 pKd low. The right value is in the file,
    so this needs nothing external.
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
        print(f"base_score came back as {current:g}, restored to {saved:.6f} from the file")
    return booster


def _load_bundle(model, models_dir=None):
    """A bundle dict, from a .pkl or from a variant directory."""
    if models_dir and Path(models_dir).is_file():
        return joblib.load(models_dir)

    root = Path(models_dir) if models_dir else MODELS_DIR
    pkl = root / MODELS[model]["bundle"]
    if pkl.exists():
        return joblib.load(pkl)

    ## fall back to the raw training output: work_full_v*/<variant>/*.json
    ## the classifier is optional, so it is not in `need`
    vdir = root / model
    need = [vdir / "xgb_regressor.json", root / "rdkit_columns.json"]
    if not all(p.exists() for p in need):
        raise FileNotFoundError(
            f"no model for {model!r}. Looked for {pkl}, then for "
            f"{[str(p) for p in need]}.\n"
            f"Searched (newest first): {[str(d) for d in MODEL_DIRS]}\n"
            f"Build them with ML_model/train_savant_full_dedup.ipynb, or point at "
            f"them with --models."
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

class Savant:
    """One variant's regressor and classifier, plus the features they need."""

    def __init__(self, model="esm650", models_dir=None):
        if model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}")
        b = _load_bundle(model, models_dir)
        self.model = model
        self.bundle = b
        self.esm_model = b["esm_model"]
        self.rdkit_columns = b["rdkit_columns"]
        self.block_widths = b["block_widths"]
        self.n_features_in_ = b["n_features"]
        self.binder_pkd = b.get("binder_pkd", BINDER_PKD)
        self.regressor = _load_booster(b["regressor_json"])
        ## Optional. A regressor gives every threshold for free at inference time;
        ## a classifier gives one, fixed at training time. Bundles from
        ## train_savant_full_dedup.ipynb ship without one.
        cj = b.get("classifier_json")
        self.classifier = _load_booster(cj) if cj else None

    def features(self, seqs, smiles, ligand_batch=64, protein_batch=8):
        """Assemble the matrix in the training block order.

        ORDER IS LOAD-BEARING. The width assert below is a backstop, not a
        guarantee - two different orders can produce the same width.
        """
        mols = _mols(smiles)
        blocks = [
            _encode(list(smiles), CHEMBERTA, ligand_batch),
            descriptors_named(mols, self.rdkit_columns),
            ecfp4(mols),
            ## fp16 and a 1024-token cap, matching training
            _encode(list(seqs), self.esm_model, protein_batch, half=True, max_length=1024),
            aa_composition(seqs),
        ]
        widths = [b.shape[1] for b in blocks]
        if widths != list(self.block_widths):
            raise ValueError(
                f"block widths {widths} do not match the trained layout "
                f"{list(self.block_widths)}; the RDKit column list or an encoder "
                f"does not match the bundle."
            )
        X = np.concatenate(blocks, axis=1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"built {X.shape[1]} features, model expects {self.n_features_in_}")
        return X

    def predict_from(self, X):
        """Score an already-built matrix.

        The three columns are always present. With no classifier in the bundle,
        `p_binder` is NaN and `is_binder` comes from thresholding the predicted
        pKd at `binder_pkd` - so downstream code that selects on column presence
        keeps working, and an all-NaN probability column is correctly ignored
        rather than silently treated as a real prediction.
        """
        pk = self.regressor.inplace_predict(X)
        if self.classifier is None:
            return pd.DataFrame({"p_binder": np.full(len(pk), np.nan),
                                 "is_binder": pk > self.binder_pkd, "pKd": pk})
        p = self.classifier.inplace_predict(X)
        return pd.DataFrame({"p_binder": p, "is_binder": p >= 0.5, "pKd": pk})

    def predict(self, seqs, smiles, **kw):
        """Per pair: probability of binding, the call, and predicted pKd."""
        return self.predict_from(self.features(seqs, smiles, **kw))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODELS), default="esm650")
    ap.add_argument("--seq")
    ap.add_argument("--smiles", help="strongly preferred over reading it from --pdb")
    ap.add_argument("--pdb", help="read the sequence (and, as a fallback, the ligand) from a structure")
    ap.add_argument("--protein-chain", default="A")
    ap.add_argument("--ligand-chain", default="B")
    ap.add_argument("--csv", help="batch: columns seq + smiles, or pdb + smiles")
    ap.add_argument("--out")
    ap.add_argument("--models", help="override the model directory, or a bundle .pkl")
    ap.add_argument("--ligand-batch", type=int, default=64)
    ap.add_argument("--protein-batch", type=int, default=8)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if "pdb" in df.columns:
            parsed = [read_pdb(p, args.protein_chain, args.ligand_chain) for p in df["pdb"]]
            df["seq"] = [p[0] for p in parsed]
            if "smiles" not in df.columns:
                df["smiles"] = [p[1] for p in parsed]
            else:
                df["smiles"] = df["smiles"].fillna(pd.Series([p[1] for p in parsed]))
    elif args.pdb or (args.seq and args.smiles):
        row = {"seq": args.seq, "smiles": args.smiles}
        if args.pdb:
            seq, smiles = read_pdb(args.pdb, args.protein_chain, args.ligand_chain)
            row["seq"] = row["seq"] or seq
            row["smiles"] = row["smiles"] or smiles
            if not args.smiles:
                print("note: ligand SMILES inferred from geometry; bond orders are a "
                      "guess and every ligand feature depends on them - pass --smiles")
        df = pd.DataFrame([row])
    else:
        ap.error("need --csv, or --pdb, or --seq with --smiles")

    missing = [c for c in ("seq", "smiles") if df[c].isna().any()]
    if missing:
        ap.error(f"rows with no {missing}")

    sv = Savant(model=args.model, models_dir=args.models)
    scored = pd.concat(
        [df.reset_index(drop=True),
         sv.predict(df["seq"].tolist(), df["smiles"].tolist(),
                    ligand_batch=args.ligand_batch, protein_batch=args.protein_batch)],
        axis=1)

    print(f"model: {sv.model} ({sv.n_features_in_} features, {sv.esm_model}), "
          f"{len(scored)} pair(s)")
    if sv.classifier is None:
        print(f"no classifier in this bundle: p_binder is NaN, is_binder is "
              f"pKd > {sv.binder_pkd}")
    else:
        print(f"binder threshold: p_binder >= 0.5 (trained at pKd > {sv.binder_pkd})")
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(scored[["p_binder", "is_binder", "pKd"]].to_string(index=False))


if __name__ == "__main__":
    main()
