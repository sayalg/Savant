"""Savant, structure-based - the earlier generation, kept for the benchmark.

Superseded for everyday use by the ../savant/ package, which is structure-free,
runs on CPU and needs no docking. This file is retained because it is the only
generation that consumes docked-complex energetics: it is the "structure-based"
arm of the paper's feature-ablation table and of the benchmark comparison, and it
supplies the haddock_* columns of the scored benchmark.

**It runs HADDOCK3 itself.** Earlier revisions imported a separate `savant`
package for that; the code is now inlined here, so this file plus its model
weights are the whole dependency. `haddock3` must be on PATH.

Three models, selected with --model:

    structure-free    1972   seq + SMILES, no docking          (models_structure_free)
    structure-based   1979   structure-free + 7 HADDOCK3 cols  (models_structure_free)
    full              2935   the 1.8M-pair model, ESM2-650M    (models_full)

`structure-free` and `structure-based` are the two tiers from
train_structure_free_model.ipynb (both keep ECFP4), and they are what this file
exists for. They come in three splits (--split): protein_grouped (default),
ligand_grouped, both_disjoint.

`full` loads the same bundle that the ../savant/ package serves and has no split
variants. Prefer that package for it: it is the paper's model of record and
carries no docking code you are not using.

Feature layout - ORDER IS LOAD-BEARING and NOT caught by the width check, since
two different orders can share a width. It must match how each training matrix
was built.

    structure-free / -based (per train_structure_free_model.ipynb, cell 25):
        ChemBERTa ligand embedding   384
        ESM2-8M protein embedding    320
        RDKit descriptors (AddHs)     11
        RDKit full descriptor set    212   (columns from feature_spec.json)
        AA composition + length       21
        ECFP4 count fingerprint     1024
        HADDOCK3 block                 7   (structure-based only)

    full (per train_savant_full_dedup.ipynb):
        ChemBERTa ligand embedding   384
        RDKit named descriptor set   226   (columns from rdkit_columns.json)
        ECFP4 count fingerprint     1024
        ESM2-650M protein embedding 1280
        AA composition + length       21

Note AA comes BEFORE ECFP in the structure-free/-based tiers.

Before scoring with structure-free / structure-based, build the feature spec ON
THE MACHINE THAT TRAINED THE MODELS (recovers the kept RDKit column list, which
differs by RDKit version, and the XGBoost intercept the pickles can lose):

    python savant_structure_based.py --build-spec

`full` needs no spec: its boosters are stored as save_model() JSON, which keeps
its own column list and intercept.

    # structure-based: docks the complex itself, no --docking needed
    python savant_structure_based.py --model structure-based --pdb complex.pdb \
        --protein-chain A --ligand-chain B --smiles "CCO"

    # reuse terms you already have, skipping the docking run
    python savant_structure_based.py --model structure-based --pdb complex.pdb --smiles "CCO" \
        --docking "score=-89.9,total=-218,vdw=-48.4,elec=-169.6,desolv=-7.6,bsa=830.6"

    python savant_structure_based.py --model structure-free --seq MTVPDRSEI... --smiles "CCO"
    python savant_structure_based.py --csv complexes.csv --dock --out scored.csv

Docking costs minutes per complex. Worth knowing before spending it: the six
HADDOCK3 energy terms were worth about +0.05 R2 internally but only +0.012
Pearson on the 500-pair benchmark - inside the noise. That is why ../savant/
drops them.
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
ML = HERE.parents[0] / "ML_model"
MODELS_SF = ML / "models_structure_free"     # structure-free / structure-based
MODELS_FULL = ML / "models_full"             # the 1.8M-pair 'full' model
TRAIN_CSV = ML / "combined_50k_filtered.csv"
SPEC = HERE / "feature_spec.json"

ESM = "Synthyra/ESM2-8M"                      # structure-free/-based, 320-dim
ESM_650M = "facebook/esm2_t33_650M_UR50D"     # full, 1280-dim
CHEMBERTA = "DeepChem/ChemBERTa-77M-MLM"

H11_COLS = ["h_molecular_weight", "h_logP", "h_num_h_donors", "h_num_h_acceptors",
            "h_num_rotatable_bonds", "h_tpsa", "h_num_rings", "h_num_aromatic_rings",
            "h_num_atoms", "h_num_heavy_atoms", "h_formal_charge"]
BINDER_PKD = 6.0
CONTACT_CUTOFF = 6.0          # A; reproduces num_interacting_residues on dock_1
AA = "ACDEFGHIKLMNPQRSTVWY"

HADDOCK_COLS = ["score", "total", "vdw", "elec", "desolv", "bsa",
                "num_interacting_residues"]

AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
       "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
       "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
       "TRP": "W", "TYR": "Y"}


# --------------------------------------------------------------------------- #
# model registry - the only thing that changes between the three models
# --------------------------------------------------------------------------- #

MODELS = {
    "structure-free": {
        "dir": MODELS_SF,
        "blocks": ("emb", "desc11", "desc_full", "aa", "ecfp"),
        "native": False, "needs_haddock": False, "split": True,
    },
    "structure-based": {
        "dir": MODELS_SF,
        "blocks": ("emb", "desc11", "desc_full", "aa", "ecfp", "haddock"),
        "native": False, "needs_haddock": True, "split": True,
    },
    "full": {
        "dir": MODELS_FULL,
        "blocks": ("chemberta", "rdkit_named", "ecfp", "esm650", "aa"),
        "native": True, "needs_haddock": False, "split": False,
    },
}


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
        mdl = mdl.half()          # train_savant_full_dedup.ipynb encoded in fp16
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


def embeddings(seqs, smiles):
    """(n, 704): ChemBERTa ligand block first, then ESM2-8M protein."""
    return np.concatenate([_encode(list(smiles), CHEMBERTA),
                           _encode(list(seqs), ESM)], axis=1)


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


def descriptors11(mols):
    """(n, 11) on the hydrogen-added molecule, matching the training columns."""
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


def _descriptor_frame(mols):
    """The full RDKit set, cleaned. Cast first: some descriptors overflow to inf
    only on the cast to float32, so cleaning before it leaves infs behind."""
    from rdkit.Chem import Descriptors
    d = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols]).astype(np.float32)
    return d.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def descriptors_full(mols, columns):
    """(n, len(columns)) selected by name, in the order training used."""
    d = _descriptor_frame(mols)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        raise ValueError(
            f"this RDKit build does not produce {len(missing)} descriptor(s) the "
            f"models were trained on, e.g. {missing[:5]}. Rebuild the spec on the "
            f"machine that trained them: python savant_structure_based.py --build-spec"
        )
    return d[columns].to_numpy(np.float32)


def descriptors_named(mols, columns):
    """(n, len(columns)) selected by name, covering CalcMolDescriptors output and
    the h_* AddHs variants, so one saved column list describes the full block."""
    frame = _descriptor_frame(mols)
    frame = pd.concat([frame, pd.DataFrame(descriptors11(mols), columns=H11_COLS)], axis=1)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"this RDKit build does not produce {len(missing)} descriptor(s) the "
            f"model was trained on, e.g. {missing[:5]}"
        )
    return frame[columns].to_numpy(np.float32)


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
    """(sequence, smiles, num_interacting_residues) from a docked complex.

    The sequence is exact. The SMILES is inferred from ligand geometry, so bond
    orders are a guess - pass the real SMILES, since every ligand feature
    downstream is computed from it.
    """
    from rdkit import Chem, RDLogger
    from scipy.spatial import cKDTree
    RDLogger.DisableLog("rdApp.*")

    seen, seq, res_key, p_xyz, l_xyz, het = set(), [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21]
        if line.startswith("ATOM  ") and chain == protein_chain:
            key = line[22:27]
            if key not in seen:
                seen.add(key)
                seq.append(AA3.get(line[17:20].strip(), "X"))
            res_key.append(key)
            p_xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith("HETATM") and chain == ligand_chain:
            het.append(line)
            l_xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    if not seq:
        raise ValueError(f"no ATOM records on chain {protein_chain!r} in {path}")
    if not het:
        raise ValueError(f"no HETATM records on chain {ligand_chain!r} in {path}")

    mol = Chem.MolFromPDBBlock("\n".join(het) + "\nEND", sanitize=True)
    smiles = Chem.MolToSmiles(mol) if mol is not None else None

    near = cKDTree(np.asarray(p_xyz)).query_ball_point(np.asarray(l_xyz), CONTACT_CUTOFF)
    hit = {k for group in near for k in (np.asarray(res_key)[group] if group else ())}
    return "".join(seq), smiles, len(hit)


# --------------------------------------------------------------------------- #
# HADDOCK3 - self-contained, no dependency on the legacy `savant` package
# --------------------------------------------------------------------------- #
#
# Ported verbatim from the haddock.py and structure.py of the retired package
# (see the `dev` branch) so this file stands alone. Note `savant/` at the repo
# root is now the production model, not that package.
# The config below is not a fresh design: it is byte-for-byte the
# `[topoaa]` + `[emref]` workflow that produced combined_50k_filtered.csv, and
# the six energy terms are read from the same emref CAPRI table. Changing either
# would silently shift the features away from what the models were trained on.

HADDOCK_ENERGY_COLS = ["score", "total", "vdw", "elec", "desolv", "bsa"]

_CFG_TEMPLATE = """\
run_dir = "{run_dir}"
mode = "local"
ncores = {ncores}

molecules = [
    "{protein_pdb}",
    "{ligand_pdb}"
]

[topoaa]
autotoppar = true

[emref]
nemsteps = {nemsteps}
"""


def haddock3_binary():
    """Resolved path to the haddock3 executable, or None.

    Resolving with which() and invoking the full path, rather than passing the
    bare name to subprocess, means the same lookup decides both "is it
    available?" and "what do we run" - and it works where a bare name would not
    be resolved (a .bat shim on Windows, for instance).
    """
    import shutil
    return shutil.which("haddock3")


def haddock3_available():
    """True if the haddock3 executable is on PATH."""
    return haddock3_binary() is not None


def split_chains(pdb_path, protein_chain, ligand_chain, workdir):
    """Write the protein and ligand chains to separate PDBs -> (protein, ligand).

    HADDOCK3 needs the two molecules as separate files, and PRODIGY-style tools
    need them on separate chains, which is why the whole pipeline insists on the
    protein/ligand chain contract.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    class _ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return 1 if chain.get_id() == self.chain_id else 0

    if protein_chain == ligand_chain:
        raise ValueError("protein-chain and ligand-chain must be different")

    pdb_path, workdir = Path(pdb_path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    structure = PDBParser(QUIET=True).get_structure("complex", str(pdb_path))
    present = {chain.get_id() for chain in next(iter(structure))}
    for label, chain_id in (("protein", protein_chain), ("ligand", ligand_chain)):
        if chain_id not in present:
            raise ValueError(
                f"{label} chain {chain_id!r} not found in {pdb_path.name}. "
                f"Available chains: {sorted(present)}")

    io = PDBIO()
    io.set_structure(structure)
    protein_pdb, ligand_pdb = workdir / "protein.pdb", workdir / "ligand.pdb"
    io.save(str(protein_pdb), _ChainSelect(protein_chain))
    io.save(str(ligand_pdb), _ChainSelect(ligand_chain))
    return protein_pdb, ligand_pdb


def _find_capri_ss(run_dir):
    """The emref CAPRI single-structure table in a HADDOCK3 run dir."""
    preferred = sorted(Path(run_dir).glob("analysis/*emref*/capri_ss.tsv"))
    if preferred:
        return preferred[-1]
    ## fall back to any capri_ss.tsv, highest-numbered module last
    any_ss = sorted(Path(run_dir).rglob("capri_ss.tsv"))
    if any_ss:
        return any_ss[-1]
    raise FileNotFoundError(
        f"No capri_ss.tsv under {run_dir}. HADDOCK3 produced no analysis output; "
        "inspect the run directory.")


def run_haddock3(protein_pdb, ligand_pdb, workdir, ncores=None, nemsteps=50):
    """Run HADDOCK3 on a split complex -> the six energy terms."""
    import os
    import shutil
    import subprocess

    binary = haddock3_binary()
    if binary is None:
        raise RuntimeError(
            "haddock3 is not on PATH. Install it (pip install haddock3) or load "
            "your cluster's HADDOCK3 module.")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = workdir / "haddock_run"
    ## HADDOCK3 refuses to start if run_dir already exists
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cfg = _CFG_TEMPLATE.format(
        run_dir=run_dir.as_posix(),
        ncores=ncores or os.cpu_count() or 4,
        protein_pdb=Path(protein_pdb).resolve().as_posix(),
        ligand_pdb=Path(ligand_pdb).resolve().as_posix(),
        nemsteps=nemsteps)
    cfg_file = workdir / "score.cfg"
    cfg_file.write_text(cfg)

    proc = subprocess.run([binary, str(cfg_file)], capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        raise RuntimeError("HADDOCK3 exited with code "
                           f"{proc.returncode}:\n" + "\n".join(tail))

    table = pd.read_csv(_find_capri_ss(run_dir), sep="\t")
    row = table[HADDOCK_ENERGY_COLS].iloc[0]
    return {col: float(row[col]) for col in HADDOCK_ENERGY_COLS}


def dock(pdb_path, protein_chain="A", ligand_chain="B", workdir=None,
         ncores=None, nemsteps=50, keep_workdir=False):
    """All 7 HADDOCK3 columns for one complex, in HADDOCK_COLS order.

    Six come from HADDOCK3's emref table; `num_interacting_residues` is a
    distance query on the pose at CONTACT_CUTOFF, which is what the training set
    used. Returns a dict keyed exactly like HADDOCK_COLS.
    """
    import shutil
    import tempfile

    tmp = workdir is None
    workdir = Path(tempfile.mkdtemp(prefix="savant_dock_")) if tmp else Path(workdir)
    try:
        protein_pdb, ligand_pdb = split_chains(pdb_path, protein_chain,
                                               ligand_chain, workdir)
        energies = run_haddock3(protein_pdb, ligand_pdb, workdir,
                                ncores=ncores, nemsteps=nemsteps)
        _, _, contacts = read_pdb(pdb_path, protein_chain, ligand_chain)
        return {**energies, "num_interacting_residues": float(contacts)}
    finally:
        if tmp and not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# feature spec (structure-free / structure-based only)
# --------------------------------------------------------------------------- #

def build_spec(csv=TRAIN_CSV, out=SPEC, split="protein_grouped"):
    """Recreate the notebook's preprocessing to recover two things the models
    need but do not carry: which RDKit descriptor columns were kept, and the
    XGBoost intercept. Run this where the models were trained."""
    from sklearn.model_selection import GroupShuffleSplit

    df = pd.read_csv(csv)[["seq", "smiles", "neg_log10_affinity_M"] + HADDOCK_COLS]
    df = df[df.neg_log10_affinity_M >= 4]
    df = df[[m is not None for m in _safe_mols(df.smiles)]].reset_index(drop=True)
    df = df[~(df["vdw"] > 1000)].reset_index(drop=True)     # failed dockings

    mols = _mols(df.smiles)
    d = _descriptor_frame(mols)
    columns = d.loc[:, d.std() > 0].columns.tolist()

    y = df.neg_log10_affinity_M.to_numpy(np.float32)
    groups = df.smiles.to_numpy() if split == "ligand_grouped" else df.seq.to_numpy()
    train, _ = next(GroupShuffleSplit(1, test_size=0.25, random_state=1337)
                    .split(np.zeros((len(y), 1)), groups=groups))
    i_fit, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=1337)
                    .split(train, y[train], groups=df.seq.to_numpy()[train]))

    spec = {"split": split,
            "n_rows": int(len(df)),
            "rdkit_descriptor_columns": columns,
            "xgb_base_score": float(y[train[i_fit]].mean())}
    Path(out).write_text(json.dumps(spec, indent=1) + "\n")
    print(f"rows {len(df)}, rdkit columns {len(columns)}, "
          f"base_score {spec['xgb_base_score']:.6f}")
    print(f"wrote {out}")
    return spec


def _safe_mols(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return [Chem.MolFromSmiles(s) for s in smiles]


def load_spec(path=SPEC):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Build it on the machine that trained the models:\n"
            f"    python savant_structure_based.py --build-spec"
        )
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #

def _parse_base_score(raw):
    """XGBoost reports base_score either as a scalar ('5E-1') or as a one-element
    vector ('[6.670542E0]'), depending on the build. Accept both."""
    return float(str(raw).strip().strip("[]").split(",")[0])


def _load_booster(src):
    """Load a save_model() JSON booster (from a path or raw bytes), restoring
    base_score if the load drops it.

    The JSON carries the fitted intercept, but some XGBoost builds report 0.5
    after load and predict ~6 pKd low. The right value is in the file, so this
    needs no external spec.
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


def _load_regressor(path, base_score):
    """Load a pickled XGBoost regressor, repairing the intercept if the pickle
    lost it.

    Loading a model into the build that trained it keeps the fitted intercept.
    Loading it elsewhere can reset base_score to the 0.5 default, which shifts
    every predicted pKd by the training mean - only then do we restore it.
    """
    model = joblib.load(path)
    booster = model.get_booster()
    current = _parse_base_score(
        json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"])

    if abs(current - 0.5) < 1e-9:
        booster.set_param({"base_score": base_score})
        print(f"{Path(path).name}: base_score was the 0.5 default, "
              f"restored to {base_score:.6f}")
    elif abs(current - base_score) > 0.01:
        print(f"{Path(path).name}: warning - model carries base_score "
              f"{current:.6f} but the spec computed {base_score:.6f}; "
              f"the spec may have been built from a different split")
    return model


def _load_full(models_dir):
    """Load the 'full' model from a work dir (JSON files) or a bundle .pkl.

    Returns (rdkit_columns, regressor_booster, classifier_booster).
    """
    d = Path(models_dir)
    if d.is_file():                      # a joblib bundle
        payload = joblib.load(d)
        return (payload["rdkit_columns"],
                _load_booster(payload["regressor_json"]),
                _load_booster(payload["classifier_json"]))
    return (json.loads((d / "rdkit_columns.json").read_text()),
            _load_booster(d / "xgb_regressor.json"),
            _load_booster(d / "xgb_classifier.json"))


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

class Savant:
    """One model's classifier and regressor, plus the features it needs."""

    def __init__(self, model="structure-free", split="protein_grouped",
                 models_dir=None, spec=None):
        if model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}")
        cfg = MODELS[model]
        self.model = model
        self.blocks = cfg["blocks"]
        self.native = cfg["native"]
        self.needs_haddock = cfg["needs_haddock"]
        models_dir = Path(models_dir) if models_dir else cfg["dir"]

        if self.native:
            ## boosters saved with save_model(): version-portable, and they keep
            ## their own base_score, so no feature_spec.json is needed
            self.rdkit_columns, self.regressor, self.classifier = _load_full(models_dir)
            self.n_features_in_ = self.regressor.num_features()
        else:
            self.spec = spec or load_spec()
            self.rdkit_columns = self.spec["rdkit_descriptor_columns"]
            stem = f"{model.replace('-', '_')}__{split}"     # structure_free__protein_grouped
            self.classifier = joblib.load(models_dir / f"clf_{stem}.pkl")
            self.regressor = _load_regressor(models_dir / f"xgb_{stem}.pkl",
                                             self.spec["xgb_base_score"])
            self.n_features_in_ = self.regressor.n_features_in_

    def features(self, seqs, smiles, docking=None):
        """Assemble the matrix for this model, in the training block order."""
        mols = _mols(smiles)
        build = {
            "emb": lambda: embeddings(seqs, smiles),
            "chemberta": lambda: _encode(list(smiles), CHEMBERTA, 64),
            ## fp16 and a 1024-token cap, matching train_savant_full_dedup.ipynb
            "esm650": lambda: _encode(list(seqs), ESM_650M, 8, half=True, max_length=1024),
            "desc11": lambda: descriptors11(mols),
            "desc_full": lambda: descriptors_full(mols, self.rdkit_columns),
            "rdkit_named": lambda: descriptors_named(mols, self.rdkit_columns),
            "aa": lambda: aa_composition(seqs),
            "ecfp": lambda: ecfp4(mols),
        }
        blocks = [build[name]() for name in self.blocks if name != "haddock"]

        if "haddock" in self.blocks:
            if docking is None:
                raise ValueError(f"{self.model!r} needs {HADDOCK_COLS}; "
                                 f"use --model structure-free without them")
            dock = np.asarray(docking, dtype=np.float32).reshape(len(seqs), -1)
            if (dock[:, HADDOCK_COLS.index("vdw")] > 1000).any():
                print("warning: vdw > 1000 indicates a failed docking run; rows "
                      "like these were dropped from training and will score poorly")
            blocks.append(dock)

        X = np.concatenate(blocks, axis=1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"built {X.shape[1]} features, model expects {self.n_features_in_}. "
                f"Either the RDKit column list came from a different build, or the "
                f"model files do not match --model {self.model!r}."
            )
        return X

    def predict_from(self, X):
        """Score an already-built matrix."""
        if self.native:
            p = self.classifier.inplace_predict(X)          # binary:logistic
            pk = self.regressor.inplace_predict(X)
        else:
            p = self.classifier.predict_proba(X)[:, 1]
            pk = self.regressor.predict(X)
        return pd.DataFrame({"p_binder": p, "is_binder": p >= 0.5, "pKd": pk})

    def predict(self, seqs, smiles, docking=None):
        """Per pair: probability of binding, the call, and predicted pKd."""
        return self.predict_from(self.features(seqs, smiles, docking))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_docking(text):
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in HADDOCK_COLS:
            raise ValueError(f"unknown docking key {key!r}; expected {HADDOCK_COLS}")
        out[key] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(MODELS),
                    help="which model to use. Default: structure-based when all 7 "
                         "HADDOCK columns are present, else structure-free. 'full' "
                         "is only used when named explicitly.")
    ap.add_argument("--build-spec", action="store_true",
                    help="recover the RDKit column list and intercept from the training csv")
    ap.add_argument("--pdb")
    ap.add_argument("--protein-chain", default="A")
    ap.add_argument("--ligand-chain", default="B")
    ap.add_argument("--seq")
    ap.add_argument("--smiles", help="overrides the ligand read from --pdb; strongly preferred")
    ap.add_argument("--docking", help='pre-computed terms, e.g. "score=-89.9,vdw=-48.4"')
    ap.add_argument("--dock", action="store_true",
                    help="run HADDOCK3 on the structure to produce the 6 energy "
                         "terms. Implied by --model structure-based with a "
                         "structure and no --docking.")
    ap.add_argument("--ncores", type=int, default=None, help="cores for HADDOCK3")
    ap.add_argument("--nemsteps", type=int, default=50,
                    help="HADDOCK3 emref minimisation steps (50 built the training set)")
    ap.add_argument("--csv", help="batch: columns seq+smiles, or pdb, plus any HADDOCK columns")
    ap.add_argument("--out")
    ap.add_argument("--models", help="override the model directory (or a full-model bundle .pkl)")
    ap.add_argument("--split", default="protein_grouped",
                    choices=["protein_grouped", "ligand_grouped", "both_disjoint"],
                    help="which trained split to load (structure-free/-based only)")
    args = ap.parse_args()

    if args.build_spec:
        build_spec(split=args.split)
        return

    pdbs = None                       # per-row structure paths, when we have them
    if args.csv:
        df = pd.read_csv(args.csv)
        if "pdb" in df.columns:
            pdbs = df["pdb"].tolist()
            parsed = [read_pdb(p, args.protein_chain, args.ligand_chain) for p in pdbs]
            df["seq"] = [p[0] for p in parsed]
            if "smiles" not in df.columns:
                df["smiles"] = [p[1] for p in parsed]
            df["smiles"] = df["smiles"].fillna(pd.Series([p[1] for p in parsed]))
            df["num_interacting_residues"] = [p[2] for p in parsed]
    elif args.pdb or (args.seq and args.smiles):
        row = {"seq": args.seq, "smiles": args.smiles}
        if args.pdb:
            pdbs = [args.pdb]
            seq, smiles, contacts = read_pdb(args.pdb, args.protein_chain, args.ligand_chain)
            row["seq"] = row["seq"] or seq
            row["smiles"] = row["smiles"] or smiles
            row["num_interacting_residues"] = contacts
            if not args.smiles:
                print("note: ligand SMILES inferred from geometry; bond orders are a "
                      "guess and every ligand feature depends on them - pass --smiles")
        row.update(_parse_docking(args.docking))
        df = pd.DataFrame([row])
    else:
        ap.error("need --csv, or --pdb, or --seq with --smiles")

    ## Run HADDOCK3 ourselves rather than making the caller supply --docking.
    ## Implied when structure-based is asked for with a structure and no
    ## pre-computed terms, since that combination has no other way to proceed.
    want_dock = args.dock or (args.model == "structure-based" and not args.docking)
    if want_dock:
        if pdbs is None:
            ap.error("--dock needs a structure: pass --pdb, or a csv with a 'pdb' column")
        if not haddock3_available():
            ap.error("haddock3 is not on PATH. Load your cluster's HADDOCK3 module, "
                     "or supply the terms yourself with --docking.")
        for i, pdb_path in enumerate(pdbs):
            print(f"[{i + 1}/{len(pdbs)}] HADDOCK3 on {Path(pdb_path).name} "
                  f"(minutes per complex)...", flush=True)
            terms = dock(pdb_path, args.protein_chain, args.ligand_chain,
                         ncores=args.ncores, nemsteps=args.nemsteps)
            for key, value in terms.items():
                df.loc[df.index[i], key] = value

    missing = [c for c in HADDOCK_COLS if c not in df.columns or df[c].isna().any()]
    model = args.model or ("structure-free" if missing else "structure-based")
    if MODELS[model]["needs_haddock"] and missing:
        ap.error(f"--model {model} needs all HADDOCK columns; missing {missing}. "
                 "Add --dock to compute them from the structure.")

    sv = Savant(model=model, split=args.split, models_dir=args.models)
    ## not `dock`: that is the module-level function that runs HADDOCK3
    dock_matrix = (df[HADDOCK_COLS].to_numpy(np.float32)
                   if MODELS[model]["needs_haddock"] else None)
    scored = pd.concat([df.reset_index(drop=True),
                        sv.predict(df["seq"].tolist(), df["smiles"].tolist(),
                                   dock_matrix)], axis=1)

    tag = sv.model if sv.native else f"{sv.model}, {args.split}"
    print(f"model: {tag} ({sv.n_features_in_} features), {len(scored)} pair(s)")
    if missing and model == "structure-free":
        print(f"      no HADDOCK columns: {missing}")
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(scored[["p_binder", "is_binder", "pKd"]].to_string(index=False))


if __name__ == "__main__":
    main()
