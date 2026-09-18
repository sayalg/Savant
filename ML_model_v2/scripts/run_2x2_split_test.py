#!/usr/bin/env python
"""The honest protein+scaffold 2x2 split, as a standalone script - not a notebook, so it
runs headless after the SLURM array in embed_structure_encoders.py finishes.

    python run_2x2_split_test.py --emb-dir work/emb --csv ../input/input.csv

Auto-discovers every `*.npy` in --emb-dir whose row count matches the current protein
index and sweeps all of them - including `esmfold2.npy` / `saprot.npy` once
`embed_structure_encoders.py --merge` has produced them, alongside whatever esm2/esmc
embeddings already exist from earlier work (point --emb-dir at
`../ML_model_v2/work/emb` or `../ML_model/work_two_part/emb`, or a directory
containing copies/symlinks of both, to compare old and new encoders in one run).

Two diagnostics per encoder, both scored on ALL FOUR cells of the split (not just the
deployment cell), because this project's own history is full of a diagnostic that looked
fine on one cell and inverted on another:

  1. Part-A ridge: can the protein's median pKd be predicted from this embedding alone?
     Cheap (one ridge per encoder, seconds), and it is the diagnostic that already caught
     the leakage-inflated "bigger encoder = better" ordering from an earlier, less honest
     split in this project - so it stays the first thing computed, before spending any
     GPU-hours on a full regressor.
  2. Part-B XGBoost: a real pKd regressor with this protein encoder plus the fixed
     ChemBERTa-77M-MLM ligand block, scored the same way `train_savant_v2.ipynb` scores
     its own regressor - so a number from this script is directly comparable to that
     notebook's table, not a different metric that merely looks similar.

Reuses the label-cleaning and split logic from `train_savant_v2.ipynb` verbatim in spirit
(same bounds, same collapse-to-median, same double-disjoint construction) - if you already
ran that notebook, pass its `work/pairs_clean.parquet` and `work/static.npz` directly with
--pairs/--static instead of rebuilding them here.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SEED = 1337
MIN_PKD, MAX_PKD = 4.0, 12.0
MAX_LABEL_SD = 1.5
## Same list train_savant_v2.ipynb uses. These are assay ceilings reported as if they were
## measurements; they must be identified identically in both places or the two protein
## indexes drift apart.
CENSOR_CAPS_NM = [10000.0, 100000.0, 1000000.0, 10000000.0]
MIN_LIG_FOR_OFFSET = 10
TEST_PROTEIN_FRAC, TEST_SCAFFOLD_FRAC, INDOMAIN_FRAC = 0.20, 0.20, 0.05
ECFP_BITS = 1024
AA = "ACDEFGHIKLMNPQRSTVWY"
LIGAND_HF = "DeepChem/ChemBERTa-77M-MLM"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# data (only rebuilt if --pairs/--static are not supplied)
# --------------------------------------------------------------------------- #

def build_pairs(csv: Path, out_dir: Path):
    pairs_p, static_p = out_dir / "pairs_clean.parquet", out_dir / "static.npz"
    if pairs_p.exists() and static_p.exists():
        log(f"reusing {pairs_p} and {static_p}")
        return pairs_p, static_p

    from rdkit import Chem, RDLogger
    from rdkit.Chem import (Crippen, Descriptors, Lipinski, rdFingerprintGenerator,
                            rdMolDescriptors)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    ## Censor caps MUST be dropped here, for two independent reasons, and this script
    ## used to do neither:
    ##
    ##  1. Label correctness. A row recorded at exactly 10 uM / 100 uM / 1 mM / 10 mM is
    ##     almost always a ">= this" non-binder readout, not a measurement. input.csv holds
    ##     ~82,820 of them. Keeping them trains and scores against a pKd that was never
    ##     measured.
    ##  2. Index agreement. train_savant_v2.ipynb drops them BEFORE factorising proteins,
    ##     so proteins that appear only in capped rows are absent from its protein list -
    ##     and therefore from the manifest, and therefore from every embedding cache. When
    ##     this script kept them it produced 21,330 proteins against the caches' 21,148,
    ##     and all 182 extras were unembeddable. The sweep then rejected every encoder.
    ##
    ## The columns are read only if present, so a CSV without affinity_Kd_nM still works.
    cols = ["seq", "smiles", "neg_log10_affinity_M"]
    have_kd = "affinity_Kd_nM" in pd.read_csv(csv, nrows=0).columns
    if have_kd:
        cols.append("affinity_Kd_nM")
    raw = pd.read_csv(csv, usecols=cols)
    n0 = len(raw)
    if have_kd:
        cap = np.zeros(len(raw), bool)
        for c in CENSOR_CAPS_NM:
            cap |= np.isclose(raw.affinity_Kd_nM.to_numpy(np.float64), c, rtol=1e-6)
        raw = raw[~cap]
        log(f"  dropped {int(cap.sum()):,} censor-cap rows of {n0:,} "
            f"(matching train_savant_v2.ipynb, which does this before factorising "
            f"proteins - keeping them puts proteins in the index that no embedding "
            f"cache contains)")
    else:
        log(f"  {csv.name} has no affinity_Kd_nM column - cannot identify censor caps")
    raw = raw[raw.neg_log10_affinity_M.between(MIN_PKD, MAX_PKD)].reset_index(drop=True)
    raw["li"] = pd.factorize(raw.smiles)[0]; raw["pi"] = pd.factorize(raw.seq)[0]
    ligands = raw.drop_duplicates("li").sort_values("li").smiles.to_numpy()
    proteins = raw.drop_duplicates("pi").sort_values("pi").seq.to_numpy()

    g = raw.groupby(["li", "pi"])["neg_log10_affinity_M"]
    pairs = g.agg(pkd="median", n_meas="size", label_sd="std").reset_index()
    pairs["label_sd"] = pairs.label_sd.fillna(0.0)
    pairs = pairs[pairs.label_sd <= MAX_LABEL_SD].reset_index(drop=True)
    pairs["w"] = np.sqrt(pairs.n_meas.to_numpy(np.float32))

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_BITS)
    d11, full, fps, scaf, ok = [], [], [], [], []
    for i, s in enumerate(ligands):
        m = Chem.MolFromSmiles(s); ok.append(m is not None)
        if m is None:
            d11.append([0.0]*11); full.append({}); fps.append(np.zeros(ECFP_BITS)); scaf.append("")
            continue
        mh = Chem.AddHs(m)
        d11.append([Descriptors.MolWt(mh), Crippen.MolLogP(mh), Lipinski.NumHDonors(mh),
                    Lipinski.NumHAcceptors(mh), Lipinski.NumRotatableBonds(mh),
                    rdMolDescriptors.CalcTPSA(mh), rdMolDescriptors.CalcNumRings(mh),
                    rdMolDescriptors.CalcNumAromaticRings(mh), mh.GetNumAtoms(),
                    mh.GetNumHeavyAtoms(), Chem.GetFormalCharge(mh)])
        full.append(Descriptors.CalcMolDescriptors(m))
        fps.append(gen.GetCountFingerprintAsNumPy(m))
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m); scaf.append(sc if sc else Chem.MolToSmiles(m))
        if (i+1) % 100_000 == 0: log(f"  ligands {i+1:,}/{len(ligands):,}")
    fd = pd.DataFrame(full).astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fd = fd.loc[:, fd.max() != fd.min()]
    aac = np.array([[s.count(a)/len(s) for a in AA] + [len(s)] for s in proteins], np.float32)

    pairs = pairs[np.asarray(ok)[pairs.li.to_numpy()]].reset_index(drop=True)
    pairs.to_parquet(pairs_p, index=False)
    np.savez_compressed(static_p, d11=np.asarray(d11, np.float32), dfull=fd.to_numpy(np.float32),
                        ecfp=np.asarray(fps, np.float32), aac=aac, scaf=np.asarray(scaf),
                        proteins=proteins, ligands=ligands)
    (out_dir / "rdkit_columns.json").write_text(json.dumps(list(fd.columns)))
    log(f"built {pairs_p} ({len(pairs):,} pairs) and {static_p}")
    return pairs_p, static_p


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #

def make_split(pi_all, sp_all, seed=SEED):
    rng = np.random.default_rng(seed)
    per_p = pd.Series(pi_all).value_counts()
    bins = np.floor(np.log2(per_p.clip(lower=1))).astype(int)
    test_p = []
    for b in np.unique(bins):
        ps = per_p.index[bins == b].to_numpy()
        test_p.extend(rng.choice(ps, size=max(1, int(round(TEST_PROTEIN_FRAC*len(ps)))),
                                 replace=False).tolist())
    test_p = set(test_p)
    uscaf = np.unique(sp_all)
    test_s = set(rng.choice(uscaf, size=max(1, int(round(TEST_SCAFFOLD_FRAC*len(uscaf)))),
                            replace=False).tolist())
    npr = np.array([p in test_p for p in pi_all]); ns_ = np.array([s in test_s for s in sp_all])
    train_pool = np.where(~npr & ~ns_)[0]; rng.shuffle(train_pool)
    n_ind = int(INDOMAIN_FRAC * len(train_pool))
    cells = {"seen prot / seen chem":   train_pool[:n_ind],
             "seen prot / NOVEL chem":  np.where(~npr & ns_)[0],
             "NOVEL prot / seen chem":  np.where(npr & ~ns_)[0],
             "NOVEL prot / NOVEL chem": np.where(npr & ns_)[0]}
    tr_idx = train_pool[n_ind:]
    tr_prot, tr_scaf = set(pi_all[tr_idx]), set(sp_all[tr_idx])
    for k in ("NOVEL prot / seen chem", "NOVEL prot / NOVEL chem"):
        assert not (set(pi_all[cells[k]]) & tr_prot), f"protein leak into {k}"
    for k in ("seen prot / NOVEL chem", "NOVEL prot / NOVEL chem"):
        assert not (set(sp_all[cells[k]]) & tr_scaf), f"scaffold leak into {k}"
    return tr_idx, cells, rng


# --------------------------------------------------------------------------- #
# discover encoders
# --------------------------------------------------------------------------- #

def _keys_for(p: Path):
    """The `<slug>_keys.npy` companion written by train_savant_v2.ipynb, or None."""
    kp = p.with_name(p.stem + "_keys.npy")
    if not kp.exists():
        return None
    return np.load(kp, allow_pickle=True)


def align_cache(p: Path, want, out_dir: Path, label: str):
    """Return a path to an embedding matrix whose rows line up with `want`, or None.

    Three outcomes, and the distinction between the first two is the whole point:

      * the cache already has one row per wanted text, in that order -> use it in place;
      * the cache is a SUPERSET with a key file (the usual case here - the caches were
        built on input.csv + the expanded corpus, and this script rebuilds from input.csv
        alone) -> write an aligned subset next to the run and use that;
      * anything else -> None, and the caller skips the encoder.

    A bare row-count match is NOT accepted when a key file exists and disagrees. Two
    different protein indexes of equal length would otherwise pair every protein with
    another protein's embedding, silently, and every number downstream would be wrong in
    a way no assertion later could catch.
    """
    want = list(want)
    keys = _keys_for(p)
    a0 = np.load(p, mmap_mode="r")
    shape = a0.shape
    mm0 = getattr(a0, "_mmap", None)
    del a0
    if mm0 is not None:
        mm0.close()
    ## An encoder is a (n_items, n_dims) matrix. `<mode>_valid.npy` is a 1-D boolean mask
    ## written next to the merged embeddings, and it has exactly the row count an encoder
    ## would have - so row count alone cannot tell them apart. Requiring 2-D does.
    if len(shape) != 2:
        log(f"  {label}: {len(shape)}-D array {shape}, not an embedding matrix - skipping")
        return None
    n_rows = shape[0]

    if keys is None:
        if n_rows == len(want):
            log(f"  {label}: {n_rows:,} rows, no key file - accepted on row count alone")
            return p
        log(f"  {label}: {n_rows:,} rows, expected {len(want):,}, and no key file to map "
            f"them onto - skipping")
        return None

    if len(keys) != n_rows:
        log(f"  {label}: {n_rows:,} rows but {len(keys):,} keys - inconsistent cache, "
            f"skipping")
        return None

    if n_rows == len(want) and np.array_equal(np.asarray(keys), np.asarray(want, object)):
        log(f"  {label}: {n_rows:,} rows, keys match exactly - included")
        return p

    idx = pd.Series(np.arange(len(keys)), index=keys).reindex(want).to_numpy()
    missing = int(np.isnan(idx).sum())
    if missing:
        log(f"  {label}: {missing:,} of {len(want):,} wanted texts are not in this cache "
            f"- skipping (it was built against a different index)")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / p.name
    a = np.load(p, mmap_mode="r")
    sub = np.array(a[idx.astype(int)], dtype=np.float32)
    mm = getattr(a, "_mmap", None)
    del a
    if mm is not None:
        mm.close()          # np.array() already copied; release the handle explicitly
    np.save(dst, sub)
    np.save(dst.with_name(dst.stem + "_keys.npy"), np.asarray(want, object))
    log(f"  {label}: SUBSET {n_rows:,} -> {len(want):,} rows by key -> {dst.name}")
    return dst


def discover_encoders(emb_dir: Path, proteins, explicit: list[str] | None, out_dir: Path):
    found = {}
    if explicit:
        candidates = [emb_dir / f"{n}.npy" for n in explicit]
    else:
        ## Exclude the `*_keys.npy` companions. They are object-dtype arrays of SMILES /
        ## sequences, and np.load(mmap_mode="r") raises "Array can't be memory-mapped:
        ## Python objects in dtype" on them - which is how this script used to die before
        ## it ever reported a single encoder.
        candidates = sorted(q for q in emb_dir.glob("*.npy")
                            if not q.stem.endswith(("_keys", "_valid")))
    aligned_dir = out_dir / "emb_aligned"
    for p in candidates:
        if not p.exists():
            log(f"  {p.name}: not found, skipping"); continue
        try:
            dst = align_cache(p, proteins, aligned_dir, p.name)
        except ValueError as e:
            ## Defensive: an object-dtype .npy that is not named *_keys.npy would still
            ## reach here. Skip it rather than taking the whole sweep down.
            log(f"  {p.name}: not a numeric matrix ({e}) - skipping"); continue
        if dst is not None:
            found[p.stem] = dst
    return found


# --------------------------------------------------------------------------- #
# Part A - ridge diagnostic
# --------------------------------------------------------------------------- #

def part_a(emb_path, pi_all, y_all, tr_idx, cells, deploy_key):
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    def offsets_from(idx):
        d = pd.DataFrame({"pi": pi_all[idx], "y": y_all[idx]})
        a = d.groupby("pi")["y"].agg(["median", "size"])
        return a[a["size"] >= MIN_LIG_FOR_OFFSET]["median"]

    off_tr = offsets_from(tr_idx)
    off_te = offsets_from(cells[deploy_key])
    if len(off_te) < 10:
        off_te = offsets_from(np.concatenate([cells["NOVEL prot / seen chem"], cells[deploy_key]]))
    if len(off_tr) < 10 or len(off_te) < 10:
        return ({"r": np.nan, "rho": np.nan, "rmse": np.nan, "rmse_const": np.nan,
                 "n_test_proteins": len(off_te)},
                pd.DataFrame(columns=["pi", "true_offset", "pred_offset"]))

    E = np.load(emb_path, mmap_mode="r")
    Xtr = np.asarray(E[off_tr.index.to_numpy()], np.float32)
    Xte = np.asarray(E[off_te.index.to_numpy()], np.float32)
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(sc.transform(Xtr), off_tr.to_numpy())
    p = m.predict(sc.transform(Xte)); t = off_te.to_numpy()
    ## Per-protein predictions come back alongside the summary for the same reason Part B
    ## returns per-pair ones: with n ~ 400 test proteins, an unpaired comparison of two
    ## encoders' r cannot resolve anything smaller than ~0.08, so the ordering this
    ## diagnostic reports is uninterpretable without a paired test.
    detail = pd.DataFrame({"pi": off_te.index.to_numpy(), "true_offset": t, "pred_offset": p})
    return {"r": float(np.corrcoef(t, p)[0, 1]), "rho": float(spearmanr(t, p).correlation),
           "rmse": float(np.sqrt(((t-p)**2).mean())), "rmse_const": float(t.std()),
           "n_test_proteins": len(off_te)}, detail


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

MIN_TGT_PAIRS = 10


def decompose(y, p, g):
    """within-target r, between-target r, and the median per-target Spearman.

    Deliberately the same estimator as `train_savant_within.ipynb` (subtract each target's
    mean from BOTH truth and prediction, then correlate) so a number from this sweep can be
    put next to one from the notebook without a footnote.

    within_r is the metric that matters for the screening claim: it is invariant to any
    per-target constant, so a model is judged only on how it ranks ligands against one
    target. Pooled Pearson mixes that with between-target placement, which this benchmark's
    own construction makes adversarial - which is why the encoder ordering by pooled r is
    not safe to act on by itself.
    """
    from scipy.stats import spearmanr

    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    codes, uniq = pd.factorize(g)
    ng = len(uniq)
    cnt = np.bincount(codes, minlength=ng).astype(np.float64)
    my = np.bincount(codes, weights=y, minlength=ng) / np.maximum(cnt, 1)
    mp = np.bincount(codes, weights=p, minlength=ng) / np.maximum(cnt, 1)
    cy, cp = y - my[codes], p - mp[codes]

    within = (float(np.corrcoef(cy, cp)[0, 1])
              if cy.std() > 1e-9 and cp.std() > 1e-9 else np.nan)
    between = (float(np.corrcoef(my, mp)[0, 1])
               if ng > 2 and mp.std() > 1e-12 and my.std() > 1e-12 else np.nan)

    sp = []
    d = pd.DataFrame({"y": y, "p": p, "c": codes})
    for _, sub in d.groupby("c", sort=False):
        if len(sub) < MIN_TGT_PAIRS or sub.p.std() < 1e-12 or sub.y.std() < 1e-12:
            continue
        s = spearmanr(sub.y, sub.p).correlation
        if np.isfinite(s):
            sp.append(s)
    return {"within_r": within, "between_r": between, "n_targets": ng,
            "n_targets_scored": len(sp),
            "tgt_spearman_median": float(np.median(sp)) if sp else np.nan,
            "tgt_spearman_pos_frac": float(np.mean(np.array(sp) > 0)) if sp else np.nan}


# --------------------------------------------------------------------------- #
# Part B - real regressor
# --------------------------------------------------------------------------- #

def part_b(emb_path, S, pairs, tr_idx, cells, deploy_key, ligand_emb):
    import xgboost as xgb
    from scipy.stats import spearmanr

    li_all, pi_all, y_all, w_all = (pairs.li.to_numpy(), pairs.pi.to_numpy(),
                                    pairs.pkd.to_numpy(np.float32), pairs.w.to_numpy(np.float32))
    P = np.load(emb_path, mmap_mode="r")

    def X(idx):
        return np.hstack([ligand_emb[li_all[idx]], S["d11"][li_all[idx]], S["dfull"][li_all[idx]],
                          S["ecfp"][li_all[idx]], np.asarray(P[pi_all[idx]], np.float32),
                          S["aac"][pi_all[idx]]]).astype(np.float32)

    rng = np.random.default_rng(SEED)
    tr_p = np.unique(pi_all[tr_idx])
    val_p = set(rng.choice(tr_p, size=max(1, int(0.15*len(tr_p))), replace=False).tolist())
    is_val = np.array([p in val_p for p in pi_all[tr_idx]])
    fit_idx, val_idx = tr_idx[~is_val], tr_idx[is_val]

    dtr = xgb.QuantileDMatrix(X(fit_idx), label=y_all[fit_idx], weight=w_all[fit_idx])
    dva = xgb.QuantileDMatrix(X(val_idx), label=y_all[val_idx], ref=dtr)
    params = dict(objective="reg:squarederror", eval_metric="rmse", max_depth=8, eta=0.05,
                 subsample=0.8, colsample_bytree=0.5, min_child_weight=10, reg_lambda=2.0,
                 tree_method="hist", seed=SEED)
    b = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")],
                 early_stopping_rounds=50, verbose_eval=False)
    del dtr, dva; gc.collect()

    out, preds = {}, []
    for cell, idx in cells.items():
        p = b.inplace_predict(X(idx)); y = y_all[idx]
        out[cell] = {"pearson": float(np.corrcoef(y, p)[0, 1]),
                    "spearman": float(spearmanr(y, p).correlation),
                    "rmse": float(np.sqrt(((y - p) ** 2).mean())),
                    "rmse_const": float(y.std()),
                    **decompose(y, p, pi_all[idx])}
        ## Per-pair predictions, kept so two encoders can be compared with a PAIRED test.
        ## Without them only an unpaired comparison of two r values is possible, which
        ## throws away most of the power - a 0.03 gap at n=438 came out p=0.58 unpaired and
        ## could not be called either way. Cheap to store, impossible to recover later.
        preds.append(pd.DataFrame({"cell": cell, "pi": pi_all[idx], "li": li_all[idx],
                                   "y": y, "p": p}))
    return out, pd.concat(preds, ignore_index=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(ROOT.parent / "input" / "input.csv"))
    ap.add_argument("--emb-dir", default=str(ROOT / "work" / "emb"))
    ap.add_argument("--out-dir", default=str(ROOT / "work" / "split_test"))
    ap.add_argument("--pairs", help="reuse an existing pairs_clean.parquet")
    ap.add_argument("--static", help="reuse an existing static.npz")
    ap.add_argument("--encoders", nargs="*",
                    help="specific .npy stems to sweep (default: auto-discover all in --emb-dir)")
    ap.add_argument("--skip-part-b", action="store_true",
                    help="ridge diagnostic only - skip the slower real regressor")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.pairs and args.static:
        pairs_p, static_p = Path(args.pairs), Path(args.static)
    else:
        pairs_p, static_p = build_pairs(Path(args.csv), out_dir)

    pairs = pd.read_parquet(pairs_p)
    z = np.load(static_p, allow_pickle=True)
    S = {k: z[k] for k in z.files}
    proteins = S["proteins"]
    scaf_u = pd.factorize(S["scaf"])[0]
    pi_all, li_all = pairs.pi.to_numpy(), pairs.li.to_numpy()
    y_all, sp_all = pairs.pkd.to_numpy(np.float32), scaf_u[li_all]

    tr_idx, cells, _ = make_split(pi_all, sp_all)
    DEPLOY = "NOVEL prot / NOVEL chem"
    log(f"train {len(tr_idx):,} pairs; cells: " +
        ", ".join(f"{k}={len(v):,}" for k, v in cells.items()))

    log(f"discovering protein encoders in {args.emb_dir} ({len(proteins):,} proteins expected)")
    encoders = discover_encoders(Path(args.emb_dir), proteins, args.encoders, out_dir)
    if not encoders:
        raise SystemExit(
            f"no usable .npy files in {args.emb_dir}.\n"
            "Each candidate must either have one row per protein in this run's index, or "
            "carry a `<slug>_keys.npy` companion listing the sequence for each row so a "
            "superset cache can be subset onto it.\n"
            "  * no key files at all? run scripts/backfill_embedding_keys.py\n"
            "  * no embeddings at all? run scripts/embed_structure_encoders.py first, or "
            "point --emb-dir at an existing cache (e.g. ../ML_model/work_two_part/emb)")

    ligand_emb = None
    if not args.skip_part_b:
        lig_cache = out_dir / "ligand_emb_chemberta.npy"
        ## Same superset problem as the protein encoders, and far more expensive to get
        ## wrong: without this the script re-embeds ~960k ligands on a GPU even though
        ## work/emb/chemberta-mlm77.npy already holds every one of them.
        lig_src = None
        if lig_cache.exists():
            lig_src = align_cache(lig_cache, S["ligands"], out_dir / "emb_aligned",
                                  lig_cache.name)
        if lig_src is None:
            donor = Path(args.emb_dir) / "chemberta-mlm77.npy"
            if donor.exists():
                lig_src = align_cache(donor, S["ligands"], out_dir / "emb_aligned",
                                      donor.name)
        if lig_src is not None:
            ligand_emb = np.load(lig_src)
        else:
            log("  no reusable ChemBERTa cache covers this ligand set - embedding "
                f"{len(S['ligands']):,} ligands (needs a GPU)")
            import torch
            from transformers import AutoModel, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            mdl = AutoModel.from_pretrained(LIGAND_HF).to(dev).eval()
            tok = AutoTokenizer.from_pretrained(LIGAND_HF)
            texts = list(S["ligands"]); out = np.zeros((len(texts), 384), np.float32)
            with torch.inference_mode():
                for i in range(0, len(texts), 512):
                    b = tok(texts[i:i+512], padding=True, truncation=True, max_length=256,
                           return_tensors="pt")
                    b = {k: v.to(dev) for k, v in b.items()}
                    h = mdl(**b).last_hidden_state
                    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                    out[i:i+512] = ((h*m).sum(1)/m.sum(1).clamp_min(1)).float().cpu().numpy()
            np.save(lig_cache, out)
            ## Write the key file too, so a later run with a different ligand set can
            ## subset this instead of paying for the GPU pass again.
            np.save(lig_cache.with_name(lig_cache.stem + "_keys.npy"),
                    np.asarray(texts, object))
            ligand_emb = out

    rows_a, rows_b, pred_a, pred_b = [], {}, [], []
    for name, path in encoders.items():
        log(f"=== {name} ===")
        a, a_detail = part_a(path, pi_all, y_all, tr_idx, cells, DEPLOY)
        rows_a.append({"encoder": name, **a})
        pred_a.append(a_detail.assign(encoder=name))
        log(f"  Part A (offset ridge): r={a['r']:.3f} rmse={a['rmse']:.3f} "
            f"vs const={a['rmse_const']:.3f} (n={a['n_test_proteins']} proteins)")
        if not args.skip_part_b:
            b, b_detail = part_b(path, S, pairs, tr_idx, cells, DEPLOY, ligand_emb)
            rows_b[name] = b
            pred_b.append(b_detail.assign(encoder=name))
            log(f"  Part B (regressor) deployment cell: "
                f"within_r={b[DEPLOY]['within_r']:.3f} pearson={b[DEPLOY]['pearson']:.3f}")

    A = pd.DataFrame(rows_a).sort_values("r", ascending=False)
    A.to_csv(out_dir / "part_a_sweep.csv", index=False)
    if pred_a:
        pd.concat(pred_a, ignore_index=True).to_csv(out_dir / "part_a_predictions.csv",
                                                    index=False)
    print("\n=== Part A: offset ridge, sorted by r ===")
    print(A.round(4).to_string(index=False))
    if len(A) > 1 and A.n_test_proteins.notna().all():
        ## Say out loud how small a difference this diagnostic can actually resolve, so
        ## nobody reads an ordering into a gap that is pure sampling noise.
        n_a = int(A.n_test_proteins.iloc[0])
        res = 2 * 1.96 * np.sqrt(2.0 / max(n_a - 3, 1))
        print(f"\n  resolution limit: with n={n_a} proteins, two encoders' r must differ "
              f"by roughly {np.tanh(res):.3f} (Fisher z, unpaired) before the ordering "
              f"means anything. Use part_a_predictions.csv for a paired test.")

    if rows_b:
        B = pd.DataFrame({(enc, cell): vals for enc, cellvals in rows_b.items()
                          for cell, vals in cellvals.items()}).T
        B.index.names = ["encoder", "cell"]
        B.to_csv(out_dir / "part_b_sweep.csv")
        pd.concat(pred_b, ignore_index=True).to_parquet(
            out_dir / "part_b_predictions.parquet", index=False)
        for metric, blurb in [("within_r", "the screening metric - ranking ligands "
                                           "WITHIN one target"),
                              ("pearson", "pooled, mixes within- and between-target")]:
            print(f"\n=== Part B: {metric} by encoder x cell ({blurb}) ===")
            print(B[metric].unstack("cell").reindex(columns=list(cells)).round(4).to_string())

    log(f"\nwrote {out_dir}/part_a_sweep.csv, part_a_predictions.csv"
        + (", part_b_sweep.csv, part_b_predictions.parquet" if rows_b else ""))
    if rows_b:
        log("per-pair predictions are saved - two encoders can now be compared with a "
            "PAIRED test (Steiger) instead of two unpaired r values")


if __name__ == "__main__":
    main()
