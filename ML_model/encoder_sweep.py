"""Encoder sweep: which protein/ligand embedding pair actually helps Savant?

    # once, on a GPU node - builds every embedding cache
    python encoder_sweep.py --stage embed

    # then, on CPU - trains and scores the grid
    python encoder_sweep.py --stage train --mode staged

What this measures, and why it is built this way
------------------------------------------------

The v3 benchmark already showed that scaling the protein encoder made Savant
*worse* pooled (esm8 r=0.330 vs esm650 r=0.227) while leaving within-target
ranking untouched (0.433 vs 0.436). The entire loss was between-target
(0.257 -> 0.100). So this sweep reports the **decomposition, not just pooled r** -
a model that wins on pooled r by memorising target identity is the failure mode
being hunted, and only the between/within split makes it visible.

Three rules keep it an encoder comparison rather than an accident:

1. **One fixed XGBoost config for every run.** No per-encoder tuning, or the
   winner is whichever model got the luckiest hyperparameters.
2. **One fixed set of split indices for every run**, computed once and reused.
3. **Two test sets, always reported together.** The internal protein+scaffold
   double-disjoint split, and the external 500-pair benchmark. v2's step 7/8
   found the internal split *ranked models the wrong way round*, so a winner
   that only wins internally is not a winner.

Training uses `combined_50k_filtered.csv` on purpose. The external benchmark was
gated at ECFP4 Tanimoto <= 0.50 against exactly those 32,086 ligands, so it stays
a valid held-out set here. It is NOT valid for a model trained on input.csv -
see `bench_ligand_novelty_vs_input.csv` before mixing the two.

Cost: the embed stage dominates and is cached per unique sequence / SMILES, so
adding a model later only pays for that model. 12,251 proteins and 32,086
ligands; ESM2-650M is ~5 min on a GPU, ProtT5-XL ~15 min, the ligand models are
a couple of minutes each.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark/eval/output/bench_scored.csv"

# --------------------------------------------------------------------------- #
# encoder registry
# --------------------------------------------------------------------------- #
# kind: "bert" -> AutoModel + plain tokenizer
#       "t5"   -> T5EncoderModel, residues space-separated, UZOB -> X
#       "ankh" -> T5EncoderModel, tokenised as a list of characters

PROTEIN_ENCODERS = {
    "esm2-8m":   dict(hf="facebook/esm2_t6_8M_UR50D",              kind="bert"),
    "esm2-35m":  dict(hf="facebook/esm2_t12_35M_UR50D",            kind="bert"),
    "esm2-150m": dict(hf="facebook/esm2_t30_150M_UR50D",           kind="bert"),
    "esm2-650m": dict(hf="facebook/esm2_t33_650M_UR50D",           kind="bert"),
    "esmc-300m": dict(hf="Synthyra/ESMplusplus_small",             kind="bert", trust=True),
    "prott5":    dict(hf="Rostlab/prot_t5_xl_half_uniref50-enc",   kind="t5"),
    "ankh-base": dict(hf="ElnaggarLab/ankh-base",                  kind="ankh"),
}

LIGAND_ENCODERS = {
    "chemberta-mlm77": dict(hf="DeepChem/ChemBERTa-77M-MLM",       kind="bert"),
    "chemberta-mtr77": dict(hf="DeepChem/ChemBERTa-77M-MTR",       kind="bert"),
    "chemberta-mtr10": dict(hf="DeepChem/ChemBERTa-10M-MTR",       kind="bert"),
    "chemberta-zinc":  dict(hf="seyonec/ChemBERTa-zinc-base-v1",   kind="bert"),
    "molformer":       dict(hf="ibm/MoLFormer-XL-both-10pct",      kind="bert", trust=True),
    "none":            dict(hf=None,                               kind="none"),
}

## baselines the staged mode holds fixed
BASE_PROTEIN = "esm2-8m"
BASE_LIGAND = "chemberta-mlm77"

AA = "ACDEFGHIKLMNPQRSTVWY"
XGB_PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", max_depth=7,
                  eta=0.05, subsample=0.8, colsample_bytree=0.5,
                  min_child_weight=5, tree_method="hist")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def digest(items) -> str:
    h = hashlib.sha1()
    for s in items:
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #

def _load(spec, device):
    import torch
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

    name, kind, trust = spec["hf"], spec["kind"], spec.get("trust", False)
    if kind in ("t5", "ankh"):
        tok = T5Tokenizer.from_pretrained(name, do_lower_case=False, legacy=True)
        mdl = T5EncoderModel.from_pretrained(name)
    else:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
        mdl = AutoModel.from_pretrained(name, trust_remote_code=trust,
                                        deterministic_eval=True) if "MoLFormer" in name \
            else AutoModel.from_pretrained(name, trust_remote_code=trust)
    return tok, mdl.to(device).eval(), torch


def embed(texts, spec, batch_size, max_length, device=None):
    """Mean-pooled last hidden state, padding masked out.

    Returns (n, dim) float32. Batched shortest-first so padding is minimal.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, mdl, _ = _load(spec, device)
    kind = spec["kind"]
    log(f"  {spec['hf']} on {device}, {len(texts)} items")

    order = np.argsort([len(t) for t in texts])
    out = None
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            sel = order[start:start + batch_size]
            raw = [texts[i] for i in sel]

            if kind == "t5":
                prepped = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in raw]
                batch = tok(prepped, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            elif kind == "ankh":
                prepped = [list(re.sub(r"[UZOB]", "X", s))[:max_length] for s in raw]
                batch = tok.batch_encode_plus(prepped, padding=True,
                                              is_split_into_words=True,
                                              return_tensors="pt")
            else:
                batch = tok(raw, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")

            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu().numpy()

            if out is None:
                out = np.zeros((len(texts), pooled.shape[1]), dtype=np.float32)
            out[sel] = pooled
            if (start // batch_size) % 50 == 0:
                log(f"    {min(start + batch_size, len(order))}/{len(order)}")

    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def cached_embed(texts, slug, spec, cache_dir, batch_size, max_length):
    """Embed once per unique text, keyed by (model, exact text list)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{slug}__{digest(texts)}.npy"
    if path.exists():
        log(f"  reuse {path.name}")
        return np.load(path)
    arr = embed(texts, spec, batch_size, max_length)
    np.save(path, arr)
    log(f"  wrote {path.name} {arr.shape}")
    return arr


# --------------------------------------------------------------------------- #
# non-embedding feature blocks
# --------------------------------------------------------------------------- #

def ligand_blocks(smiles):
    """(11 AddHs descriptors, full RDKit descriptor frame, ECFP4 counts)."""
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

    ## cast first, then clean - Ipc and Kappa2 only overflow to inf on the cast
    fd = pd.DataFrame(full).astype(np.float32)
    fd = fd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fd = fd.loc[:, fd.max() != fd.min()]
    return (np.asarray(d11, np.float32), fd.to_numpy(np.float32),
            np.asarray(fps, np.float32))


def aa_composition(seqs):
    return np.array([[s.count(a) / len(s) for a in AA] + [len(s)] for s in seqs],
                    dtype=np.float32)


def scaffolds_of(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        out.append(sc if sc else Chem.MolToSmiles(m))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_frames(args):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    tr = pd.read_csv(args.csv, usecols=["seq", "smiles", "neg_log10_affinity_M", "vdw"])
    n0 = len(tr)
    tr = tr[tr.neg_log10_affinity_M >= args.min_pkd]
    tr = tr[~(tr["vdw"] > 1000)]                      # clashing poses, not energies
    tr = tr[[Chem.MolFromSmiles(s) is not None for s in tr.smiles]].reset_index(drop=True)
    log(f"train rows {n0} -> {len(tr)}")

    bench = pd.read_csv(BENCH, usecols=["target_sequence", "ligand_smiles",
                                        "neg_log10_affinity_M", "uniprot_id"])
    bench = bench.rename(columns={"target_sequence": "seq", "ligand_smiles": "smiles"})
    bench = bench.dropna(subset=["seq", "smiles", "neg_log10_affinity_M"])
    bench = bench[[Chem.MolFromSmiles(s) is not None for s in bench.smiles]].reset_index(drop=True)
    log(f"benchmark rows {len(bench)}, {bench.uniprot_id.nunique()} targets")
    return tr, bench


def build_static(tr, bench, out):
    """Everything that does not depend on which encoder is being tested."""
    cache = out / "static.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    all_lig = pd.Index(pd.concat([tr.smiles, bench.smiles]).unique())
    all_prot = pd.Index(pd.concat([tr.seq, bench.seq]).unique())
    log(f"unique ligands {len(all_lig)}, unique proteins {len(all_prot)}")

    d11, dfull, ecfp = ligand_blocks(all_lig.tolist())
    aac = aa_composition(all_prot.tolist())
    scaf = scaffolds_of(all_lig.tolist())

    d = dict(
        ligands=all_lig.to_numpy(), proteins=all_prot.to_numpy(),
        d11=d11, dfull=dfull, ecfp=ecfp, aac=aac, scaf=scaf,
        tr_li=all_lig.get_indexer(tr.smiles), tr_pi=all_prot.get_indexer(tr.seq),
        be_li=all_lig.get_indexer(bench.smiles), be_pi=all_prot.get_indexer(bench.seq),
        y_tr=tr.neg_log10_affinity_M.to_numpy(np.float32),
        y_be=bench.neg_log10_affinity_M.to_numpy(np.float32),
        bench_group=bench.uniprot_id.to_numpy(),
    )
    np.savez_compressed(cache, **d)
    log(f"static blocks cached -> {cache}")
    return d


def disjoint_split(prot_idx, scaf_idx, seed):
    """Protein AND scaffold double-disjoint, as in savant_v2 step 7.

    Rows that straddle - new protein but trained scaffold, or the reverse - are
    dropped rather than assigned to a side, because keeping them reintroduces
    exactly the leak this split exists to remove.
    """
    rng = np.random.default_rng(seed)
    up, us = np.unique(prot_idx), np.unique(scaf_idx)
    te_p = set(rng.choice(up, size=max(1, int(0.20 * len(up))), replace=False).tolist())
    te_s = set(rng.choice(us, size=max(1, int(0.20 * len(us))), replace=False).tolist())

    in_tep = np.array([p in te_p for p in prot_idx])
    in_tes = np.array([s in te_s for s in scaf_idx])
    test = np.where(in_tep & in_tes)[0]
    train = np.where(~in_tep & ~in_tes)[0]
    dropped = len(prot_idx) - len(test) - len(train)
    log(f"double-disjoint: train {len(train)}, test {len(test)}, "
        f"straddling dropped {dropped} ({dropped/len(prot_idx):.0%})")
    return train, test


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def decompose(y, p, groups):
    """(between_r, within_r): does the model rank targets, or ligands inside one?"""
    df = pd.DataFrame({"y": y, "p": p, "g": groups})
    m = df.groupby("g")[["y", "p"]].mean()
    between = float(np.corrcoef(m.y, m.p)[0, 1]) if len(m) > 2 else np.nan
    c = df.join(m, on="g", rsuffix="_m")
    cy, cp = c.y - c.y_m, c.p - c.p_m
    within = (float(np.corrcoef(cy, cp)[0, 1])
              if cy.std() > 1e-9 and cp.std() > 1e-9 else np.nan)
    return between, within


def score(y, p, groups=None):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    out = dict(
        n=len(y),
        pearson=float(np.corrcoef(y, p)[0, 1]),
        spearman=float(spearmanr(y, p).correlation),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)),
        ## a tool above this line is not beating 'predict the training mean'
        rmse_constant=float(np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))),
        pred_std_ratio=float(p.std() / y.std()),
    )
    if groups is not None:
        out["between_r"], out["within_r"] = decompose(y, p, groups)
    return out


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #

def assemble(S, prot_emb, lig_emb, side):
    """Stack the feature blocks for one (encoder pair, dataset side)."""
    li, pi = (S["tr_li"], S["tr_pi"]) if side == "train" else (S["be_li"], S["be_pi"])
    blocks = [S["d11"][li], S["dfull"][li], S["ecfp"][li], S["aac"][pi]]
    if lig_emb is not None:
        blocks.insert(0, lig_emb[li])
    if prot_emb is not None:
        blocks.insert(1, prot_emb[pi])
    return np.hstack(blocks).astype(np.float32)


def run_one(S, split, prot_emb, lig_emb, rounds, threads, seed):
    import xgboost as xgb

    X_tr = assemble(S, prot_emb, lig_emb, "train")
    X_be = assemble(S, prot_emb, lig_emb, "bench")
    y_tr, y_be = S["y_tr"], S["y_be"]
    train, test = split

    ## inner split for early stopping, grouped on protein so it holds out the
    ## same kind of thing the outer split does
    rng = np.random.default_rng(seed)
    gp = S["tr_pi"][train]
    up = np.unique(gp)
    val_p = set(rng.choice(up, size=max(1, int(0.2 * len(up))), replace=False).tolist())
    is_val = np.array([p in val_p for p in gp])
    fit, val = train[~is_val], train[is_val]

    dfit = xgb.QuantileDMatrix(X_tr[fit], label=y_tr[fit])
    dval = xgb.QuantileDMatrix(X_tr[val], label=y_tr[val], ref=dfit)
    params = {**XGB_PARAMS, "nthread": threads, "seed": seed}
    b = xgb.train(params, dfit, num_boost_round=rounds, evals=[(dval, "val")],
                  early_stopping_rounds=40, verbose_eval=False)

    internal = score(y_tr[test], b.inplace_predict(X_tr[test]), S["tr_pi"][test])
    external = score(y_be, b.inplace_predict(X_be), S["bench_group"])
    return internal, external, X_tr.shape[1], int(b.best_iteration)


def pairs_for(mode, prot_models, lig_models):
    if mode == "full":
        return list(itertools.product(prot_models, lig_models))
    seen, out = set(), []
    for p in prot_models:                       # phase A: protein axis
        out.append((p, BASE_LIGAND))
    for l in lig_models:                        # phase B: ligand axis
        out.append((BASE_PROTEIN, l))
    return [x for x in out if not (x in seen or seen.add(x))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=str(HERE / "combined_50k_filtered.csv"))
    ap.add_argument("--out", default=str(HERE / "work_sweep"))
    ap.add_argument("--stage", choices=["embed", "train", "all"], default="all")
    ap.add_argument("--mode", choices=["staged", "full"], default="staged")
    ap.add_argument("--protein-models", default=",".join(PROTEIN_ENCODERS))
    ap.add_argument("--ligand-models", default=",".join(LIGAND_ENCODERS))
    ap.add_argument("--min-pkd", type=float, default=4.0)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--protein-batch", type=int, default=8)
    ap.add_argument("--ligand-batch", type=int, default=256)
    ap.add_argument("--protein-maxlen", type=int, default=1024)
    ap.add_argument("--ligand-maxlen", type=int, default=256)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb_dir = out / "emb"

    prot_models = [m for m in args.protein_models.split(",") if m in PROTEIN_ENCODERS]
    lig_models = [m for m in args.ligand_models.split(",") if m in LIGAND_ENCODERS]

    tr, bench = load_frames(args)
    S = build_static(tr, bench, out)
    proteins, ligands = list(S["proteins"]), list(S["ligands"])

    if args.stage in ("embed", "all"):
        for m in prot_models:
            log(f"protein encoder {m}")
            cached_embed(proteins, m, PROTEIN_ENCODERS[m], emb_dir,
                         args.protein_batch, args.protein_maxlen)
        for m in lig_models:
            if LIGAND_ENCODERS[m]["hf"] is None:
                continue
            log(f"ligand encoder {m}")
            cached_embed(ligands, m, LIGAND_ENCODERS[m], emb_dir,
                         args.ligand_batch, args.ligand_maxlen)
        if args.stage == "embed":
            log("embed stage done - rerun with --stage train")
            return

    ## one split, computed once, reused by every run
    _, scaf_u = np.unique(S["scaf"], return_inverse=True)
    split = disjoint_split(S["tr_pi"], scaf_u[S["tr_li"]], args.seed)

    def load_emb(slug, registry, texts):
        if registry[slug]["hf"] is None:
            return None
        return np.load(emb_dir / f"{slug}__{digest(texts)}.npy")

    rows = []
    for p_slug, l_slug in pairs_for(args.mode, prot_models, lig_models):
        log(f"=== {p_slug} x {l_slug} ===")
        try:
            pe = load_emb(p_slug, PROTEIN_ENCODERS, proteins)
            le = load_emb(l_slug, LIGAND_ENCODERS, ligands)
        except FileNotFoundError as e:
            log(f"  missing cache, skipping: {e}")
            continue

        t0 = time.time()
        internal, external, width, best_it = run_one(
            S, split, pe, le, args.rounds, args.threads, args.seed)
        row = {"protein": p_slug, "ligand": l_slug, "n_features": width,
               "best_iter": best_it, "fit_s": round(time.time() - t0, 1),
               **{f"int_{k}": v for k, v in internal.items()},
               **{f"ext_{k}": v for k, v in external.items()}}
        rows.append(row)
        log(f"  internal r {internal['pearson']:.3f} | external r {external['pearson']:.3f} "
            f"(between {external['between_r']:.3f}, within {external['within_r']:.3f}, "
            f"pred_std {external['pred_std_ratio']:.2f})")
        pd.DataFrame(rows).to_csv(out / "encoder_sweep.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        log("no runs completed")
        return

    cols = ["protein", "ligand", "n_features", "int_pearson", "ext_pearson",
            "ext_between_r", "ext_within_r", "ext_rmse", "ext_rmse_constant",
            "ext_pred_std_ratio"]
    print("\n" + df[cols].sort_values("ext_pearson", ascending=False).round(3).to_string(index=False))
    df.to_csv(out / "encoder_sweep.csv", index=False)
    json.dump({"xgb": XGB_PARAMS, "seed": args.seed, "csv": args.csv,
               "base_protein": BASE_PROTEIN, "base_ligand": BASE_LIGAND},
              open(out / "sweep_config.json", "w"), indent=1)

    if args.mode == "staged" and len(df) > 2:
        top_p = (df[df.ligand == BASE_LIGAND].nlargest(2, "ext_pearson").protein.tolist())
        top_l = (df[df.protein == BASE_PROTEIN].nlargest(2, "ext_pearson").ligand.tolist())
        print(f"\nphase C - cross the winners:\n"
              f"  python encoder_sweep.py --stage train --mode full "
              f"--protein-models {','.join(top_p)} --ligand-models {','.join(top_l)}")
    log(f"wrote {out / 'encoder_sweep.csv'}")


if __name__ == "__main__":
    main()
