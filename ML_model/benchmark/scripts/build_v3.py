"""Benchmark v3 - 500 Kd pairs disjoint from BOTH training sets.

    python build_v3.py --chembl37 work/chembl_37/chembl_37.db \
                       --chembl34 work/chembl_34/chembl_34.db \
                       --train ../../../Input/input.csv --workers 16

Three gates, each aimed at a specific way the v2 benchmark turned out to be
uninformative.

1. TEMPORAL, for Boltz-2. Only Kd measurements from documents dated >= --min-year,
   and only ligands whose InChIKey skeleton does not appear anywhere in ChEMBL
   v34 (2024-04-15), the release Boltz-2's affinity module drew on. This is the
   'unseen' stratum of the v2 contamination audit, applied as a gate instead of
   a label. Boltz's training set was never published, so this bounds
   contamination rather than eliminating it - say so when reporting.

2. LIGAND, for Savant. ECFP4 Tanimoto < --max-tanimoto to every ligand in
   input.csv, plus no shared InChIKey skeleton. Note this now gates against
   input.csv's 991k ligands, not combined_50k_filtered's 32k, so it is far
   stricter than the v2 gate. Only ~12% of typical med-chem ligands clear it.

3. TARGET-FIRST SELECTION, for the metric. v2 selected pairs, which left only
   33 targets carrying >= 8 pairs - too few to report within-target ranking, so
   the headline number ended up dominated by between-target variance. On v2 a
   protein-mean lookup with no ligand information scored Pearson 0.524 / ROC
   0.745, beating both Savant and Boltz-2. v3 selects targets carrying at least
   --min-per-target qualifying pairs with real label spread, so within-target
   Spearman is computable across the whole set.

Proteins are NOT gated. input.csv holds 22,836 sequences covering essentially
all well-measured targets, so requiring novel proteins would leave only sparsely
studied ones - unrepresentative, and short of pairs per target. The target prior
is handled by measuring it: 06_verify reports the protein-mean baseline as a row.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (CENSOR_CAPS_NM, MAX_CHAIN_LEN, MAX_LIGAND_TANIMOTO, MAX_PKD,
                    MIN_CHAIN_LEN, MORGAN_BITS, MORGAN_RADIUS, OUT, WORK, log,
                    sanitize_seq)

KD_SQL = """
SELECT act.standard_value      AS kd_nM,
       act.standard_type       AS standard_type,
       act.standard_relation   AS relation,
       cs.canonical_smiles     AS ligand_smiles,
       cs.standard_inchi_key   AS inchikey,
       seq.sequence            AS target_sequence,
       seq.accession           AS uniprot_id,
       td.pref_name            AS protein_name,
       td.organism             AS organism,
       td.chembl_id            AS chembl_target_id,
       md.chembl_id            AS molecule_chembl_id,
       d.year                  AS doc_year,
       d.chembl_id             AS doc_chembl_id
FROM activities act
JOIN assays a              ON a.assay_id      = act.assay_id
JOIN docs d                ON d.doc_id        = a.doc_id
JOIN target_dictionary td  ON td.tid          = a.tid
JOIN target_components tc  ON tc.tid          = td.tid
JOIN component_sequences seq ON seq.component_id = tc.component_id
JOIN compound_structures cs ON cs.molregno    = act.molregno
JOIN molecule_dictionary md ON md.molregno    = act.molregno
WHERE act.standard_type IN ({types})
  AND act.standard_units = 'nM'
  AND act.standard_value > 0
  AND td.target_type     = 'SINGLE PROTEIN'
  AND a.confidence_score >= 8
  AND d.year >= ?
"""

V34_SKELETONS_SQL = """
SELECT DISTINCT cs.standard_inchi_key
FROM compound_structures cs
WHERE cs.standard_inchi_key IS NOT NULL
"""


# --------------------------------------------------------------------------- #
# fingerprint gate
# --------------------------------------------------------------------------- #

def _max_tanimoto_worker(args):
    """Max Tanimoto of every candidate against one slice of the reference set.

    Each worker fingerprints its own slice, so the 991k reference fingerprints
    are never pickled between processes.
    """
    cand_smiles, ref_slice = args
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    ref = [gen.GetFingerprint(m) for m in
           (Chem.MolFromSmiles(s) for s in ref_slice) if m is not None]
    out = np.zeros(len(cand_smiles), dtype=np.float32)
    if not ref:
        return out
    for i, s in enumerate(cand_smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            out[i] = 1.0                      # unparseable: treat as blocked
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), ref))
    return out


def max_tanimoto_to_training(cand_smiles, train_smiles, workers):
    """Elementwise max over reference slices, so memory stays flat."""
    slices = np.array_split(np.asarray(train_smiles), max(workers, 1))
    log(f"  ligand gate: {len(cand_smiles)} candidates vs {len(train_smiles)} "
        f"training ligands on {workers} worker(s)")
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            parts = pool.map(_max_tanimoto_worker,
                             [(list(cand_smiles), list(s)) for s in slices])
    else:
        parts = [_max_tanimoto_worker((list(cand_smiles), list(s))) for s in slices]
    return np.maximum.reduce(parts)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def load_chembl(db, min_year, types, min_conf):
    """Kd and/or Ki. Both are direct binding constants and are routinely pooled;
    for a competitive inhibitor Ki equals Kd of the inhibitor-enzyme complex. They
    are not interchangeable in general - Ki from an IC50 via Cheng-Prusoff carries
    the substrate concentration with it - so standard_type is kept per row, and
    the mix belongs in the writeup."""
    sql = (KD_SQL.replace("{types}", ",".join(f"'{t}'" for t in types))
                 .replace("a.confidence_score >= 8", f"a.confidence_score >= {int(min_conf)}"))
    log(f"reading {'/'.join(types)} activities from {db} "
        f"(doc year >= {min_year}, confidence >= {min_conf})")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(sql, con, params=(min_year,))
    log(f"  {len(df)} rows, {df.uniprot_id.nunique()} targets, "
        f"{df.inchikey.nunique()} ligands")
    for t, n in df.standard_type.value_counts().items():
        log(f"    {t}: {n}")
    return df


def clean(df):
    n0 = len(df)
    caps = set(CENSOR_CAPS_NM)
    df = df[df.relation.isin(["=", None]) | df.relation.isna()]
    df = df[~df.kd_nM.isin(caps)]
    df["target_sequence"] = df.target_sequence.map(sanitize_seq)
    df = df[df.target_sequence.str.len().between(MIN_CHAIN_LEN, MAX_CHAIN_LEN)]
    df["neg_log10_affinity_M"] = 9.0 - np.log10(df.kd_nM.astype(float))
    df = df[df.neg_log10_affinity_M.between(2.0, MAX_PKD)]
    df["skeleton"] = df.inchikey.str.slice(0, 14)
    ## one measurement per (target, ligand): median of replicates
    df = (df.groupby(["uniprot_id", "skeleton"], as_index=False)
            .agg({"neg_log10_affinity_M": "median", "kd_nM": "median",
                  "ligand_smiles": "first", "inchikey": "first",
                  "target_sequence": "first", "protein_name": "first",
                  "organism": "first", "chembl_target_id": "first",
                  "molecule_chembl_id": "first", "doc_year": "min",
                  "doc_chembl_id": "first", "standard_type": "first"}))
    log(f"  cleaned {n0} -> {len(df)} unique (target, ligand) pairs")
    return df


def gate_chembl34(df, db):
    """Drop ligands whose skeleton appears anywhere in ChEMBL v34."""
    if not db:
        log("  WARNING: no --chembl34, temporal gate is year-only. A 2025 paper "
            "can still report a molecule that was already in v34.")
        return df
    log(f"  reading v34 ligand skeletons from {db}")
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        keys = pd.read_sql_query(V34_SKELETONS_SQL, con)
    seen = set(keys.standard_inchi_key.dropna().str.slice(0, 14))
    log(f"  v34 holds {len(seen)} distinct ligand skeletons")
    keep = ~df.skeleton.isin(seen)
    log(f"  v34 gate: {len(df)} -> {int(keep.sum())} pairs "
        f"({int((~keep).sum())} dropped as seen in v34)")
    return df[keep].reset_index(drop=True)


def gate_training(df, train_csv, max_tan, workers):
    log(f"  reading training ligands from {train_csv}")
    train = pd.read_csv(train_csv, usecols=["smiles"]).smiles.dropna().unique()
    ## skeleton gate first - it is free and removes the exact matches
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    uniq = df.drop_duplicates("skeleton")[["skeleton", "ligand_smiles"]]
    tan = max_tanimoto_to_training(uniq.ligand_smiles.tolist(), train, workers)
    blocked = set(uniq.skeleton[tan >= max_tan])
    df = df.assign(max_train_ligand_tanimoto=df.skeleton.map(
        dict(zip(uniq.skeleton, tan))))
    keep = ~df.skeleton.isin(blocked)
    log(f"  training gate (Tanimoto < {max_tan}): {len(df)} -> {int(keep.sum())} pairs")
    return df[keep].reset_index(drop=True)


def select(df, n_pairs, min_per_target, min_std, seed):
    """Target-first: keep targets that can support within-target ranking."""
    stats = df.groupby("uniprot_id").neg_log10_affinity_M.agg(["count", "std"])
    ok = stats[(stats["count"] >= min_per_target) & (stats["std"] >= min_std)]
    log(f"  targets with >= {min_per_target} pairs and pKd std >= {min_std}: "
        f"{len(ok)} (of {len(stats)})")
    if ok.empty:
        raise SystemExit("no target meets the selection floor - relax "
                         "--min-per-target/--min-target-std, lower --min-year, "
                         "or add Ki to the query")

    rng = np.random.default_rng(seed)
    df = df[df.uniprot_id.isin(ok.index)].copy()
    ## round-robin over targets so no single well-studied target dominates
    picked, pools = [], {t: g.sample(frac=1, random_state=seed).index.tolist()
                         for t, g in df.groupby("uniprot_id")}
    order = list(pools)
    rng.shuffle(order)
    while len(picked) < n_pairs and any(pools.values()):
        for t in order:
            if pools[t] and len(picked) < n_pairs:
                picked.append(pools[t].pop())
    out = df.loc[picked].copy()
    kept = out.groupby("uniprot_id").size()
    out = out[out.uniprot_id.isin(kept[kept >= min_per_target].index)]
    log(f"  selected {len(out)} pairs across {out.uniprot_id.nunique()} targets "
        f"(median {out.groupby('uniprot_id').size().median():.0f} per target)")
    return out.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chembl37", required=True, help="SQLite for the recent release")
    p.add_argument("--chembl34", help="SQLite for v34; without it the gate is year-only")
    p.add_argument("--train", required=True, help="input.csv, for the ligand gate")
    p.add_argument("--min-year", type=int, default=2025)
    p.add_argument("--types", nargs="+", default=["Kd", "Ki"],
                   choices=["Kd", "Ki", "IC50", "EC50"],
                   help="affinity types to pool (default: Kd Ki)")
    p.add_argument("--min-conf", type=int, default=8,
                   help="ChEMBL assay confidence_score floor (8 = direct single protein)")
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--min-per-target", type=int, default=10)
    p.add_argument("--min-target-std", type=float, default=0.5)
    p.add_argument("--max-tanimoto", type=float, default=MAX_LIGAND_TANIMOTO)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(OUT, "savant_benchmark_v3.csv"))
    p.add_argument("--dry-run", action="store_true",
                   help="report the funnel and stop before selection")
    args = p.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    df = load_chembl(args.chembl37, args.min_year, args.types, args.min_conf)
    df = clean(df)
    df = gate_chembl34(df, args.chembl34)
    df = gate_training(df, args.train, args.max_tanimoto, args.workers)

    stats = df.groupby("uniprot_id").neg_log10_affinity_M.agg(["count", "std"])
    log(f"funnel: {len(df)} pairs on {len(stats)} targets survive both gates")

    ## Choose the per-target floor from data. Within-target ranking needs
    ## TARGETS, not pairs: v2 managed 33 targets at >= 8 pairs each. If no row
    ## here clears ~25 targets, widen the source (--types Kd Ki, --min-conf 7)
    ## rather than lowering the floor, which weakens every per-target rho.
    print()
    print(f"  {'pairs/target':>14s} {'targets':>8s} {'pairs':>8s}"
          f"   (pKd std >= {args.min_target_std})")
    for k in (4, 6, 8, 10, 15, 20):
        ok = stats[(stats["count"] >= k) & (stats["std"] >= args.min_target_std)]
        print(f"  {k:>14d} {len(ok):>8d} {int(ok['count'].sum()):>8d}")
    print()
    if args.dry_run:
        return

    out = select(df, args.n_pairs, args.min_per_target, args.min_target_std, args.seed)
    out.insert(0, "benchmark_id", [f"v3_{i:04d}" for i in range(1, len(out) + 1)])

    ## v2's novelty_tier graded protein similarity to the old training set. Every
    ## v3 protein is in input.csv by design, so that axis is constant here and the
    ## meaningful one is chemical distance - which is also the axis both gates act
    ## on. Same column name so the scoring notebook's by-tier cell still works.
    out["novelty_tier"] = pd.cut(
        out.max_train_ligand_tanimoto, [-0.01, 0.25, 0.35, 1.0],
        labels=["distant", "moderate", "near"]).astype(str)
    out["deltaG_kcal_per_mol"] = -1.364 * out.neg_log10_affinity_M
    out["source"] = (f"ChEMBL docs year>={args.min_year}, "
                     f"{'/'.join(args.types)}, conf>={args.min_conf}")
    out.to_csv(args.out, index=False)
    log(f"wrote {args.out}")

    ## the numbers to quote when describing the set
    print(f"\n  pairs                 {len(out)}")
    print(f"  targets               {out.uniprot_id.nunique()}")
    print(f"  ligands               {out.skeleton.nunique()}")
    print(f"  pKd  mean {out.neg_log10_affinity_M.mean():.2f}  "
          f"std {out.neg_log10_affinity_M.std():.2f}  "
          f"range {out.neg_log10_affinity_M.min():.2f}-{out.neg_log10_affinity_M.max():.2f}")
    print(f"  max Tanimoto to input.csv: median "
          f"{out.max_train_ligand_tanimoto.median():.3f}, "
          f"max {out.max_train_ligand_tanimoto.max():.3f}")
    print(f"  doc years             {out.doc_year.min():.0f}-{out.doc_year.max():.0f}")
    print(f"  affinity types        {out.standard_type.value_counts().to_dict()}")
    print(f"  novelty_tier          {out.novelty_tier.value_counts().to_dict()}")
    print(f"  pairs per target      min {out.groupby('uniprot_id').size().min()}, "
          f"median {out.groupby('uniprot_id').size().median():.0f}, "
          f"max {out.groupby('uniprot_id').size().max()}")


if __name__ == "__main__":
    main()
