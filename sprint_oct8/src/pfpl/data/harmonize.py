"""Cross-source dedupe and aggregation (PLAN.md Section 5.2).

ChEMBL, BindingDB, PDBbind and Papyrus overlap heavily - they are largely the same
literature, re-curated. Double counting is not merely wasteful: the same
measurement on both sides of a split is leakage, and the duplicated rows inflate
every confidence interval. Section 5.2 therefore dedupes at the MEASUREMENT level
(doc_id, uniprot, inchikey, value) before aggregating to one label per pair.

Two decisions worth being explicit about:

* **Dedupe on InChIKey, never on the SMILES string.** The same molecule is spelled
  differently by every source; the previous sprint kept both copies of thousands of
  ligands for one run by keying on the raw string.
* **Aggregate by median, and keep the spread.** Section 4.4 notes that 6-8% of
  curated matched pairs disagree by more than one log unit, and Section 9 wants
  metrics stratified by that. `pK_spread` and `n_measurements` survive into the
  pair table so a high-variance pair can be down-weighted or reported separately.

The Ki/Kd cross-type table (Section 5.2) is written out as well: it is the
project's own measurement of whether pooling Ki with Kd is defensible, which is
otherwise an assumption carried through the whole plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S


def dedupe_measurements(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows that are the same measurement re-curated by a second source.

    Keyed on (doc_id, uniprot|sequence_hash, inchikey_full, type, rounded value).
    Rounding to 2 decimal places in pK absorbs unit-conversion jitter between
    sources without merging genuinely different measurements (0.01 pK is 2% in
    concentration; inter-lab scatter is ~0.5 pK).
    """
    d = df.copy()
    prot = d["uniprot"].fillna("").astype(str)
    prot = prot.where(prot != "", d["sequence_hash"].fillna("").astype(str))
    key = pd.Series(list(zip(
        d["doc_id"].fillna("").astype(str), prot,
        d["inchikey_full"].fillna("").astype(str), d["type"].astype(str),
        np.round(pd.to_numeric(d["pK"], errors="coerce"), 2))), index=d.index)
    ## Rows with no document id cannot be shown to be duplicates of each other, so
    ## they are exempt - dropping them would silently delete whole sources.
    has_doc = d["doc_id"].notna() & (d["doc_id"].astype(str).str.len() > 0)
    ## Source preference when a duplicate IS found: structure-linked first.
    order = {"pdbbind": 0, "chembl": 1, "papyrus": 2, "bindingdb": 3}
    d["_pref"] = d["source_db"].map(order).fillna(9)
    d["_key"] = key
    dup = has_doc & d.duplicated(subset=["_key"], keep=False)
    keep = ~dup | (d.sort_values("_pref").groupby("_key", sort=False).cumcount() == 0) \
        .reindex(d.index).fillna(True)
    kept = d[keep].drop(columns=["_pref", "_key"])
    if verbose:
        print(f"  cross-source dedupe: {len(d):,} -> {len(kept):,} "
              f"({len(d) - len(kept):,} duplicate measurements removed)")
        print(f"    by source: {kept['source_db'].value_counts().to_dict()}")
    return kept


def cross_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs measured as BOTH Ki and Kd - the evidence for or against pooling.

    PLAN.md pools Ki with Kd by default and lists "Ki and Kd pooling adds assay
    noise" as a risk mitigated by a Kd-only ablation. This quantifies the risk on
    this project's own data instead of assuming it.
    """
    g = (df.dropna(subset=["inchikey_full", "sequence_hash"])
           .groupby(["sequence_hash", "inchikey_full", "type"])["pK"]
           .median().unstack("type"))
    both = g.dropna(subset=[c for c in ("Ki", "Kd") if c in g.columns])
    if both.empty or "Ki" not in both or "Kd" not in both:
        return pd.DataFrame()
    both = both.assign(delta=both["Ki"] - both["Kd"])
    print(f"  Ki vs Kd on {len(both):,} identical pairs: "
          f"median delta {both.delta.median():+.3f}, "
          f"MAD {float((both.delta - both.delta.median()).abs().median()):.3f}, "
          f"|delta|>1 in {float((both.delta.abs() > 1).mean()):.1%}")
    return both.reset_index()


def aggregate_pairs(df: pd.DataFrame, label_types=("Kd", "Ki"),
                    verbose: bool = True) -> pd.DataFrame:
    """One row per (protein, ligand): median pK, with the spread retained.

    Censored rows are aggregated separately and only used when a pair has NO exact
    measurement - a ">10 uM" alongside a measured 40 nM is not information, it is a
    different assay having failed.
    """
    d = df[df["type"].isin(label_types)].copy()
    if len(d) < len(df) and verbose:
        print(f"  label_types={list(label_types)}: {len(df):,} -> {len(d):,}")
    d = d.dropna(subset=["sequence_hash", "inchikey_full", "pK"])

    keys = ["sequence_hash", "inchikey_full"]
    exact = d[d["relation"] == "="]
    cens = d[d["relation"] != "="]

    def agg(g: pd.DataFrame, censored: bool) -> pd.DataFrame:
        a = g.groupby(keys).agg(
            pK=("pK", "median"),
            pK_spread=("pK", lambda x: float(np.ptp(x)) if len(x) > 1 else 0.0),
            n_measurements=("pK", "size"),
            type=("type", lambda x: x.mode().iat[0]),
            n_types=("type", "nunique"),
            sequence=("sequence", "first"),
            ligand_smiles_std=("ligand_smiles_std", "first"),
            inchikey_flat=("inchikey_flat", "first"),
            uniprot=("uniprot", "first"),
            pdb_id=("pdb_id", "first"),
            structure_path=("structure_path", "first"),
            structure_source=("structure_source", "first"),
            source_db=("source_db", lambda x: ",".join(sorted(set(x)))),
            doc_id=("doc_id", "first"),
        ).reset_index()
        a["censored"] = censored
        a["relation"] = g.groupby(keys)["relation"].first().to_numpy() if censored else "="
        return a

    ex = agg(exact, False) if len(exact) else pd.DataFrame(columns=keys)
    cn = agg(cens, True) if len(cens) else pd.DataFrame(columns=keys)
    if len(cn) and len(ex):
        cn = cn[~cn.set_index(keys).index.isin(ex.set_index(keys).index)]
    out = pd.concat([ex, cn], ignore_index=True) if len(cn) else ex

    ## Section 5.2 source weights: crystal-linked > literature > modelled.
    w = {"crystal": 1.0, "modeled": 0.5, "cofolded": 0.5}
    out["weight"] = out["structure_source"].map(w).fillna(0.8)
    ## A pair whose replicates disagree by >1 log unit is downweighted rather than
    ## dropped; Section 4.4 puts that at 6-8% of curated data and the heteroscedastic
    ## loss is meant to absorb it.
    out.loc[out["pK_spread"] > 1.0, "weight"] *= 0.5
    out["flag_high_spread"] = out["pK_spread"] > 1.0

    if verbose:
        print(f"  aggregated to {len(out):,} unique (protein, ligand) pairs "
              f"from {len(d):,} measurements")
        print(f"    proteins {out.sequence_hash.nunique():,}  "
              f"ligands {out.inchikey_full.nunique():,}  "
              f"censored {int(out.censored.sum()):,}  "
              f"high-spread {int(out.flag_high_spread.sum()):,} "
              f"({out.flag_high_spread.mean():.1%})")
        print(f"    replicated pairs (n>1): "
              f"{int((out.n_measurements > 1).sum()):,}; "
              f"median spread among them "
              f"{out.loc[out.n_measurements > 1, 'pK_spread'].median():.2f} pK")
    S.assert_label_policy(out, "aggregated pairs")
    return out


def noise_ceiling(pairs: pd.DataFrame) -> dict:
    """Quantify this corpus's own reproducibility (Appendix A, last row).

    Section 4.4 quotes ~0.5 log inter-lab scatter and 6-8% of pairs differing by >1
    unit from the literature. Measuring it here turns a borrowed number into a
    property of the actual training data, and it is the only honest way to say
    whether an RMSE of 1.3 is good.
    """
    rep = pairs[pairs.n_measurements > 1]
    if rep.empty:
        return {"n_replicated": 0}
    return {
        "n_replicated": int(len(rep)),
        "frac_of_corpus": float(len(rep) / len(pairs)),
        "median_spread_pK": float(rep.pK_spread.median()),
        "mean_spread_pK": float(rep.pK_spread.mean()),
        "frac_spread_gt_1": float((rep.pK_spread > 1).mean()),
        "implied_sd_pK": float(rep.pK_spread.median() / 1.35),   # range -> sd, n=2
        "note": "an RMSE below the implied sd is fitting noise, not signal",
    }
