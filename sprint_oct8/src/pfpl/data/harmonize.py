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

import re

import numpy as np
import pandas as pd

from . import schema as S

## Preference when a duplicate IS found: structure-linked first, then the sources
## that carry per-measurement rows, then the aggregating ones.
SOURCE_PREF = {"pdbbind": 0, "chembl": 1, "iuphar": 2, "bindingdb": 3, "papyrus": 4}

_PMID = re.compile(r"(?:PMID[:\s]*)?\b(\d{5,9})(?:\.0)?\b", re.I)

## UniProt's own accession grammar (www.uniprot.org/help/accession_numbers).
## Worth copying exactly: a hand-rolled "[A-NR-Z0-9][A-Z0-9]{5,9}" excludes O, P
## and Q, which silently drops P00533 - i.e. most of the corpus.
UNIPROT_ACC = r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"


def normalize_doc_id(x) -> str:
    """One document-id vocabulary across sources: `PMID:<n>` where a PubMed id is
    recoverable, the source's own id otherwise.

    Wave 1 keyed dedupe on the raw strings and every source spells them
    differently - ChEMBL `CHEMBL1134862`, BindingDB `19231178` (often `19231178.0`
    after a float round-trip), Papyrus `PMID:19231178;PMID:...`, IUPHAR
    `PMID:19231178` - so no key ever matched across sources and the dedupe step
    removed nothing at all.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return ""
    if s.upper().startswith(("CHEMBL", "PDBBIND:")):
        return s.upper()
    m = _PMID.search(s)
    return f"PMID:{m.group(1)}" if m else s


def normalize_accession(s: pd.Series) -> pd.Series:
    """`  p00533-2 `, `P00533; Q9Y6K9` -> `P00533`. Non-accessions -> "".

    Isoform suffixes and multi-accession lists have to be stripped in ONE place:
    the dedupe key and the protein canonicalisation must agree, or PDBbind's
    `P00533-2` is a different protein from ChEMBL's `P00533` in one step and the
    same protein in the other.
    """
    acc = (s.fillna("").astype(str).str.strip().str.upper()
            .str.split(r"[;,|\s]", regex=True).str[0].str.split("-").str[0])
    return acc.where(acc.str.fullmatch(UNIPROT_ACC, na=False), "")


def _protein_key(d: pd.DataFrame) -> pd.Series:
    prot = normalize_accession(d["uniprot"])
    return prot.where(prot != "", d["sequence_hash"].fillna("").astype(str))


def dedupe_measurements(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows that are the same measurement re-curated by a second source.

    Two passes, both CROSS-source only - two rows from the SAME source are that
    source's own replicates and are never merged here:

    1. document match: (normalised doc_id, protein, InChIKey, type, pK to 0.01).
    2. value match: (protein, InChIKey, type, relation, pK to 0.01) present in more
       than one source -> keep the most-preferred source's row. ChEMBL document ids
       carry no PubMed id to align on, so this is what catches BindingDB's
       imported-from-ChEMBL copies. Two independent labs agreeing to 0.01 pK on the
       same pair is rare; when it happens the pair's median label is unchanged and
       only a replicate count is lost.

    Rounding to 2 decimals in pK absorbs unit-conversion jitter between sources
    (0.01 pK is 2% in concentration; inter-lab scatter is ~0.5 pK).
    """
    d = df.copy()
    n0 = len(d)
    d["doc_id"] = [normalize_doc_id(x) for x in d["doc_id"]]
    d["_pref"] = d["source_db"].map(SOURCE_PREF).fillna(9)
    pk = np.round(pd.to_numeric(d["pK"], errors="coerce"), 2).astype(str)
    ik = d["inchikey_full"].fillna("").astype(str)
    typ = d["type"].astype(str)
    rel = d["relation"].fillna("=").astype(str)
    prot = _protein_key(d)

    ## Sort by preference so `keep="first"` keeps the preferred source's copy.
    d["_key"] = d["doc_id"] + "|" + prot + "|" + ik + "|" + typ + "|" + pk
    d["_vkey"] = prot + "|" + ik + "|" + typ + "|" + rel + "|" + pk
    d = d.sort_values("_pref", kind="stable")

    ## ---- pass 1: same document, different source ------------------------------
    ## A row with no document id cannot be shown to be a duplicate this way; pass 2
    ## still sees it.
    multi = d.groupby("_key")["source_db"].transform("nunique") > 1
    dup1 = (d["doc_id"] != "") & multi & d.duplicated("_key", keep="first")
    d = d[~dup1]
    n1 = len(d)

    ## ---- pass 2: same value, different source ---------------------------------
    dup2 = d["_pref"] > d.groupby("_vkey")["_pref"].transform("min")
    kept = d[~dup2].drop(columns=["_pref", "_key", "_vkey"]).sort_index()

    if verbose:
        print(f"  cross-source dedupe: {n0:,} -> {len(kept):,}  "
              f"(same document {n0 - n1:,}; same value from a less-preferred source "
              f"{n1 - len(kept):,})")
        print(f"    by source: {kept['source_db'].value_counts().to_dict()}")
    return kept


def canonicalize_proteins(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Collapse construct variants of one target onto a single sequence.

    Wave 1 identified a protein by its exact sequence, so one target entered as
    several "proteins" - ChEMBL's canonical UniProt sequence, BindingDB's
    crystallised construct, PDBbind's SEQRES - and the corpus reported 12,884
    proteins over only 4,409 mmseqs clusters with a median of ONE ligand each.
    Whole-cluster splits were unaffected (mmseqs put the copies together), but
    per-target metrics were computed over fragments of the same target: only 556
    of 2,266 test targets had enough ligands to score.

    Within one UniProt accession, every row takes the LONGEST sequence seen for
    that accession - the construct closest to the full protein, and the one whose
    PLM embedding covers the most residues. Rows with no accession are untouched.
    """
    d = df.copy()
    acc = normalize_accession(d["uniprot"])
    has = acc != ""
    if not has.any():
        return d
    d["_acc"] = acc
    n_before = d.loc[has, "sequence_hash"].nunique()

    sub = d[has]
    longest = (sub.assign(_len=sub["sequence"].str.len())
                  .sort_values("_len", ascending=False)
                  .drop_duplicates("_acc")
                  .set_index("_acc")["sequence"])
    new_seq = d["_acc"].map(longest)
    changed = has & new_seq.notna() & (new_seq != d["sequence"])
    d.loc[changed, "sequence"] = new_seq[changed]

    from .standardize import sequence_hash
    if changed.any():
        d.loc[changed, "sequence_hash"] = [sequence_hash(s) for s in
                                           d.loc[changed, "sequence"]]
    n_after = d.loc[has, "sequence_hash"].nunique()
    d = d.drop(columns=["_acc"])
    if verbose:
        print(f"  protein canonicalisation: {n_before:,} -> {n_after:,} distinct "
              f"sequences over {int(has.sum()):,} rows with a UniProt accession "
              f"({int(changed.sum()):,} rows re-pointed to the longest construct)")
    return d


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
