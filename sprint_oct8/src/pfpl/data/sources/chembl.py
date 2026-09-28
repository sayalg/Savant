"""ChEMBL binding-assay Kd/Ki, read from the Phase 1a cache (PLAN.md Section 4.1).

This loader does NOT touch the network. `scripts/phase1a_acquire.py` fetches and
checkpoints the activity pages; this reads `data/raw/chembl/activities.parquet` and
`targets.json`. The split keeps a four-hour resumable download out of a step that
needs RDKit and 200 GB of RAM.

Filters were applied at the query (see phase1a):

    standard_type in (Kd, Ki)      the label policy
    assay_type = B                 BINDING assays only. A Ki from a functional
                                   assay is a different quantity measured a
                                   different way; pooling them is the same
                                   category error as pooling IC50, and it is the
                                   filter the previous scrape was missing.
    target_confidence_score = 9    single protein, directly assigned
    standard_units = nM            one unit, no conversion guesswork

Applied here, because they need the downloaded rows in hand:

    data_validity_comment is null  drops "outside typical range" and
                                   author-confirmed errors
    single-protein sequence        a row whose target will not resolve to one
                                   sequence is dropped rather than attached to a
                                   complex

`relation` is kept, not filtered: "=" is the main set and ">"/"<" become censored
labels for the one-sided loss (Section 5.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...paths import RAW
from .. import schema as S
from . import register


@register("chembl")
def load(cache_dir: str | None = None, limit: int | None = None,
         require_pchembl: bool = False, **kw) -> pd.DataFrame:
    cache = Path(cache_dir) if cache_dir else RAW / "chembl"
    act_p, tgt_p = cache / "activities.parquet", cache / "targets.json"
    if not act_p.exists():
        ck = cache / "checkpoint.json"
        partial = ""
        if ck.exists():
            c = json.loads(ck.read_text())
            partial = (f"\n  A partial fetch exists ({c.get('offset', 0):,} of "
                       f"{c.get('total', '?')} rows). Re-running resumes it.")
        raise SystemExit(
            f"{act_p} missing - ChEMBL has not been acquired.{partial}\n\n"
            "    python scripts/phase1a_acquire.py --chembl\n\n"
            "  Phase 1a is the only step that touches the network; this loader "
            "reads local files only.")

    df = pd.read_parquet(act_p)
    if limit:
        df = df.head(limit)
    print(f"  chembl: {len(df):,} cached binding-assay activities")

    if "data_validity_comment" in df.columns:
        before = len(df)
        df = df[df["data_validity_comment"].isna()]
        print(f"    data_validity_comment filter: {before:,} -> {len(df):,}")

    ## Belt and braces: the query asked for assay_type=B, but a cache built by an
    ## older run might predate that filter, and a functional Ki would be invisible
    ## downstream.
    if "assay_type" in df.columns:
        before = len(df)
        df = df[df["assay_type"].isna() | (df["assay_type"] == "B")]
        if len(df) < before:
            print(f"    assay_type=B filter: {before:,} -> {len(df):,} "
                  "(cache predates the query filter)")

    targets = json.loads(tgt_p.read_text()) if tgt_p.exists() else {}
    if not targets:
        raise SystemExit(f"{tgt_p} missing - re-run scripts/phase1a_acquire.py --chembl "
                         "to resolve target accessions")

    ## The ChEMBL target endpoint returns an accession but NO sequence (verified
    ## live: target_components[0] has accession / component_description /
    ## component_type / xrefs / synonyms and no residues). Sequences come from the
    ## UniProt cache Phase 1a builds. Reading a `sequence` key off the target
    ## response instead resolves every row to None and silently drops the entire
    ## ChEMBL contribution at Phase 1.
    from . import uniprot_sequences
    seqs = uniprot_sequences()
    if not seqs:
        raise SystemExit(
            "no UniProt sequence cache - ChEMBL targets carry accessions only.\n"
            "    python scripts/phase1a_acquire.py --uniprot")

    uni = df["target_chembl_id"].map(lambda t: (targets.get(t) or {}).get("uniprot"))
    seq = uni.map(lambda a: seqs.get(str(a).split("-")[0]) if pd.notna(a) else None)
    before = len(df)
    ok = seq.notna()
    n_no_acc = int(uni.isna().sum())
    df, seq, uni = df[ok], seq[ok], uni[ok]
    print(f"    sequence resolved via UniProt: {before:,} -> {len(df):,} "
          f"({n_no_acc:,} had no single-protein accession, "
          f"{before - len(df) - n_no_acc:,} accession not in the cache)")

    t = S.normalize_type(df["standard_type"])
    why = S.classify_rejections(df["standard_type"])
    if not t.isin(S.ALLOWED_TYPES).all():
        print(f"    types: {why.value_counts().to_dict()}")
    keep = t.isin(S.ALLOWED_TYPES)
    df, t, seq, uni = df[keep], t[keep], seq[keep], uni[keep]

    out = pd.DataFrame({
        "type": t, "type_source": "column",
        "source_record_id": df["activity_id"],
        "ligand_smiles_raw": df["canonical_smiles"],
        "sequence": seq, "uniprot": uni,
        "doc_id": df["document_chembl_id"],
        "assay_id": df.get("assay_chembl_id"),
        "value_nM": pd.to_numeric(df["standard_value"], errors="coerce"),
        "pK": pd.to_numeric(df.get("pchembl_value"), errors="coerce"),
        ## standard_relation applies to standard_value (a CONCENTRATION in nM);
        ## the schema stores relations in pK space, so it flips.
        "relation": S.flip_relation(df["standard_relation"]),
    })
    ## pchembl_value is null for a fair fraction of rows; derive pK from the value
    ## rather than discarding them. conform() does this, but only when the column
    ## is entirely absent, so do it explicitly.
    need = out["pK"].isna() & out["value_nM"].notna()
    out.loc[need, "pK"] = S.to_pK(out.loc[need, "value_nM"])
    if require_pchembl:
        out = out[out["pK"].notna()]

    out = out[out["relation"].isin(S.ALLOWED_RELATIONS)]
    out = S.conform(out, "chembl")
    out = out[out["pK"].notna() & (out["pK"] > 0) & (out["pK"] < 14)]
    S.assert_label_policy(out, "chembl")
    print(f"    kept {len(out):,} rows  {out['type'].value_counts().to_dict()}  "
          f"relations {out['relation'].value_counts().to_dict()}")
    return out
