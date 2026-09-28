"""Tables already on disk from the previous sprint, re-read under the new policy.

There is a 633,838-row `merged_kd_ki.csv` on scratch (6,176 proteins, 300,109
ligands, already deduped on InChIKey and already filtered to Kd/Ki). Re-deriving it
would cost a day of the sprint for no new information, so it is a first-class
source here - but it is re-validated rather than trusted: the type column is run
through `schema.normalize_type` again and `assert_label_policy` is applied before
it is returned.

The older `input.csv` (1.8M rows) has NO assay-type column. Section 0.4 forbids
IC50 in any form and an untyped table cannot be shown to be free of it, so it is
refused unless the caller declares the type explicitly:

    -o data.sources.input_csv.assume_type=Kd

which is recorded per row as `type_source='declared'` so the corpus can always be
re-filtered later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from . import register

## Column aliases seen across the project's own CSVs.
_SMILES = ("smiles", "ligand_smiles", "canonical_smiles", "ligand_smiles_raw",
           "SMILES", "Smiles")
_SEQ = ("sequence", "protein_sequence", "seq", "target_sequence", "Sequence")
_VALUE = ("value_nM", "standard_value", "value", "affinity_nM")
_PK = ("pK", "pchembl_value", "pchembl_value_Mean", "neg_log10_affinity_M",
       "pkd", "pKd", "pKi", "label", "y")
_ACC = ("uniprot", "accession", "UniProt", "target_uniprot", "uniprot_id")
_DOC = ("doc_id", "pmid", "PMID", "doi", "DOI", "reference")


def _first(df: pd.DataFrame, names) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


@register("local_csv")
def load(path, name: str = "local_csv", assume_type: str | None = None,
         nrows: int | None = None, **kw) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"local_csv source {name!r}: {p} does not exist")
    df = pd.read_csv(p, nrows=nrows, low_memory=False)
    print(f"  {name}: {len(df):,} rows from {p}")

    smi = _first(df, _SMILES)
    seq = _first(df, _SEQ)
    if smi is None or seq is None:
        raise SystemExit(
            f"{name}: need a SMILES column (one of {_SMILES}) and a sequence column "
            f"(one of {_SEQ}); found {list(df.columns)[:25]}")

    ## Type resolution and the hard refusal both live in schema.require_typed.
    t = S.require_typed(df, name, assume_type)
    rejected = S.classify_rejections(
        df[_first(df, ("type", "standard_type", "activity_type", "assay_type", "measure"))]
        if _first(df, ("type", "standard_type", "activity_type", "assay_type", "measure"))
        else pd.Series("declared:" + str(assume_type), index=df.index))
    keep = t.isin(S.ALLOWED_TYPES)
    if not keep.all():
        print(f"    dropped {int((~keep).sum()):,} rows by type: "
              f"{rejected[~keep].value_counts().to_dict()}")

    d = df[keep].copy()
    out = pd.DataFrame(index=d.index)
    out["ligand_smiles_raw"] = d[smi]
    out["sequence"] = d[seq]
    out["type"] = t[keep].to_numpy()
    out["type_source"] = t.attrs.get("type_source", "column")
    out["source_record_id"] = (d[_first(d, ("id", "activity_id", "record_id"))]
                               if _first(d, ("id", "activity_id", "record_id"))
                               else np.arange(len(d)))
    for col, names in (("uniprot", _ACC), ("doc_id", _DOC)):
        c = _first(d, names)
        out[col] = d[c] if c else pd.NA
    rel = _first(d, ("relation", "standard_relation"))
    out["relation"] = d[rel].fillna("=") if rel else "="

    vc, pc = _first(d, _VALUE), _first(d, _PK)
    if pc:
        out["pK"] = pd.to_numeric(d[pc], errors="coerce")
    if vc:
        out["value_nM"] = pd.to_numeric(d[vc], errors="coerce")
    if "pK" not in out and "value_nM" not in out:
        raise SystemExit(f"{name}: no affinity column; looked for {_VALUE} and {_PK}")

    out = S.conform(out, name)
    out = out[out["pK"].notna() & np.isfinite(out["pK"].astype(float))]
    ## A pK outside 0-14 is a unit error somewhere upstream, not a measurement.
    out = out[(out["pK"] > 0) & (out["pK"] < 14)]
    ## Section 5.2: censored values keep their relation and get a one-sided loss;
    ## anything with an unparseable relation is dropped rather than read as "=".
    out = out[out["relation"].isin(S.ALLOWED_RELATIONS)]
    S.assert_label_policy(out, name)
    print(f"    kept {len(out):,} rows  "
          f"({out['type'].value_counts().to_dict()}, "
          f"relations {out['relation'].value_counts().to_dict()})")
    return out
