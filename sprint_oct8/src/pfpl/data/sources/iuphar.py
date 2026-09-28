"""IUPHAR / Guide to PHARMACOLOGY Kd/Ki (PLAN.md Section 4.1).

Small and hand-curated: ~9.5k Kd/Ki interactions against ChEMBL's hundreds of
thousands. It earns its place not by volume but because ChEMBL, BindingDB and
Papyrus are all literature scrapes of largely the same primary sources, whereas
IUPHAR is expert-curated end to end. It is the closest thing to an independent
check on the others, and it over-represents GPCRs, ion channels and transporters,
which the structure-linked sources cover badly.

Two things about the download that a naive read gets wrong, both verified against
the live 2026.3 release:

1. **The version banner is a QUOTED CSV field** - `"# GtoPdb Version: ..."` - so
   `comment="#"` never sees a leading `#` and pandas takes the banner as the
   header, yielding a one-column frame. It needs `skiprows=1`.
2. **interactions.csv has no SMILES**, only `Ligand ID`. The structures live in a
   separate `ligands.csv`, so both files are required and joined here.

Affinities are already negative logs (`Affinity Median` is pKi/pKd/pIC50/...), so
there is no unit conversion to get wrong - but equally no protection against
silently reading a pIC50, which is why the type goes through the same
`normalize_type` as every other source. In the 2026.3 release that rejects 9,805
pIC50, 2,674 pEC50, 145 pKB, 116 pA2 and 2,643 untyped rows, and keeps 7,240 pKi
and 2,311 pKd.

Sequences are in neither file - rows carry a UniProt accession only - so they are
resolved through the shared cache that `phase1a_acquire.py --uniprot` builds. That
cache serves ChEMBL too, which has exactly the same gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...paths import RAW
from .. import schema as S
from . import register

## Real column names in the released CSVs (2026.3), not the API's snake_case.
C_TYPE = "Affinity Units"          # "pKi", "pKd", "pIC50", "-", ...
C_VALUE = "Affinity Median"
C_REL = "Original Affinity Relation"
C_UNIPROT = "Target UniProt ID"
C_SPECIES = "Target Species"
C_PMID = "PubMed ID"
C_LIGID = "Ligand ID"
L_LIGID = "Ligand ID"
L_SMILES = "SMILES"


def _read(p: Path) -> pd.DataFrame:
    ## skiprows=1, not comment="#": see the module docstring.
    return pd.read_csv(p, skiprows=1, low_memory=False)


@register("iuphar")
def load(path: str | None = None, ligands: str | None = None,
         species: str | None = "Human", **kw) -> pd.DataFrame:
    base = RAW / "iuphar"
    p = Path(path) if path else base / "interactions.csv"
    lp = Path(ligands) if ligands else base / "ligands.csv"
    if not p.exists() or not lp.exists():
        missing = [str(x) for x in (p, lp) if not x.exists()]
        raise SystemExit(
            f"IUPHAR not acquired - missing {missing}.\n"
            "    python scripts/phase1a_acquire.py --iuphar\n"
            "  (both interactions.csv and ligands.csv are needed: the interactions\n"
            "   file carries only a Ligand ID, and the structures are in the other.)\n"
            "  Or disable it: -o data.sources.iuphar.enabled=false")

    df = _read(p)
    lig = _read(lp)
    print(f"  iuphar: {len(df):,} interactions, {len(lig):,} ligands")

    missing = [c for c in (C_TYPE, C_VALUE, C_LIGID) if c not in df.columns]
    if missing or L_SMILES not in lig.columns:
        raise SystemExit(
            f"iuphar: expected column(s) {missing or [L_SMILES]} absent - the release "
            f"schema changed.\n  interactions: {list(df.columns)[:12]}\n"
            f"  ligands: {list(lig.columns)[:12]}")

    if species and C_SPECIES in df.columns:
        before = len(df)
        df = df[df[C_SPECIES].astype(str).str.strip().str.lower() == species.lower()]
        print(f"    species={species}: {before:,} -> {len(df):,}")

    ## "pKi"/"pKd" -> "Ki"/"Kd". Everything else (pIC50, pEC50, pKB, pA2, "-") is
    ## rejected by normalize_type and reported by name.
    raw_type = df[C_TYPE].astype(str).str.strip().str.replace(
        r"^p(?=K[idID]$)", "", regex=True)
    t = S.normalize_type(raw_type)
    counts = raw_type[~t.isin(S.ALLOWED_TYPES)].value_counts().head(6).to_dict()
    print(f"    kept types {t[t.isin(S.ALLOWED_TYPES)].value_counts().to_dict()}; "
          f"rejected {counts}")
    keep = t.isin(S.ALLOWED_TYPES)
    d = df[keep].copy()

    ## `Affinity Median` is already a negative log. A range is written "7.5 - 8.2";
    ## the median column is normally a single value, but split defensively.
    pk = pd.to_numeric(
        d[C_VALUE].astype(str).str.split("-").str[0].str.strip(), errors="coerce")

    rel = (d[C_REL].fillna("=").astype(str).str.strip()
           if C_REL in d.columns else pd.Series("=", index=d.index))
    ## `Original Affinity Relation` is on the original CONCENTRATION, so it flips
    ## into the pK space the schema stores. See schema.ALLOWED_RELATIONS.
    rel = S.flip_relation(rel)

    smiles = d[C_LIGID].map(
        lig.drop_duplicates(L_LIGID).set_index(L_LIGID)[L_SMILES])

    ## Sequences come from the shared UniProt cache Phase 1a builds - neither IUPHAR
    ## file carries one. Without it every row here fails Phase 1's standardisation.
    from . import uniprot_sequences
    seqs = uniprot_sequences()
    acc = (d[C_UNIPROT].astype(str).str.split("|").str[0].str.strip()
           if C_UNIPROT in d else pd.Series(pd.NA, index=d.index))
    sequence = acc.map(lambda a: seqs.get(str(a).split("-")[0]))
    if not seqs:
        print("    no UniProt sequence cache; these rows will be dropped by Phase 1\n"
              "      python scripts/phase1a_acquire.py --uniprot")

    out = pd.DataFrame({
        "type": t[keep].to_numpy(), "type_source": "column",
        "source_record_id": d[C_LIGID].astype(str) + "@" +
        d.get(C_UNIPROT, pd.Series("", index=d.index)).astype(str),
        "ligand_smiles_raw": smiles,
        "uniprot": acc.to_numpy(),
        "sequence": sequence.to_numpy(),
        ## A row can cite several papers as "856582|22168767", so this is a string
        ## split, not an int cast. The first PMID is the dedupe key harmonize uses;
        ## the alternative (exploding to one row per citation) would inflate the
        ## measurement count for no gain.
        "doc_id": ("PMID:" + d[C_PMID].astype(str).str.split("|").str[0].str.strip())
        if C_PMID in d else pd.NA,
        "pK": pk.to_numpy(),
        "relation": rel.to_numpy(),
    })
    before = len(out)
    out = out[out["ligand_smiles_raw"].notna()]
    if len(out) < before:
        print(f"    dropped {before - len(out):,} rows with no SMILES in ligands.csv")

    out = out[out["relation"].isin(S.ALLOWED_RELATIONS)]
    out = S.conform(out, "iuphar")
    out = out[out["pK"].notna() & (out["pK"] > 0) & (out["pK"] < 14)]
    S.assert_label_policy(out, "iuphar")
    print(f"    kept {len(out):,} rows  {out['type'].value_counts().to_dict()}  "
          f"relations {out['relation'].value_counts().to_dict()}")
    n_seq = int(out["sequence"].notna().sum())
    print(f"    sequences resolved from the UniProt cache: {n_seq:,}/{len(out):,}")
    return out
