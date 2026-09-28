"""The unified measurement schema and the label policy.

PLAN.md Section 0.4 rule 1 and Section 5.2: the target is pKd, trained on Kd and Ki
ONLY. IC50/EC50 never enter as labels, auxiliary heads, features, sample weights,
filters, or model-selection criteria. This module is the single place that decides
what a measurement type is, so there is exactly one implementation to audit and
`tests/test_labels.py` has one thing to assert against.

The previous sprint learned this the expensive way. Its merge step wrote

    if kd_ki_only and t:

so a source with no type column produced an empty `t`, the filter never ran, and
1.8M untyped rows entered the corpus tagged "unknown". A source whose type cannot
be established is now REFUSED (see `normalize_type` / `require_typed`), and the
only way past that is an explicit, recorded `--assume-type source=Kd`.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

## One row per measurement (PLAN.md Section 5.1).
COLUMNS = [
    "measurement_id", "source_db", "source_record_id", "doc_id",
    "uniprot", "pdb_id", "sequence", "sequence_hash", "is_mutant", "mutation_string",
    "ligand_smiles_raw", "ligand_smiles_std", "inchikey_full", "inchikey_flat",
    "type", "relation", "value_nM", "pK", "temperature", "pH",
    "assay_id", "structure_path", "structure_source", "type_source",
]

ALLOWED_TYPES = ("Kd", "Ki")

## `relation` ALWAYS describes the stored `pK` label, never the concentration.
##
## This has to be stated once and obeyed everywhere, because pK = -log10(Kd) is
## monotonically DECREASING: "Kd > 10 uM" (weak binding) is "pK < 5". A source that
## reports its relation on the concentration - ChEMBL's standard_relation,
## BindingDB's ">10000" strings, Papyrus's relation column, PDBbind's "Kd<10uM" -
## must therefore FLIP it on the way in, via `flip_relation`.
##
## Getting this wrong is invisible: the row count is right, the label is right, and
## only the direction of the one-sided loss in train/loop.py:censored_hinge is
## backwards, so the model is actively taught the opposite bound for those pairs.
ALLOWED_RELATIONS = ("=", ">", "<")
_FLIP = {">": "<", "<": ">", "=": "=", ">=": "<", "<=": ">", "~": "="}


def flip_relation(s: pd.Series) -> pd.Series:
    """Concentration-space relation -> pK-space relation. See ALLOWED_RELATIONS."""
    out = s.fillna("=").astype(str).str.strip().map(_FLIP)
    return out.where(out.isin(ALLOWED_RELATIONS), other="=")

## Everything that is NOT an equilibrium dissociation constant. Listed explicitly so
## a new spelling shows up as "unrecognised" rather than slipping into Kd.
FORBIDDEN_TYPES = (
    "IC50", "EC50", "AC50", "ID50", "ED50", "GI50", "CC50", "LD50", "MIC",
    "POTENCY", "INHIBITION", "ACTIVITY", "PIC50", "PEC50", "KM", "KCAT", "KB",
    "KINACT", "T12", "RESIDENCETIME", "DELTATM", "PA2",
)

_KD = {"KD", "KDNM", "KDUM", "PKD", "KDAPP", "KDAPPARENT", "DISSOCIATIONCONSTANT"}
_KI = {"KI", "KINM", "KIUM", "PKI", "KIAPP", "KIAPPARENT", "INHIBITIONCONSTANT"}

_WS = re.compile(r"[\s_\-()/]+")


def _canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(_WS, "", regex=True)


def normalize_type(s: pd.Series) -> pd.Series:
    """Map raw assay-type strings onto {'Kd', 'Ki', ''}.

    '' means "not an allowed type", which covers both a forbidden type (IC50) and an
    unrecognised one. Callers must treat '' as a rejection, never as a default.
    """
    up = _canon(s)
    out = pd.Series("", index=s.index, dtype=object)
    out[up.isin(_KD)] = "Kd"
    out[up.isin(_KI)] = "Ki"
    return out


def classify_rejections(s: pd.Series) -> pd.Series:
    """Why each non-Kd/Ki row was dropped - for reports/dropped_types.csv."""
    up = _canon(s)
    why = pd.Series("unrecognised", index=s.index, dtype=object)
    why[up.isin(_KD | _KI)] = "kept"
    why[up.isin(set(FORBIDDEN_TYPES))] = "forbidden(IC50/EC50-family)"
    why[up.isin(("", "NAN", "NONE", "NA"))] = "missing"
    return why


def require_typed(df: pd.DataFrame, source: str, assumed: str | None = None) -> pd.Series:
    """Resolve the type column for a source, or refuse the source outright.

    `assumed` is the escape hatch for a legacy table with no type column (the old
    `input.csv`). It must be passed explicitly, it must be Kd or Ki, and it is
    recorded per row in `type_source` so the corpus can always be re-filtered.
    """
    for col in ("type", "standard_type", "activity_type", "assay_type", "measure"):
        if col in df.columns and df[col].notna().any():
            t = normalize_type(df[col])
            t.attrs["type_source"] = "column"
            return t
    if assumed is None:
        raise SystemExit(
            f"source {source!r} has no recognisable assay-type column "
            f"(looked for type/standard_type/activity_type/assay_type/measure).\n\n"
            "Refusing it rather than guessing. PLAN.md Section 0.4 forbids IC50 in any\n"
            "form, and an untyped table cannot be shown to be free of it - the previous\n"
            "sprint admitted 1.8M untyped rows exactly this way.\n\n"
            "If you know every row in this file is one type, say so explicitly:\n"
            f"    --assume-type {source}=Kd\n"
            "It will be recorded per row as type_source='declared'.")
    if assumed not in ALLOWED_TYPES:
        raise SystemExit(f"--assume-type {source}={assumed} is not one of {ALLOWED_TYPES}")
    t = pd.Series(assumed, index=df.index, dtype=object)
    t.attrs["type_source"] = "declared"
    return t


def to_pK(value_nM) -> np.ndarray:
    """pK = -log10(value in M) = 9 - log10(value in nM)."""
    v = np.asarray(value_nM, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v) & (v > 0)
    out[ok] = 9.0 - np.log10(v[ok])
    return out


def from_pK(pk) -> np.ndarray:
    return 10.0 ** (9.0 - np.asarray(pk, dtype=float))


def assert_label_policy(df: pd.DataFrame, where: str = "corpus") -> None:
    """Hard gate. Called at ingestion, before every training run, and in CI."""
    if "type" not in df.columns:
        raise AssertionError(f"{where}: no `type` column - the label policy cannot be checked")
    bad = sorted(set(df["type"].dropna().astype(str)) - set(ALLOWED_TYPES))
    if bad:
        counts = df["type"].value_counts().to_dict()
        raise AssertionError(
            f"{where}: {len(bad)} disallowed measurement type(s) present: {bad}\n"
            f"  counts: {counts}\n"
            f"  PLAN.md Section 0.4: only {ALLOWED_TYPES} may ever enter this project.")
    if "relation" in df.columns:
        badrel = sorted(set(df["relation"].dropna().astype(str)) - set(ALLOWED_RELATIONS))
        if badrel:
            raise AssertionError(f"{where}: unknown relation(s) {badrel}")
    ## A column whose NAME leaks an excluded assay is a design error even if unused:
    ## it means something upstream carried the value along and a later join could
    ## read it. PLAN.md Section 6.5 asks the test to check exactly this.
    leak = [c for c in df.columns
            if any(f in _WS.sub("", c.upper()) for f in ("IC50", "EC50", "AC50", "GI50"))]
    if leak:
        raise AssertionError(
            f"{where}: column(s) {leak} carry an excluded assay type. Drop them at the\n"
            "loader; PLAN.md forbids IC50/EC50 as labels, features AND filters.")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in COLUMNS})


def conform(df: pd.DataFrame, source_db: str) -> pd.DataFrame:
    """Add missing schema columns, order them, and stamp the source."""
    d = df.copy()
    d["source_db"] = source_db
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    if d["pK"].isna().all() and d["value_nM"].notna().any():
        d["pK"] = to_pK(d["value_nM"])
    if d["value_nM"].isna().all() and d["pK"].notna().any():
        d["value_nM"] = from_pK(d["pK"])
    d["relation"] = d["relation"].fillna("=")
    ## Identifier columns are STRINGS, always. ChEMBL emits an int activity_id,
    ## IUPHAR "9141@P16050", PDBbind "3zzf" - concatenating them gives an object
    ## column that pyarrow refuses to write ("Could not convert '9141@P16050' with
    ## type str: tried to convert to int64"), and the failure lands after every
    ## loader has already run.
    for c in ("source_record_id", "doc_id", "assay_id", "pdb_id", "uniprot"):
        d[c] = d[c].astype(object).where(d[c].notna(), None)
        d[c] = d[c].map(lambda v: None if v is None else str(v))
    return d[COLUMNS]
