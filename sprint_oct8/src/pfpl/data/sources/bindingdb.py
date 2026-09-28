"""BindingDB Kd/Ki (PLAN.md Section 4.1).

BindingDB ships one wide TSV with a separate column per measurement type:

    Ki (nM)   Kd (nM)   IC50 (nM)   EC50 (nM)   kon   koff   ...

Only the Ki and Kd columns are ever read. The IC50/EC50 columns are dropped at the
`usecols` level, so the excluded values never enter a DataFrame at all - which is
what makes `schema.assert_label_policy`'s column-name check meaningful rather than
decorative.

Values carry inline relations (">10000", "<0.5"); they are parsed into the schema's
`relation` + `value_nM` rather than coerced to NaN.

The full download is ~1.5 GB compressed. Section 4.4 expects several hundred
thousand Kd/Ki pairs after dedupe against ChEMBL, Ki-dominated.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from . import register

## Column names, checked against BindingDB_All_202609_tsv.zip (640 columns). The
## per-chain columns gained a chain-number suffix (" 1", " 2", ...) at some point;
## each constant lists the current name first and the older spelling after it, and
## `_resolve` picks whichever the dump actually has. With the single-chain filter
## on, chain 1 IS the target.
C_SMILES = ("Ligand SMILES",)
C_SEQ = ("BindingDB Target Chain Sequence 1", "BindingDB Target Chain Sequence")
C_UNIPROT = ("UniProt (SwissProt) Primary ID of Target Chain 1",
             "UniProt (SwissProt) Primary ID of Target Chain")
C_INCHIKEY = ("Ligand InChI Key",)
C_RSID = ("BindingDB Reactant_set_id",)
C_PMID = ("PMID",)
## The complex's own structure(s) first; the chain's PDB IDs (any structure of that
## protein) only as a fallback for older dumps.
C_PDB = ("PDB ID(s) for Ligand-Target Complex", "PDB ID(s) of Target Chain 1",
         "PDB ID(s) of Target Chain")
C_CHAINS = ("Number of Protein Chains in Target (>1 implies a multichain complex)",)
C_KI, C_KD = ("Ki (nM)",), ("Kd (nM)",)
C_TEMP, C_PH = ("Temp (C)", "Temperature (C)"), ("pH",)


def _resolve(cols, aliases) -> str | None:
    return next((a for a in aliases if a in cols), None)

_NUM = re.compile(r"^\s*([<>~]?=?)\s*([0-9.eE+-]+)\s*$")


def parse_value(s):
    """'>10000' -> ('>', 10000.0). Unparseable -> ('', nan)."""
    if isinstance(s, (int, float)) and np.isfinite(s):
        return "=", float(s)
    m = _NUM.match(str(s))
    if not m:
        return "", np.nan
    rel = m.group(1).replace("=", "") or "="
    try:
        return (rel if rel in S.ALLOWED_RELATIONS else "="), float(m.group(2))
    except ValueError:
        return "", np.nan


def _open(path: Path):
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        name = [n for n in z.namelist() if n.endswith(".tsv")][0]
        return z.open(name)
    return open(path, "rb")


@register("bindingdb")
def load(path: str | None = None, nrows: int | None = None,
         single_chain_only: bool = True, **kw) -> pd.DataFrame:
    ## Local files only. scripts/phase1a_acquire.py is the single place that touches
    ## the network, so a 200 GB-RAM harmonisation job cannot stall on a 1.5 GB
    ## download, and a flaky link costs a retry rather than an allocation.
    from ...paths import RAW
    ## Newest complete dump; never a .part or a checksum-failed .zip.bad.
    p = Path(path).expanduser() if path else next(
        iter(sorted((RAW / "bindingdb").glob("BindingDB_All_*_tsv.zip"), reverse=True)),
        None)
    if p is None or not Path(p).exists():
        raise SystemExit(
            f"no BindingDB dump under {RAW / 'bindingdb'} - it has not been acquired.\n"
            "    python scripts/phase1a_acquire.py --bindingdb\n"
            "  Or disable it: -o data.sources.bindingdb.enabled=false")
    p = Path(p)

    ## usecols is the label-policy boundary: IC50/EC50 are never read into memory.
    with _open(p) as fh:
        head = pd.read_csv(fh, sep="\t", nrows=1, low_memory=False,
                           on_bad_lines="skip")
    col = {k: _resolve(head.columns, a) for k, a in dict(
        smiles=C_SMILES, seq=C_SEQ, uniprot=C_UNIPROT, inchikey=C_INCHIKEY,
        rsid=C_RSID, pmid=C_PMID, pdb=C_PDB, chains=C_CHAINS, ki=C_KI, kd=C_KD,
        temp=C_TEMP, ph=C_PH).items()}
    missing = [k for k in ("smiles", "seq", "ki", "kd") if col[k] is None]
    if missing:
        raise SystemExit(f"bindingdb: expected column(s) {missing} absent; the dump "
                         "schema changed. Columns seen: "
                         f"{[c for c in head.columns if 'Chain' in c or 'nM' in c][:20]}")
    with _open(p) as fh:
        df = pd.read_csv(fh, sep="\t", usecols=[c for c in col.values() if c],
                         nrows=nrows, low_memory=False, on_bad_lines="skip",
                         dtype=str)
    get = lambda d, k: d[col[k]] if col[k] else pd.Series(pd.NA, index=d.index)  # noqa: E731
    print(f"  bindingdb: {len(df):,} rows (Ki/Kd columns only)")

    if single_chain_only and col["chains"]:
        n = pd.to_numeric(df[col["chains"]], errors="coerce")
        before = len(df)
        df = df[(n == 1) | n.isna()]
        print(f"    single-chain filter: {before:,} -> {len(df):,}")

    frames = []
    for key, typ in (("ki", "Ki"), ("kd", "Kd")):
        sub = df[df[col[key]].notna() & (df[col[key]].str.strip() != "")].copy()
        if sub.empty:
            continue
        parsed = [parse_value(v) for v in sub[col[key]]]
        out = pd.DataFrame({
            "type": typ, "type_source": "column",
            "ligand_smiles_raw": sub[col["smiles"]], "sequence": sub[col["seq"]],
            "inchikey_full": get(sub, "inchikey"),
            "uniprot": get(sub, "uniprot"), "doc_id": get(sub, "pmid"),
            "pdb_id": get(sub, "pdb"),
            "source_record_id": get(sub, "rsid"),
            ## Temperature is often "25 C" / "298.15 K"; keep the number only.
            "temperature": pd.to_numeric(get(sub, "temp").astype(str).str.extract(
                r"([-\d.]+)")[0], errors="coerce"),
            "pH": pd.to_numeric(get(sub, "ph"), errors="coerce"),
            ## parse_value reads ">10000" off a nM column - a CONCENTRATION
            ## bound - so it flips into pK space.
            "relation": S.flip_relation(pd.Series([r for r, _ in parsed],
                                                  index=sub.index)),
            "value_nM": [v for _, v in parsed],
        }, index=sub.index)
        frames.append(out)
    out = pd.concat(frames, ignore_index=True) if frames else S.empty_frame()
    out = out[out["relation"].isin(S.ALLOWED_RELATIONS)]
    if out["source_record_id"].isna().all():
        out["source_record_id"] = np.arange(len(out))
    out = S.conform(out, "bindingdb")
    out = out[out["pK"].notna() & (out["pK"] > 0) & (out["pK"] < 14)]
    S.assert_label_policy(out, "bindingdb")
    print(f"    kept {len(out):,} rows  {out['type'].value_counts().to_dict()}")
    return out
