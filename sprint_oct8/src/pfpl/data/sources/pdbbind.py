"""PDBbind: crystal complexes with Kd/Ki (PLAN.md Sections 4.1, 4.2).

PDBbind does THREE jobs in this project, and they are easy to conflate:

  1. **Kd/Ki labels** for the affinity corpus, at source weight 1.0 - the highest
     of any source (Section 5.2), because the measurement is tied to a solved
     structure of the very complex being measured.
  2. **Geometry** - distogram and pocket targets for Stage 1, and the supervised
     anchor that Section 2.1 lists as a REQUIRED mitigation for every later stage.
  3. **The Stage 3 crystal fine-tune set** (Section 8), which is the last stage of
     the curriculum and the intended use here: pretrain on the large literature
     corpus, fine-tune on the crystal-linked subset.

This loader serves (1). `features/geom_labels.py` + `scripts/phase2_geometry.py`
serve (2), and `configs/train/stage3_finetune.yaml` serves (3).

To serve (1) at all it has to produce a sequence and a SMILES, which the index file
does not contain - so it extracts them once from `<id>_protein.pdb` (SEQRES-
equivalent, via the ATOM records) and `<id>_ligand.sdf`, and caches the result. An
earlier version left both as NA, which meant every PDBbind row was silently dropped
by Phase 1's standardisation and the highest-weight source in the plan contributed
nothing.

Label policy applies here exactly as everywhere else. The index records the type
inline (`Kd=49uM`, `Ki=400mM`, `IC50=10nM`) and IC50 is a large fraction of the
general set. Those rows are dropped from the labels; whether their STRUCTURES may
still be used for geometry-only supervision is `use_ic50_only_structures`
(Section 4.2), default OFF.

Licensing (Section 0.4.9): v2020 is the last freely available release, registration
terms apply, and PDBbind-derived processed data must not be redistributed. Nothing
here downloads it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import schema as S
from . import register

## TWO index layouts are in circulation and they differ by one column:
##
##   classic  INDEX_general_PL_data.2020
##       3zzf  2.20  2012   0.40  Ki=400mM  // 3zzf.pdf (NLG)
##       code  res   year  -logKd  affinity
##
##   PDBbind+ INDEX_general_PL.2020R1.lst
##       2tpi  2.10  1982  Kd=49uM       // 2tpi.pdf (2-mer)
##       code  res   year  affinity                 <- NO -logKd column
##
## A parser that assumes the 5-field form matches NOTHING on the 4-field one and
## returns an empty frame - no exception, no warning, PDBbind silently contributes
## zero rows. `-logKd` is redundant anyway (it is derivable from the affinity), so
## it is optional here and the value is recomputed rather than trusted.
_ROW = re.compile(
    r"^(?P<code>\w{4})\s+(?P<res>[\d.]+|NMR)\s+(?P<year>\d{4})\s+"
    r"(?:(?P<logkd>[-\d.]+)\s+)?"
    ## `[A-Za-z][A-Za-z0-9]*`, not `[A-Za-z]+`: the type name contains digits
    ## (IC50, EC50). Rejecting those at the REGEX means they never enter the frame,
    ## so Section 4.2's "report the sizes under both settings" cannot be answered
    ## and `use_ic50_only_structures=true` would silently find nothing. They are
    ## parsed here and rejected by `normalize_type`, which is where the label
    ## policy belongs and where the counts become reportable.
    r"(?P<aff>[A-Za-z][A-Za-z0-9]*[<>~=][^\s]+)")
_AFF = re.compile(r"^(?P<type>[A-Za-z0-9]+)(?P<rel>[<>~]?=?)(?P<val>[\d.]+)(?P<unit>\w+)")

UNIT_TO_NM = {"mM": 1e6, "uM": 1e3, "nM": 1.0, "pM": 1e-3, "fM": 1e-6, "M": 1e9}

## Both naming conventions. The `_data` infix disappeared in the 2020R1 package.
INDEX_GLOBS = ("**/INDEX_general_PL_data.*", "**/INDEX_general_PL.*",
               "**/INDEX_refined_data.*", "**/INDEX_refined_PL.*")


def find_index(root: Path) -> Path | None:
    """The protein-LIGAND affinity index, under either naming convention.

    Deliberately excludes INDEX_general_NL / _PN / _PP - nucleic-acid-ligand,
    protein-nucleic-acid and protein-protein complexes ship in the same tarball
    and are a different problem.
    """
    for g in INDEX_GLOBS:
        hits = [p for p in sorted(root.glob(g))
                if not re.search(r"INDEX_general_(NL|PN|PP)", p.name)
                and "name" not in p.name.lower()]
        if hits:
            return hits[0]
    return None


def parse_index(path: Path) -> pd.DataFrame:
    rows, skipped = [], 0
    for ln in Path(path).read_text(errors="replace").splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        m = _ROW.match(ln)
        if not m:
            skipped += 1
            continue
        a = _AFF.match(m.group("aff"))
        if not a or a.group("unit") not in UNIT_TO_NM:
            skipped += 1
            continue
        ## "Kd<10uM" is a CONCENTRATION bound. `relation` in this project's schema
        ## applies to the stored pK label, and pK = -log10(Kd) is monotonically
        ## DECREASING, so the inequality flips: Kd < x  <=>  pK > -log10(x).
        rel = a.group("rel").replace("=", "") or "="
        rel = {"<": ">", ">": "<", "~": "="}.get(rel, "=")
        rows.append(dict(
            pdb_id=m.group("code").lower(),
            resolution=float(m.group("res")) if m.group("res") != "NMR" else np.nan,
            year=int(m.group("year")),
            raw_type=a.group("type"),
            relation=rel if rel in S.ALLOWED_RELATIONS else "=",
            value_nM=float(a.group("val")) * UNIT_TO_NM[a.group("unit")],
            pK_index=float(m.group("logkd")) if m.group("logkd") else np.nan,
        ))
    df = pd.DataFrame(rows)
    if skipped and len(df):
        print(f"    index: {len(df):,} parsed, {skipped:,} lines skipped")
    elif not len(df):
        ## An index that parses to nothing is indistinguishable from "no data"
        ## downstream, which is exactly how this failed the first time.
        raise SystemExit(
            f"{path} parsed to ZERO rows - the index format is not recognised.\n"
            "  Expected one of:\n"
            "    3zzf  2.20  2012   0.40  Ki=400mM  // ...   (classic, 5 fields)\n"
            "    2tpi  2.10  1982  Kd=49uM        // ...   (2020R1, 4 fields)\n"
            f"  First non-comment line actually seen:\n"
            + "".join(f"    {ln}\n" for ln in
                      [l for l in Path(path).read_text(errors='replace').splitlines()
                       if l.strip() and not l.startswith('#')][:2]))
    return df


_DIR_INDEX: dict[str, dict[str, Path]] = {}


def index_complex_dirs(root: Path) -> dict[str, Path]:
    """pdb_id -> its directory, wherever it sits under `root`.

    The layout is NOT always `<root>/<id>/`. The PDBbind+ 2020R1 package nests
    complexes under a release-era directory - `P-L/1981-2000/10gs/` - so assuming
    one level finds nothing and every complex is reported as missing. One recursive
    scan, cached, handles both that and a plain flat tree.
    """
    key = str(root)
    if key not in _DIR_INDEX:
        found: dict[str, Path] = {}
        for p in root.rglob("*_protein.pdb"):
            found.setdefault(p.name.replace("_protein.pdb", "").lower(), p.parent)
        _DIR_INDEX[key] = found
        if found:
            print(f"    located {len(found):,} complex directories under {root}")
    return _DIR_INDEX[key]


def complex_paths(root: Path, pdb_id: str) -> dict:
    """PDBbind's per-complex file layout, checked rather than assumed."""
    d = index_complex_dirs(root).get(pdb_id.lower(), root / pdb_id)
    return {
        "protein": d / f"{pdb_id}_protein.pdb",
        "pocket": d / f"{pdb_id}_pocket.pdb",
        "ligand_sdf": d / f"{pdb_id}_ligand.sdf",
        "ligand_mol2": d / f"{pdb_id}_ligand.mol2",
    }


def _ligand_smiles(sdf: Path, mol2: Path) -> str | None:
    from ...preflight import require_rdkit
    require_rdkit("PDBbind ligand extraction")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    for p, reader in ((sdf, lambda q: next(iter(Chem.SDMolSupplier(str(q))), None)),
                      (mol2, lambda q: Chem.MolFromMol2File(str(q)))):
        if not p.exists():
            continue
        try:
            m = reader(p)
        except Exception:
            m = None
        if m is None and p.exists():
            ## Sanitisation fails on a fair number of PDBbind ligands (valence,
            ## aromatic perception on metal-adjacent atoms). An unsanitised parse
            ## still yields a usable connectivity graph, and Phase 1's RDKit
            ## standardisation gets another attempt at it.
            try:
                m = (next(iter(Chem.SDMolSupplier(str(p), sanitize=False)), None)
                     if p.suffix == ".sdf" else
                     Chem.MolFromMol2File(str(p), sanitize=False))
            except Exception:
                m = None
        if m is not None:
            try:
                return Chem.MolToSmiles(m)
            except Exception:
                continue
    return None


def extract_structures(root: Path, pdb_ids, cache: Path | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    """(pdb_id -> sequence, ligand SMILES), extracted once and cached.

    The index file carries neither, so without this every PDBbind row fails Phase
    1's standardisation and the highest-weight source contributes nothing.
    """
    ## Cache under WORK, never inside the PDBbind tree: a staged licensed dataset
    ## is frequently on a read-only or shared mount, and Section 0.4.9 forbids
    ## redistributing PDBbind-derived processed data - keeping it out of the source
    ## tree makes it much harder to hand the whole directory to someone by accident.
    if cache is None:
        from ...paths import WORK
        cache = WORK / "pdbbind_extracted.csv"
    have = pd.read_csv(cache) if Path(cache).exists() else pd.DataFrame(
        columns=["pdb_id", "sequence", "ligand_smiles_raw", "n_res"])
    todo = sorted(set(map(str, pdb_ids)) - set(have.pdb_id.astype(str)))
    if not todo:
        if verbose:
            print(f"    structures: {len(have):,} cached at {cache}")
        return have

    from ...features.geom_labels import parse_pdb, residue_sites
    if verbose:
        print(f"    extracting sequence + SMILES from {len(todo):,} complexes "
              f"({len(have):,} cached)")
    def _flush(rows):
        """Checkpoint so a wall-time kill costs one chunk, not the whole pass."""
        if not rows:
            return
        merged = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        tmp = Path(cache).with_suffix(".csv.part")
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(tmp, index=False)
        tmp.replace(cache)

    rows, fails = [], 0
    for n, pid in enumerate(todo, 1):
        p = complex_paths(root, pid)
        seq, smi, nres = None, None, 0
        try:
            residues = parse_pdb(p["protein"]) if p["protein"].exists() else []
            if residues:
                seq, _, _, _ = residue_sites(residues)
                nres = len(seq)
        except Exception:
            pass
        try:
            smi = _ligand_smiles(p["ligand_sdf"], p["ligand_mol2"])
        except Exception:
            pass
        if not seq or not smi:
            fails += 1
        rows.append(dict(pdb_id=pid, sequence=seq, ligand_smiles_raw=smi, n_res=nres))
        if n % 500 == 0:
            ## flush=True: without it the progress of a 12,000-complex pass sits in
            ## a pipe buffer and the job looks hung for its entire duration.
            if verbose:
                print(f"      {n:,}/{len(todo):,}  ({fails} incomplete)", flush=True)
            _flush(rows)
    _flush(rows)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    if verbose:
        print(f"    extracted {len(rows):,} ({fails:,} missing a sequence or ligand) "
              f"-> {cache}")
    return out


@register("pdbbind")
def load(path: str, index: str | None = None, sequences: str | None = None,
         use_ic50_only_structures: bool = False, extract: bool = True,
         **kw) -> pd.DataFrame:
    root = Path(path).expanduser()
    if not root.exists():
        raise SystemExit(
            f"pdbbind source: {root} not found.\n"
            "  This project never downloads PDBbind - it is licensed and not\n"
            "  redistributable (Section 0.4.9). Stage a copy, then:\n"
            "      python scripts/phase1a_acquire.py --pdbbind-dir <dir>\n"
            "  Or disable the source: -o data.sources.pdbbind.enabled=false")
    idx = Path(index) if index else find_index(root)
    if idx is None:
        raise SystemExit(
            f"no protein-ligand affinity index under {root}.\n"
            f"  Looked for: {', '.join(INDEX_GLOBS)}\n"
            "  The plain-text index tarball has probably not been unpacked. In the\n"
            "  PDBbind+ 2020R1 package it is\n"
            "      index.tar.gz -> index/INDEX_general_PL.2020R1.lst")
    df = parse_index(idx)
    print(f"  pdbbind: {len(df):,} indexed complexes from {idx.name}")

    t = S.normalize_type(df["raw_type"])
    why = S.classify_rejections(df["raw_type"])
    print(f"    types: {why.value_counts().to_dict()}")
    n_kdki = int(t.isin(S.ALLOWED_TYPES).sum())
    ## Section 4.2 asks for the sizes under BOTH settings, every time.
    print(f"    affinity-labelled: {n_kdki:,} Kd/Ki | geometry-only if "
          f"use_ic50_only_structures=true: {len(df) - n_kdki:,}")
    if use_ic50_only_structures:
        print("    NOTE use_ic50_only_structures=true: those structures may be used "
              "for distogram/pocket supervision with the affinity loss MASKED. The "
              "IC50 value is never read - see scripts/phase2_geometry.py --allow-ic50.")

    d = df[t.isin(S.ALLOWED_TYPES)].copy()
    d["type"] = t[d.index]

    if sequences and Path(sequences).exists():
        sm = pd.read_csv(sequences)
        ext = sm.rename(columns={sm.columns[0]: "pdb_id", sm.columns[1]: "sequence"})
        ext["pdb_id"] = ext.pdb_id.astype(str).str.lower()
        if "ligand_smiles_raw" not in ext.columns:
            ext["ligand_smiles_raw"] = pd.NA
    elif extract:
        ext = extract_structures(root, d.pdb_id.tolist())
    else:
        raise SystemExit(
            "pdbbind needs a sequence and a SMILES per complex, and the index file "
            "has neither.\n  Leave extract=true (the default) to read them from the "
            "staged structures.")

    d = d.merge(ext[["pdb_id", "sequence", "ligand_smiles_raw"]], on="pdb_id", how="left")
    before = len(d)
    d = d[d.sequence.notna() & d.ligand_smiles_raw.notna()]
    if len(d) < before:
        print(f"    dropped {before - len(d):,} complexes whose structure or ligand "
              f"could not be read -> {len(d):,}")

    out = pd.DataFrame({
        "type": d["type"], "type_source": "column",
        "source_record_id": d["pdb_id"], "pdb_id": d["pdb_id"],
        "doc_id": "PDBBIND:" + d["pdb_id"].astype(str),
        "relation": d["relation"], "value_nM": d["value_nM"],
        "sequence": d["sequence"], "ligand_smiles_raw": d["ligand_smiles_raw"],
        "structure_path": [str(complex_paths(root, p)["protein"]) for p in d["pdb_id"]],
        "structure_source": "crystal",
    })
    out = S.conform(out, "pdbbind")
    out = out[out["pK"].notna() & (out["pK"] > 0) & (out["pK"] < 14)]
    S.assert_label_policy(out, "pdbbind")
    print(f"    kept {len(out):,} crystal-linked Kd/Ki pairs "
          f"{out['type'].value_counts().to_dict()}  (source weight 1.0, and the "
          f"Stage 3 fine-tune set)")
    return out
