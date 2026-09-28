"""Papyrus 05.7 (PLAN.md Section 4.1): a pre-curated Kd/Ki source.

Used as an additional source and as a cross-check on the project's own ChEMBL
handling, not as the sole corpus.

What the release actually looks like (verified against Zenodo record 13787634):

* Two files are needed. `05.7++_combined_set_without_stereochemistry.tsv.xz` is the
  bioactivity table and carries only `target_id` / `accession`; the sequences live in
  `05.7_combined_set_protein_targets.tsv.xz`. Reading the first alone yields a
  sequence-less frame and every row is dropped downstream without an error.
* Both are xz-compressed. `open()` on them reads compressed bytes as text.
* Each row is an AGGREGATE over every measurement of a (compound, target) pair, with
  one 0/1 flag per activity type (`type_IC50`, `type_EC50`, `type_KD`, `type_Ki`,
  `type_other`) and `pchembl_value_Mean` averaged across ALL of them. A row flagged
  Ki AND IC50 therefore has an IC50-contaminated label. Section 0.4 rule 1 forbids
  that, so a row is kept only when exactly one of Ki/Kd is set and every other flag
  is 0. The excluded-type flags are read for that exclusion and nothing else, and
  are dropped before the frame leaves this module.
* `relation` is not all "=". It describes the activity CONCENTRATION, so it flips
  into the pK space the schema stores.
* The table is ~7 GB uncompressed. It is read in chunks with `usecols`, so the
  InChI / AuxInfo columns (most of the bytes) never enter memory.

The `papyrus-scripts` package is deliberately not used: it pulls a NumPy-1-built
dependency chain and breaks on a NumPy-2 environment.
"""
from __future__ import annotations

import gzip
import io
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from ...paths import RAW
from .. import schema as S
from . import register

ZENODO_RECORD = "13787634"

## Bioactivity table columns.
C_SMILES = "SMILES"
C_INCHIKEY = "InChIKey"
C_TARGET = "target_id"
C_ACCESSION = "accession"
C_ACTIVITY = "Activity_ID"
C_RELATION = "relation"
C_QUALITY = "Quality"
C_PCHEMBL = "pchembl_value_Mean"
C_DOC = "all_doc_ids"
C_TYPE_KI, C_TYPE_KD = "type_Ki", "type_KD"
## Read ONLY to exclude mixed aggregates; never propagated.
C_TYPE_EXCLUDED = ("type_IC50", "type_EC50", "type_other")

## Targets table columns.
T_TARGET, T_SEQ = "target_id", "Sequence"

## Papyrus++ by default (see phase1a_acquire.PAPYRUS_FILES for why). The loader
## still handles the full set - pass `path=` - including its mixed-type aggregates.
BIOACTIVITY_GLOB = "*++_combined_set_without_stereochemistry.tsv.*"
TARGETS_GLOB = "*_combined_set_protein_targets.tsv.*"


def _open_text(path: Path):
    s = str(path)
    op = lzma.open if s.endswith(".xz") else gzip.open if s.endswith(".gz") else open
    return io.TextIOWrapper(op(path, "rb"), encoding="utf-8")


def _find(glob: str, what: str) -> Path:
    hits = [p for p in sorted((RAW / "papyrus").glob(glob))
            if not p.name.endswith(".part")]
    if not hits:
        raise SystemExit(
            f"no Papyrus {what} ({glob}) under {RAW / 'papyrus'} - not acquired.\n"
            "    python scripts/phase1a_acquire.py --papyrus\n"
            "  Or disable it: -o data.sources.papyrus.enabled=false")
    return hits[-1]


def _flag(s: pd.Series) -> np.ndarray:
    """Papyrus type flags are "1"/"0", sometimes blank. Anything containing 1 is set."""
    return s.fillna("").astype(str).str.contains("1", regex=False).to_numpy()


def load_sequences(path: Path | None = None) -> pd.Series:
    p = Path(path) if path else _find(TARGETS_GLOB, "protein targets table")
    with _open_text(p) as fh:
        t = pd.read_csv(fh, sep="\t", usecols=[T_TARGET, T_SEQ], low_memory=False)
    t = t.dropna(subset=[T_SEQ]).drop_duplicates(T_TARGET)
    return t.set_index(T_TARGET)[T_SEQ]


def inspect(path: str | None = None, n: int = 5000) -> None:
    """Print the real schema of a release before trusting the constants above."""
    p = Path(path) if path else _find(BIOACTIVITY_GLOB, "bioactivity table")
    with _open_text(p) as fh:
        df = pd.read_csv(fh, sep="\t", nrows=n, low_memory=False)
    print(f"{p}\n  {len(df)} sampled rows, {len(df.columns)} columns")
    for c in df.columns:
        nn = df[c].notna().sum()
        ex = df[c].dropna().head(2).tolist()
        print(f"    {c:<28} non-null {nn:>6}  e.g. {str(ex)[:70]}")
    for c in (C_TYPE_KI, C_TYPE_KD, C_RELATION, C_QUALITY):
        if c in df.columns:
            print(f"  {c} values: {df[c].value_counts(dropna=False).head(8).to_dict()}")


@register("papyrus")
def load(path: str | None = None, targets: str | None = None, quality: str = "high",
         keep_censored: bool = True, nrows: int | None = None,
         chunksize: int = 500_000, **kw) -> pd.DataFrame:
    ## Local files only. scripts/phase1a_acquire.py is the single place that
    ## touches the network, so a harmonisation job cannot stall on a download.
    p = Path(path) if path else _find(BIOACTIVITY_GLOB, "bioactivity table")
    seqs = load_sequences(Path(targets) if targets else None)
    print(f"  papyrus: {p.name}  ({len(seqs):,} target sequences)")

    with _open_text(p) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    want = [C_ACTIVITY, C_QUALITY, C_SMILES, C_INCHIKEY, C_TARGET, C_ACCESSION,
            C_DOC, C_TYPE_KI, C_TYPE_KD, *C_TYPE_EXCLUDED, C_RELATION, C_PCHEMBL]
    missing = [c for c in (C_SMILES, C_PCHEMBL, C_TARGET, C_TYPE_KI, C_TYPE_KD)
               if c not in header]
    if missing:
        raise SystemExit(
            f"papyrus release is missing expected column(s) {missing}. The schema "
            "changed; run\n    python scripts/phase1_harmonize.py --inspect-papyrus\n"
            "and update the C_* constants in this module.")
    use = [c for c in want if c in header]

    n_seen = n_ki = n_kd = n_both = n_mixed = 0
    parts = []
    with _open_text(p) as fh:
        for chunk in pd.read_csv(fh, sep="\t", usecols=use, nrows=nrows,
                                 chunksize=chunksize, dtype=str, low_memory=False):
            n_seen += len(chunk)
            ki, kd = _flag(chunk[C_TYPE_KI]), _flag(chunk[C_TYPE_KD])
            other = np.zeros(len(chunk), bool)
            for c in C_TYPE_EXCLUDED:
                if c in chunk:
                    other |= _flag(chunk[c])
            both = ki & kd
            keep = (ki ^ kd) & ~other
            n_ki += int(ki.sum()); n_kd += int(kd.sum())
            n_both += int(both.sum()); n_mixed += int(((ki | kd) & other).sum())
            sub = chunk.loc[keep].drop(columns=[c for c in C_TYPE_EXCLUDED if c in chunk])
            sub["type"] = np.where(kd[keep], "Kd", "Ki")
            parts.append(sub)
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    print(f"    {n_seen:,} rows read.  flagged Ki {n_ki:,}  Kd {n_kd:,}")
    print(f"    dropped: Ki+Kd aggregates {n_both:,};  Ki/Kd averaged with "
          f"IC50/EC50/other {n_mixed:,} (label would be contaminated)")

    out = pd.DataFrame(index=d.index)
    out["type"] = d["type"]
    out["type_source"] = "column"
    out["ligand_smiles_raw"] = d[C_SMILES]
    out["inchikey_full"] = d.get(C_INCHIKEY, pd.NA)
    ## Mutant target_ids (P00533_L858R) map to their own sequence when the targets
    ## table has one; otherwise fall back to the accession's wild type.
    seq = d[C_TARGET].map(seqs)
    if C_ACCESSION in d:
        seq = seq.fillna((d[C_ACCESSION] + "_WT").map(seqs))
    out["sequence"] = seq
    out["uniprot"] = d.get(C_ACCESSION, pd.NA)
    out["source_record_id"] = d[C_ACTIVITY] if C_ACTIVITY in d else np.arange(len(d))
    out["doc_id"] = d.get(C_DOC, pd.NA)
    out["pK"] = pd.to_numeric(d[C_PCHEMBL], errors="coerce")
    n_noseq = int(out["sequence"].isna().sum())
    print(f"    no sequence for {n_noseq:,} rows (target absent from targets table)")

    rel = (d[C_RELATION].fillna("=").astype(str).str.strip()
           if C_RELATION in d else pd.Series("=", index=d.index))
    ## Papyrus's `relation` describes the original activity CONCENTRATION, so it
    ## flips into the pK space the schema stores.
    out["relation"] = S.flip_relation(rel)
    n_cens = int((out["relation"] != "=").sum())
    if not keep_censored:
        out = out[out["relation"] == "="]
    print(f"    censored (>/<) {n_cens:,} "
          f"{'kept for the one-sided loss' if keep_censored else 'dropped'}")

    if quality and C_QUALITY in d.columns:
        q = d.loc[out.index, C_QUALITY].astype(str).str.lower()
        allowed = {"high": {"high"}, "medium": {"high", "medium"}}.get(quality)
        if allowed is not None:
            before = len(out)
            out = out[q.isin(allowed)]
            print(f"    quality={quality}: {before:,} -> {len(out):,}")

    out = out[out["sequence"].notna()]
    out = S.conform(out, "papyrus")
    out = out[out["pK"].notna() & (out["pK"] > 0) & (out["pK"] < 14)]
    S.assert_label_policy(out, "papyrus")
    print(f"    kept {len(out):,} rows  {out['type'].value_counts().to_dict()}")
    return out
