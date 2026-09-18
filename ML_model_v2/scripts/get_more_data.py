#!/usr/bin/env python
"""Gather additional Kd/Ki binding data to combine with input.csv.

    python get_more_data.py chembl            # phase 1a - NEEDS NETWORK, resumable
    python get_more_data.py chembl-targets     # phase 1b - resolve target ids -> UniProt
    python get_more_data.py bindingdb          # phase 1c - BindingDB Kd + Ki via PyTDC
    python get_more_data.py uniprot            # phase 1d - fetch sequences for new targets
    python get_more_data.py merge              # phase 2  - combine, dedupe, exclude eval sets

Why Kd/Ki only, and why these two sources
------------------------------------------
input.csv already showed diminishing returns from MORE of the same kind of data: 47k -> 1.5M
BindingDB pairs did not move pooled or benchmark performance, because (measured, this
project) the extra rows were absorbed as target-specific memorisation. IC50 is excluded
here by design - it carries a systematic, assay-condition-dependent offset from Ki
(Cheng-Prusoff) that pools two different quantities under one label, and the point of this
script is a cleaner target, not a bigger noisier one.

ChEMBL is the source input.csv itself never drew from (it is a BindingDB derivative) - it
has 757,395+ binding-assay (assay_type='B') Ki records alone, verified live against the API
before writing this. That is genuinely new signal, not a re-draw of the same well.

Verified against the live API before writing this script:
    GET .../activity.json?standard_type=Ki&assay_type=B&limit=2
      -> canonical_smiles, target_chembl_id, standard_value, standard_units,
         standard_relation are present exactly as used below.
    GET .../target/CHEMBL261.json
      -> target_components[0]['accession'] is the UniProt accession directly.

What this does NOT do
----------------------
Pull PubChem BioAssay data. That was the right call for the earlier LIT-PCBA-style broad
NEGATIVE sourcing (screening triage), but this script's target is Kd/Ki affinity for the
regressor, which is a different problem PubChem serves far less directly (it is almost all
IC50/AC50). See litpcba/prep_litpcba.py for the negative-sourcing side.

Safety / etiquette for a public API
------------------------------------
Paginated with a checkpoint file so an interrupted run resumes instead of restarting, a
capped request rate, retries with backoff on transient failures, and a real User-Agent.
Run `chembl` on a login node - compute nodes here have no internet (same constraint
documented in tabfm_hpc_offline/README_tabfm_hpc.md and litpcba/prep_litpcba.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
REPO = ROOT.parent
INPUT_CSV = REPO / "input" / "input.csv"
BENCH_CSV = REPO / "ML_model" / "benchmark" / "eval" / "output" / "bench_scored.csv"
LITPCBA_PAIRS = REPO / "ML_model" / "litpcba" / "output" / "litpcba_pairs_ave.parquet"
DATA.mkdir(parents=True, exist_ok=True)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"
STANDARD_TYPES = ["Ki", "Kd"]
PAGE_LIMIT = 1000
REQUEST_PAUSE_S = 0.34          # ~3 req/s, polite for a shared public API
MAX_RETRIES = 5
UA = {"User-Agent": "Savant-ML_model_v2/1.0 (research; contact via repo)"}

## BindingDB's censor caps and the pKd bound checks already validated in the training
## notebooks. Kept identical here so a row entering through this script is held to the
## same standard as one from input.csv.
MIN_PKD, MAX_PKD = 4.0, 12.0

## Must match the benchmark builder's own fingerprint settings
## (ML_model/benchmark/scripts/common.py) - a filter computed with different radius/bits
## than the gate it is enforcing is not enforcing that gate.
MORGAN_RADIUS, MORGAN_BITS = 2, 2048
BENCHMARK_GATE = 0.50


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.loads(fh.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            wait = 2 ** attempt
            log(f"  retry {attempt+1}/{MAX_RETRIES} after {type(e).__name__}: {e} "
                f"(sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}: {last}")


# --------------------------------------------------------------------------- #
# phase 1a - ChEMBL activities, paginated and resumable
# --------------------------------------------------------------------------- #

def cmd_chembl(args):
    out = DATA / "chembl_activities_raw.jsonl"
    ckpt = DATA / "chembl_activities.checkpoint"
    offset = int(ckpt.read_text()) if ckpt.exists() else 0
    if offset:
        log(f"resuming from offset {offset:,} ({out.name} already has partial data)")

    ## Comma-separated, NOT pipe - verified live: standard_type__in=Ki|Kd silently matches
    ## zero rows (no error, just an empty result set), while Ki,Kd returns 636,684.
    types = ",".join(STANDARD_TYPES)
    base = (f"{CHEMBL_BASE}/activity.json?standard_type__in={types}&assay_type=B"
            f"&standard_relation=%3D&standard_units=nM&limit={PAGE_LIMIT}")

    ## confirm the __in filter actually narrows results before committing to a long crawl
    probe = _get_json(f"{base}&offset=0")
    total = probe["page_meta"]["total_count"]
    log(f"ChEMBL reports {total:,} matching activities "
        f"(standard_type in {STANDARD_TYPES}, assay_type=B, relation='=', units=nM)")
    if args.limit_rows and total > args.limit_rows:
        log(f"--limit-rows {args.limit_rows:,} set; stopping early for a bounded test run")

    n_written = 0
    with open(out, "a") as fh:
        while offset < total:
            if args.limit_rows and offset >= args.limit_rows:
                break
            ## Cap this page to whatever is left of --limit-rows, or a bounded test run
            ## fetches one full PAGE_LIMIT-sized page regardless of the requested bound -
            ## caught by actually running `--limit-rows 300` and getting 1,000 rows back.
            page_url = base
            if args.limit_rows:
                page_url = page_url.replace(f"limit={PAGE_LIMIT}",
                                            f"limit={min(PAGE_LIMIT, args.limit_rows - offset)}")
            page = _get_json(f"{page_url}&offset={offset}")
            rows = page["activities"]
            if not rows:
                break
            for r in rows:
                fh.write(json.dumps({
                    "molecule_chembl_id": r.get("molecule_chembl_id"),
                    "canonical_smiles": r.get("canonical_smiles"),
                    "target_chembl_id": r.get("target_chembl_id"),
                    "target_organism": r.get("target_organism"),
                    "standard_type": r.get("standard_type"),
                    "standard_value": r.get("standard_value"),
                    "standard_units": r.get("standard_units"),
                    "pchembl_value": r.get("pchembl_value"),
                    "assay_chembl_id": r.get("assay_chembl_id"),
                    "document_year": r.get("document_year"),
                }) + "\n")
            n_written += len(rows)
            offset += len(rows)          # not PAGE_LIMIT - the last page is short
            ckpt.write_text(str(offset))
            if (offset // PAGE_LIMIT) % 20 == 0:
                log(f"  {offset:,}/{total:,} ({100*offset/total:.1f}%), "
                    f"{n_written:,} rows written this run")
            time.sleep(REQUEST_PAUSE_S)

    log(f"done: {n_written:,} rows written this run -> {out}")
    log("next: `python get_more_data.py chembl-targets`")


# --------------------------------------------------------------------------- #
# phase 1b - resolve target_chembl_id -> UniProt accession
# --------------------------------------------------------------------------- #

def cmd_chembl_targets(args):
    raw = DATA / "chembl_activities_raw.jsonl"
    if not raw.exists():
        sys.exit(f"{raw} not found - run `chembl` first")
    ids = set()
    with open(raw) as fh:
        for line in fh:
            t = json.loads(line).get("target_chembl_id")
            if t:
                ids.add(t)
    log(f"{len(ids):,} distinct target_chembl_id values to resolve")

    out = DATA / "chembl_target_uniprot.csv"
    have = set()
    if out.exists():
        have = set(pd.read_csv(out).target_chembl_id)
        log(f"  {len(have):,} already resolved, resuming")
    todo = sorted(ids - have)

    rows = []
    for i, tid in enumerate(todo):
        try:
            t = _get_json(f"{CHEMBL_BASE}/target/{tid}.json")
        except Exception as e:
            log(f"  {tid}: FAILED {type(e).__name__}: {e}")
            continue
        comps = t.get("target_components") or []
        ## SINGLE PROTEIN only - a multi-component complex has no one sequence to embed,
        ## and guessing which subunit binds the ligand is exactly the kind of silent
        ## approximation that belongs in a documented caveat, not a default.
        acc = comps[0]["accession"] if (len(comps) == 1 and comps[0].get("accession")) else None
        rows.append({"target_chembl_id": tid, "target_type": t.get("target_type"),
                     "n_components": len(comps), "uniprot_id": acc})
        if (i + 1) % 200 == 0:
            log(f"  {i+1:,}/{len(todo):,}")
            pd.concat([pd.read_csv(out), pd.DataFrame(rows)]).to_csv(out, index=False) \
                if out.exists() else pd.DataFrame(rows).to_csv(out, index=False)
            rows = []
        time.sleep(REQUEST_PAUSE_S)
    if rows:
        pd.concat([pd.read_csv(out), pd.DataFrame(rows)]).to_csv(out, index=False) \
            if out.exists() else pd.DataFrame(rows).to_csv(out, index=False)

    m = pd.read_csv(out)
    log(f"resolved {m.uniprot_id.notna().sum():,}/{len(m):,} targets to a single UniProt "
        f"accession ({100*m.uniprot_id.notna().mean():.1f}%)")
    log("next: `python get_more_data.py uniprot`")


# --------------------------------------------------------------------------- #
# phase 1c - BindingDB Kd + Ki via PyTDC (Ki was never in input.csv)
# --------------------------------------------------------------------------- #

def cmd_bindingdb(args):
    try:
        from tdc.multi_pred import DTI
    except ImportError:
        sys.exit("PyTDC is not installed. On a node with network:\n    pip install PyTDC\n"
                 "then re-run this command.")
    for name in ("BindingDB_Kd", "BindingDB_Ki"):
        out = DATA / f"{name.lower()}.tsv"
        if out.exists() and not args.force:
            log(f"{out.name} already present"); continue
        log(f"downloading {name} via TDC ...")
        d = DTI(name=name, path=str(DATA)).get_data()
        d = d.rename(columns={"Drug_ID": "ID1", "Drug": "X1", "Target_ID": "ID2",
                              "Target": "X2"})[["ID1", "X1", "ID2", "X2", "Y"]]
        d.to_csv(out, sep="\t", index=False)
        log(f"  {len(d):,} rows -> {out.name}")
    log("next: `python get_more_data.py merge`")


# --------------------------------------------------------------------------- #
# phase 1d - UniProt sequences for newly-seen targets
# --------------------------------------------------------------------------- #

def cmd_uniprot(args):
    tmap = DATA / "chembl_target_uniprot.csv"
    accs = set()
    if tmap.exists():
        accs |= set(pd.read_csv(tmap).uniprot_id.dropna())
    ## BindingDB's own accession column, if that phase has run
    for name in ("bindingdb_kd", "bindingdb_ki"):
        p = DATA / f"{name}.tsv"
        if p.exists():
            accs |= set(pd.read_csv(p, sep="\t", usecols=["ID2"]).ID2.dropna())
    log(f"{len(accs):,} distinct UniProt accessions needed")

    out = DATA / "uniprot_sequences.csv"
    have = set()
    if out.exists():
        have = set(pd.read_csv(out).uniprot_id)
        log(f"  {len(have):,} already fetched")
    todo = sorted(a for a in accs if a not in have and isinstance(a, str))

    rows = []
    for i, acc in enumerate(todo):
        try:
            req = urllib.request.Request(UNIPROT_FASTA.format(acc), headers=UA)
            with urllib.request.urlopen(req, timeout=30) as fh:
                txt = fh.read().decode()
            seq = "".join(l.strip() for l in txt.splitlines() if not l.startswith(">"))
            if seq:
                rows.append({"uniprot_id": acc, "sequence": seq})
        except Exception as e:
            log(f"  {acc}: FAILED {type(e).__name__}")
        if (i + 1) % 200 == 0:
            log(f"  {i+1:,}/{len(todo):,}")
            pd.concat([pd.read_csv(out), pd.DataFrame(rows)]).to_csv(out, index=False) \
                if out.exists() else pd.DataFrame(rows).to_csv(out, index=False)
            rows = []
        time.sleep(0.1)
    if rows:
        pd.concat([pd.read_csv(out), pd.DataFrame(rows)]).to_csv(out, index=False) \
            if out.exists() else pd.DataFrame(rows).to_csv(out, index=False)
    log(f"total sequences on file: {len(pd.read_csv(out)):,}")
    log("next: `python get_more_data.py merge`")


# --------------------------------------------------------------------------- #
# phase 2 - combine, clean, dedupe, exclude eval sets
# --------------------------------------------------------------------------- #

def pkd_from_nm(nm):
    return 9.0 - np.log10(nm)


def _load_chembl():
    raw, tmap, seqs = (DATA / "chembl_activities_raw.jsonl", DATA / "chembl_target_uniprot.csv",
                       DATA / "uniprot_sequences.csv")
    if not (raw.exists() and tmap.exists() and seqs.exists()):
        log("ChEMBL phases incomplete - skipping this source")
        return pd.DataFrame(columns=["seq", "smiles", "neg_log10_affinity_M", "standard_type",
                                     "source"])
    d = pd.read_json(raw, lines=True)
    tm = pd.read_csv(tmap).dropna(subset=["uniprot_id"])
    sq = pd.read_csv(seqs)
    d = d.merge(tm[["target_chembl_id", "uniprot_id"]], on="target_chembl_id", how="inner")
    d = d.merge(sq, on="uniprot_id", how="inner")
    d = d.rename(columns={"canonical_smiles": "smiles", "sequence": "seq"})
    d = d[d.standard_value.notna() & (d.standard_value.astype(float) > 0)]
    d["neg_log10_affinity_M"] = pkd_from_nm(d.standard_value.astype(float))
    d["source"] = "chembl"
    log(f"ChEMBL: {len(d):,} rows with sequence + value")
    return d[["seq", "smiles", "neg_log10_affinity_M", "standard_type", "source"]]


def _load_bindingdb():
    seqs_p = DATA / "uniprot_sequences.csv"
    frames = []
    for name in ("bindingdb_kd", "bindingdb_ki"):
        p = DATA / f"{name}.tsv"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t").rename(columns={"X1": "smiles", "X2": "seq", "Y": "kd_nM"})
        d = d[d.kd_nM.notna() & (d.kd_nM > 0)]
        d["neg_log10_affinity_M"] = pkd_from_nm(d.kd_nM)
        d["standard_type"] = "Kd" if "kd" in name else "Ki"
        d["source"] = "bindingdb"
        frames.append(d[["seq", "smiles", "neg_log10_affinity_M", "standard_type", "source"]])
        log(f"{name}: {len(d):,} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["seq", "smiles", "neg_log10_affinity_M", "standard_type", "source"])


def _filter_against_benchmark(new: pd.DataFrame, gate: float) -> pd.DataFrame:
    """Drop every new row whose ligand is within `gate` ECFP4 Tanimoto of ANY v3
    benchmark ligand, or shares its InChIKey skeleton.

    Direction matters: the audit asks "how close is each BENCHMARK ligand to training?",
    but filtering asks the inverse - "is this TRAINING ligand too close to any benchmark
    ligand?" - so the running max is accumulated per new ligand, not per benchmark ligand.

    Cost is 373 BulkTanimotoSimilarity sweeps over the unique new ligands (~2 min for
    245k), plus fingerprinting. That is cheap next to a 4-hour retrain on data that
    silently invalidates the benchmark.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import inchi, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    if not BENCH_CSV.exists():
        log(f"  WARNING: {BENCH_CSV} not found - CANNOT enforce the benchmark gate. "
            "Any number produced from this data is unverifiable against that benchmark.")
        return new

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    bm = pd.read_csv(BENCH_CSV, usecols=["ligand_smiles"]).ligand_smiles.dropna().unique()

    b_fps, b_keys = [], set()
    for s in bm:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        b_fps.append(gen.GetFingerprint(m))
        try:
            b_keys.add(inchi.MolToInchiKey(m)[:14])
        except Exception:
            pass
    log(f"  benchmark gate: {len(b_fps)} ligands fingerprinted, threshold Tanimoto >= {gate}")

    uniq = new.smiles.unique()
    log(f"  fingerprinting {len(uniq):,} unique new ligands")
    fps, keys, keep = [], [], []
    for s in uniq:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(gen.GetFingerprint(m))
        try:
            keys.append(inchi.MolToInchiKey(m)[:14])
        except Exception:
            keys.append(None)
        keep.append(s)

    maxsim = np.zeros(len(fps), np.float32)
    for i, bfp in enumerate(b_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(bfp, fps), np.float32)
        np.maximum(maxsim, sims, out=maxsim)
        if (i + 1) % 100 == 0:
            log(f"    {i+1}/{len(b_fps)} benchmark ligands swept")

    skeleton_hit = np.array([bool(k and k in b_keys) for k in keys])
    too_close = (maxsim >= gate) | skeleton_hit
    banned = {s for s, bad in zip(keep, too_close) if bad}

    report = pd.DataFrame({"smiles": keep, "max_tanimoto_to_benchmark": maxsim,
                           "skeleton_matches_benchmark": skeleton_hit,
                           "dropped": too_close})
    report.to_csv(DATA / "benchmark_filter_report.csv", index=False)

    before = len(new)
    new = new[~new.smiles.isin(banned)]
    log(f"  BENCHMARK GATE: dropped {len(banned):,} of {len(keep):,} unique ligands "
        f"({before-len(new):,} of {before:,} rows) for being within Tanimoto {gate} "
        f"of a benchmark ligand")
    log(f"    of those, {int(skeleton_hit.sum()):,} were exact skeleton matches")
    log(f"    surviving max-similarity to benchmark: "
        f"{maxsim[~too_close].max() if (~too_close).any() else 0:.3f} (must be < {gate})")
    log(f"    per-ligand detail -> {DATA/'benchmark_filter_report.csv'}")
    return new


def cmd_merge(args):
    chembl = _load_chembl()
    bdb = _load_bindingdb()
    ## pandas warns (soon errors) on concatenating an all-empty frame in the mix - which
    ## is the normal case here whenever one source's phases have not been run yet.
    frames = [f for f in (chembl, bdb) if not f.empty]
    new = pd.concat(frames, ignore_index=True) if frames else chembl
    if new.empty:
        sys.exit("no rows from either source - run `chembl`/`chembl-targets`/`uniprot` "
                 "and/or `bindingdb` first")
    n0 = len(new)
    log(f"combined new rows: {n0:,}")

    new = new[new.neg_log10_affinity_M.between(MIN_PKD, MAX_PKD)]
    new = new.dropna(subset=["seq", "smiles"])
    new = new[new.seq.str.len().between(10, 2000)]
    log(f"  after pKd/length bounds: {len(new):,} ({100*len(new)/n0:.1f}%)")

    ## Exclude anything already an exact (seq, smiles) pair in input.csv - this is
    ## ADDITIONAL data, not a re-draw of what training already has.
    if INPUT_CSV.exists():
        seen = set()
        for ch in pd.read_csv(INPUT_CSV, chunksize=300_000, usecols=["seq", "smiles"]):
            seen.update(zip(ch.seq, ch.smiles))
        before = len(new)
        new = new[~new.apply(lambda r: (r.seq, r.smiles) in seen, axis=1)]
        log(f"  dropped {before-len(new):,} rows already in input.csv "
            f"({len(new):,} genuinely new)")
    else:
        log(f"  WARNING: {INPUT_CSV} not found - cannot dedupe against it")

    ## Exclude anything overlapping the EVALUATION sets by exact pair. This does not
    ## replace a Tanimoto contamination audit (run one on the merged output before
    ## training, the same way litpcba/prep_litpcba.py audits against input.csv) - it
    ## only removes the cheapest, most direct leak: the identical (protein, ligand) pair
    ## appearing in both train and test.
    for name, path, seqcol, smicol in [
        ("v3 benchmark", BENCH_CSV, "target_sequence", "ligand_smiles"),
        ("LIT-PCBA", LITPCBA_PAIRS, "target_sequence", "smiles"),
    ]:
        if not path.exists():
            log(f"  ({name} not found at {path}, skipping its exclusion)")
            continue
        ev = (pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path))
        pairs = set(zip(ev[seqcol], ev[smicol]))
        before = len(new)
        new = new[~new.apply(lambda r: (r.seq, r.smiles) in pairs, axis=1)]
        ## Always log a line, even when 0 are dropped - the check ran-and-passed and the
        ## check never-ran must not look identical in the output. A run that silently
        ## checked nothing is indistinguishable from one that checked and found no
        ## overlap, and this exclusion is the thing standing between "new data" and
        ## "quietly trained on the eval set".
        log(f"  checked against {name} ({len(pairs):,} pairs): "
            f"dropped {before-len(new):,} exact matches")

    ## THE GATE. Exact-pair exclusion above is not enough - measured, not assumed: the
    ## first run of this script passed every exact-pair check and still destroyed the v3
    ## benchmark, because 287 of its 373 ligands ended up within ECFP4 Tanimoto 0.50 of
    ## something in the new data (median 0.769, 107 of them above 0.90), all traced to the
    ## ChEMBL crawl. The benchmark's entire validity rests on that 0.50 gate, so it is
    ## enforced here rather than left to a downstream audit nobody runs in time.
    if not args.skip_benchmark_filter:
        new = _filter_against_benchmark(new, args.benchmark_gate)

    ## Collapse duplicates the same way input.csv's cleaning does, so downstream code
    ## sees one convention: median, keep n_measurements for weighting.
    g = new.groupby(["seq", "smiles"])
    out = g.agg(neg_log10_affinity_M=("neg_log10_affinity_M", "median"),
               n_measurements=("neg_log10_affinity_M", "size"),
               label_sd=("neg_log10_affinity_M", "std"),
               standard_type=("standard_type", lambda s: s.mode().iat[0]),
               source=("source", lambda s: "+".join(sorted(s.unique())))).reset_index()
    out["label_sd"] = out.label_sd.fillna(0.0)
    out["affinity_Kd_nM"] = (10 ** (-out.neg_log10_affinity_M) * 1e9).round(5)
    out.insert(0, "ID", "extra_" + (out.index + 1).astype(str))

    dest = Path(args.out) if args.out else DATA / "expanded_kd_ki.csv"
    out.to_csv(dest, index=False)
    log(f"wrote {len(out):,} new (protein, ligand) pairs -> {dest}")
    log(f"  by source: {out.source.value_counts().to_dict()}")
    log(f"  by assay type: {out.standard_type.value_counts().to_dict()}")
    log(f"\nCombine with input.csv in the training notebook by concatenating on "
        f"[seq, smiles, neg_log10_affinity_M, affinity_Kd_nM] - do not overwrite input.csv.")
    if args.skip_benchmark_filter:
        log("WARNING: --skip-benchmark-filter was set. This file may contain ligands that "
            "invalidate the v3 benchmark; do not quote benchmark numbers from a model "
            "trained on it.")
    else:
        log("The v3 benchmark gate was enforced. Verify independently before quoting any "
            "benchmark number:")
        log("    python scripts/audit_expanded_vs_benchmark.py")
    log("NOTE: this gate covers the v3 benchmark only. LIT-PCBA contamination is handled "
        "separately by ML_model/litpcba/prep_litpcba.py's own audit, which compares against "
        "input.csv - re-run it against the expanded corpus before quoting LIT-PCBA numbers.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chembl"); p.add_argument("--limit-rows", type=int, default=0,
                                                 help="stop after this many rows (0 = all)")
    sub.add_parser("chembl-targets")
    p = sub.add_parser("bindingdb"); p.add_argument("--force", action="store_true")
    sub.add_parser("uniprot")
    p = sub.add_parser("merge")
    p.add_argument("--benchmark-gate", type=float, default=BENCHMARK_GATE,
                   help=f"drop new ligands within this ECFP4 Tanimoto of any v3 benchmark "
                        f"ligand (default {BENCHMARK_GATE}, matching the benchmark's own gate)")
    p.add_argument("--skip-benchmark-filter", action="store_true",
                   help="do NOT enforce the benchmark gate - only for inspecting raw yield; "
                        "data produced this way invalidates the v3 benchmark")
    p.add_argument("--out", default=None,
                   help="write somewhere other than data/expanded_kd_ki.csv. Combined with "
                        "--skip-benchmark-filter this regenerates the PRE-filter file "
                        "without clobbering the good one - which is how "
                        "backfill_embedding_keys.py recovers the original embedding order "
                        "if merge was already run before the keys existed")
    args = ap.parse_args()
    {"chembl": cmd_chembl, "chembl-targets": cmd_chembl_targets, "bindingdb": cmd_bindingdb,
     "uniprot": cmd_uniprot, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
