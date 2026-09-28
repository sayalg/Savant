#!/usr/bin/env python3
"""Phase 1a: ACQUIRE the raw Kd/Ki sources (PLAN.md Section 4.1).

This is the only step in the project that touches the network, and it is separated
from Phase 1b (harmonisation) on purpose:

  * it is slow, I/O-bound and must be resumable - the ChEMBL pager alone is ~700k
    activities, and a wall-time kill after four hours must not mean starting over;
  * it needs no RDKit, so it cannot die on the NumPy ABI break that has bitten this
    project twice;
  * a compute node and a login node have different network behaviour, and mixing
    the download with a 200 GB-RAM pandas merge wastes an allocation either way.

Everything lands in `data/raw/<source>/` with a `MANIFEST.json` recording what was
fetched, when, from where, and how big it is. `sources/*.py` then read ONLY from
those local files - no loader reaches the network.

Sources, and why each one is here (Section 4.1):

    chembl      the largest single pool of binding-assay Ki/Kd, and the one
                input.csv never drew from (it is a BindingDB derivative)
    bindingdb   ~560k Ki + ~100k Kd; large overlap with ChEMBL, deduped in 1b
    papyrus     pre-curated and standardised; a cross-check on our own handling
    iuphar      small, hand-curated, and the best GPCR/ion-channel coverage
    pdbbind     crystal complexes: Kd/Ki labels AND the geometry for Stage 1/3.
                Licensed - this script never downloads it, only checks and reports.

    python scripts/phase1a_acquire.py --all
    python scripts/phase1a_acquire.py --chembl          # resumable, hours
    python scripts/phase1a_acquire.py --status
"""
from __future__ import annotations

import argparse
import gzip
import http.client
import json
import math
import re
import shutil
import threading
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401

import pandas as pd  # noqa: E402

from pfpl import paths  # noqa: E402

RAW = paths.RAW
MANIFEST = RAW / "MANIFEST.json"
UA = "pfpl/0.1 (academic research; protein-ligand affinity)"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_PAGE = 1000
CHEMBL_PAUSE = 0.20            # be a good citizen on a public API

## BindingDB keeps ONLY the current monthly dump (the 202508 file that used to be
## pinned here now 404s), so the name is read off the download page each run.
## The page's own links go through SDFdownload.jsp, which returns an HTML page,
## not the zip - the direct /rwd/bind/downloads/ path is the one that serves bytes
## (and honours Range, so an interrupted download resumes).
BINDINGDB_PAGE = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
BINDINGDB_DIR = "https://www.bindingdb.org/rwd/bind/downloads/"
BINDINGDB_URL = None           # None -> discover the current release
IUPHAR_BASE = "https://www.guidetopharmacology.org/DATA"
## BOTH files are needed: interactions.csv carries only a Ligand ID and
## the structures live in ligands.csv.
IUPHAR_FILES = ("interactions.csv", "ligands.csv")
ZENODO_PAPYRUS = "13787634"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def record(source: str, **kw) -> None:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m[source] = {"acquired": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1))


def get_json(url: str, retries: int = 6) -> dict:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as fh:
                return json.load(fh)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = min(2 ** i, 60)
            log(f"    {type(e).__name__}; retry {i + 1}/{retries} in {wait}s")
            time.sleep(wait)
    return {}


def download(url: str, dest: Path, expect_min_mb: float = 0.0) -> Path:
    """Stream to disk with resume, atomic rename, and a size sanity check."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > expect_min_mb * 1e6:
        log(f"    already have {dest.name} ({human(dest.stat().st_size)})")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if have:
        ## Resume where the last attempt stopped. A 1.5 GB download over a flaky
        ## link should not restart from zero.
        req.add_header("Range", f"bytes={have}-")
        log(f"    resuming at {human(have)}")
    with urllib.request.urlopen(req, timeout=1800) as r:
        total = int(r.headers.get("Content-Length") or 0) + have
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(tmp, mode) as w:
            t0, last = time.time(), have
            while chunk := r.read(1 << 20):
                w.write(chunk)
                have += len(chunk)
                if time.time() - t0 > 10:
                    rate = (have - last) / (time.time() - t0) / 1e6
                    pct = f"{100 * have / total:.0f}%" if total else "?"
                    log(f"    {human(have)} / {human(total) if total else '?'} "
                        f"({pct}) {rate:.1f} MB/s")
                    t0, last = time.time(), have
    if expect_min_mb and tmp.stat().st_size < expect_min_mb * 1e6:
        raise SystemExit(
            f"{dest.name} is only {human(tmp.stat().st_size)}, expected at least "
            f"{expect_min_mb} MB. The URL probably returned an error page rather "
            f"than the file - open {url} and check.")
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------- ChEMBL

def acquire_chembl(limit_rows: int | None = None, force: bool = False) -> None:
    """Paged, checkpointed fetch of every binding-assay Kd/Ki activity.

    Filters applied AT THE QUERY, so nothing excluded is ever downloaded:

        standard_type in (Kd, Ki)   the label policy
        assay_type = B              BINDING assays only. A Ki from a functional
                                    assay is a different quantity measured a
                                    different way; pooling them is the same
                                    category error as pooling IC50.
        standard_units = nM         one unit, no conversion guesswork

    NOT filtered here: `standard_relation`. Section 5.2 keeps ">" and "<" as
    censored labels for the one-sided loss, so dropping them at the API would
    throw away information the plan asks to use.
    """
    out = RAW / "chembl"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_p = out / "checkpoint.json"
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)

    if force:
        for f in shard_dir.glob("*.parquet"):
            f.unlink()
        ckpt_p.unlink(missing_ok=True)

    ckpt = json.loads(ckpt_p.read_text()) if ckpt_p.exists() else {"offset": 0}
    offset = int(ckpt.get("offset", 0))
    if offset:
        log(f"  resuming ChEMBL at offset {offset:,}")

    only = ",".join([
        "activity_id", "molecule_chembl_id", "canonical_smiles", "target_chembl_id",
        "standard_type", "standard_relation", "standard_value", "standard_units",
        "pchembl_value", "document_chembl_id", "data_validity_comment",
        "assay_chembl_id", "assay_type",
    ])
    params = {
        "standard_type__in": "Kd,Ki",
        "assay_type": "B",
        "standard_units": "nM",
        "target_confidence_score": 9,
        "limit": CHEMBL_PAGE,
        "format": "json",
        "only": only,
    }
    base = f"{CHEMBL_BASE}/activity.json?" + urllib.parse.urlencode(params)

    buf, t0, total = [], time.time(), None
    while True:
        js = get_json(f"{base}&offset={offset}")
        rows = js.get("activities", [])
        if total is None:
            total = js.get("page_meta", {}).get("total_count")
            log(f"  ChEMBL reports {total:,} matching activities" if total else
                "  ChEMBL total_count unavailable")
        if not rows:
            break
        buf.extend(rows)
        ## offset advances by what we RECEIVED, not by the page size: the last page
        ## is short, and stepping by PAGE_LIMIT there would skip records.
        offset += len(rows)

        if len(buf) >= 20_000:
            f = shard_dir / f"act_{offset:09d}.parquet"
            pd.DataFrame(buf).to_parquet(f)
            buf = []
            ckpt_p.write_text(json.dumps({"offset": offset, "total": total}))
            rate = offset / max(time.time() - t0, 1)
            eta = (total - offset) / rate / 60 if total and rate else float("nan")
            log(f"  {offset:,}{f' / {total:,}' if total else ''}  "
                f"{rate:.0f} rows/s  eta {eta:.0f} min")
        if limit_rows and offset >= limit_rows:
            break
        time.sleep(CHEMBL_PAUSE)

    if buf:
        pd.DataFrame(buf).to_parquet(shard_dir / f"act_{offset:09d}.parquet")
    ckpt_p.write_text(json.dumps({"offset": offset, "total": total, "done": True}))

    shards = sorted(shard_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.drop_duplicates("activity_id")
    df.to_parquet(out / "activities.parquet")
    log(f"  ChEMBL activities: {len(df):,} unique -> {out / 'activities.parquet'}")

    ## Target -> UniProt + sequence. Only SINGLE PROTEIN targets; confidence 9
    ## should already guarantee it, but a multi-component entry slipping through
    ## would attach one sequence to a complex.
    tgt_p = out / "targets.json"
    known = json.loads(tgt_p.read_text()) if tgt_p.exists() else {}
    todo = [t for t in df.target_chembl_id.dropna().unique() if t not in known]
    log(f"  resolving {len(todo):,} targets ({len(known):,} cached)")
    for i in range(0, len(todo), 20):
        chunk = todo[i:i + 20]
        url = (f"{CHEMBL_BASE}/target.json?target_chembl_id__in={','.join(chunk)}"
               "&limit=20&format=json")
        for t in get_json(url).get("targets", []):
            comps = t.get("target_components", [])
            ## accession only - the target endpoint carries NO sequence (verified
            ## live). fetch_uniprot() resolves these below.
            known[t["target_chembl_id"]] = (
                {"uniprot": comps[0].get("accession"),
                 "organism": t.get("organism")}
                if t.get("target_type") == "SINGLE PROTEIN" and len(comps) == 1
                else None)
        for c in chunk:
            known.setdefault(c, None)
        tgt_p.write_text(json.dumps(known))
        if (i // 20) % 25 == 0:
            log(f"    {i + len(chunk):,}/{len(todo):,}")
        time.sleep(CHEMBL_PAUSE)

    ## Resolve the accessions to sequences NOW, so the loader never needs network.
    seqs = fetch_uniprot({v["uniprot"] for v in known.values()
                          if v and v.get("uniprot")})
    have = set(seqs.accession.astype(str))
    n_ok = sum(1 for v in known.values()
               if v and str(v.get("uniprot", "")).split("-")[0] in have)
    record("chembl", activities=int(len(df)), targets=len(known),
           targets_with_sequence=n_ok, offset=offset, reported_total=total,
           filters="standard_type in (Kd,Ki), assay_type=B, units=nM, confidence=9",
           note="standard_relation NOT filtered - censored values are kept for the "
                "one-sided loss (Section 5.2)")
    log(f"  ChEMBL done: {len(df):,} activities, {n_ok:,} targets with a sequence")


# ------------------------------------------------------------------ UniProt

UNIPROT_BATCH = 100


def fetch_uniprot(accessions, verbose: bool = True) -> pd.DataFrame:
    """accession -> sequence, cached, batched, resumable.

    Needed because **ChEMBL's target endpoint returns an accession but NOT a
    sequence** - `target_components[0]` has `accession`, `component_description`,
    `component_type`, xrefs and synonyms, and no residues anywhere. Assuming a
    `sequence` key there silently resolves every target to None, which drops the
    entire ChEMBL contribution at Phase 1 without an error. Verified against the
    live API before writing this.

    IUPHAR has the same shape (accession, no sequence), so one cache serves both.
    """
    out = RAW / "uniprot"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "sequences.csv"
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(
        columns=["accession", "sequence"])
    known = set(have.accession.astype(str))
    ## Strip isoform suffixes ("P12345-2"): the canonical entry is what the corpus
    ## keys on, and the isoform endpoint answers differently.
    want = sorted({str(a).split("-")[0].strip() for a in accessions
                   if isinstance(a, str) and a.strip()} - known)
    if not want:
        if verbose:
            log(f"  uniprot: {len(have):,} sequences cached, nothing new")
        return have

    if verbose:
        log(f"  uniprot: fetching {len(want):,} sequences ({len(known):,} cached)")
    rows = []
    for i in range(0, len(want), UNIPROT_BATCH):
        chunk = want[i:i + UNIPROT_BATCH]
        url = ("https://rest.uniprot.org/uniprotkb/accessions?accessions="
               + ",".join(chunk) + "&format=fasta")
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as fh:
                    txt = fh.read().decode()
                break
            except Exception as e:
                if attempt == 4:
                    log(f"    batch {i} failed permanently: {type(e).__name__}")
                    txt = ""
                    break
                time.sleep(min(2 ** attempt, 30))
        acc, buf = None, []
        for ln in txt.splitlines():
            if ln.startswith(">"):
                if acc:
                    rows.append({"accession": acc, "sequence": "".join(buf)})
                parts = ln[1:].split("|")
                acc = parts[1] if len(parts) > 2 else parts[0].split()[0]
                buf = []
            elif acc:
                buf.append(ln.strip())
        if acc:
            rows.append({"accession": acc, "sequence": "".join(buf)})
        if verbose and (i // UNIPROT_BATCH) % 10 == 0:
            log(f"    {min(i + UNIPROT_BATCH, len(want)):,}/{len(want):,}")
        ## Checkpoint every batch: 6,000 accessions is ~60 requests and a kill
        ## halfway should not repeat them.
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
            "accession").to_csv(cache, index=False)
        time.sleep(0.1)

    final = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    final = final.drop_duplicates("accession")
    final.to_csv(cache, index=False)
    miss = len(want) - len(rows)
    record("uniprot", sequences=int(len(final)), fetched=len(rows), unresolved=miss)
    if verbose:
        log(f"  uniprot: {len(final):,} sequences cached "
            f"({miss:,} accessions returned nothing - obsolete or demerged entries)")
    return final


def acquire_uniprot() -> None:
    """Collect every accession seen in the acquired sources and resolve it once."""
    accs: set[str] = set()
    tgt = RAW / "chembl" / "targets.json"
    if tgt.exists():
        for v in json.loads(tgt.read_text()).values():
            if v and v.get("uniprot"):
                accs.add(v["uniprot"])
    iu = RAW / "iuphar" / "interactions.csv"
    if iu.exists():
        d = pd.read_csv(iu, skiprows=1, low_memory=False)
        col = "Target UniProt ID"
        if col in d.columns:
            for v in d[col].dropna().astype(str):
                accs.update(x.strip() for x in v.split("|") if x.strip())
    if not accs:
        log("  uniprot: no accessions found yet - run --chembl / --iuphar first")
        return
    fetch_uniprot(accs)


# ---------------------------------------------------------------- BindingDB

def current_bindingdb_url() -> str:
    """The newest BindingDB_All_YYYYMM_tsv.zip listed on the download page."""
    req = urllib.request.Request(BINDINGDB_PAGE, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    names = sorted(set(re.findall(r"BindingDB_All_\d{6}_tsv\.zip", html)))
    if not names:
        raise SystemExit(f"no BindingDB_All_*_tsv.zip listed on {BINDINGDB_PAGE}")
    return BINDINGDB_DIR + names[-1]


def _md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def acquire_bindingdb(url: str | None = BINDINGDB_URL) -> None:
    ## Already have a dump? Keep it rather than re-fetching 600 MB because a new
    ## month was published mid-sprint - the corpus should not shift under a run.
    have = sorted((RAW / "bindingdb").glob("BindingDB_All_*_tsv.zip"))
    if have and not url:
        p = have[-1]
        log(f"    already have {p.name} ({human(p.stat().st_size)}); "
            "pass --bindingdb-url to fetch a different release")
        record("bindingdb", file=str(p), bytes=p.stat().st_size)
        return
    try:
        url = url or current_bindingdb_url()
        dest = RAW / "bindingdb" / Path(urllib.parse.urlparse(url).path).name
        log(f"  BindingDB <- {url}")
        p = download(url, dest, expect_min_mb=100)
    except Exception as e:
        raise SystemExit(
            f"BindingDB download failed: {type(e).__name__}: {e}\n\n"
            f"Check the current file name at {BINDINGDB_PAGE} and pass the DIRECT "
            "link (not the SDFdownload.jsp one, which returns HTML):\n"
            "    python scripts/phase1a_acquire.py --bindingdb \\\n"
            f"        --bindingdb-url {BINDINGDB_DIR}BindingDB_All_YYYYMM_tsv.zip"
        ) from None
    ## BindingDB publishes an md5 beside every dump; a truncated zip otherwise
    ## surfaces much later as a BadZipFile inside the 200 GB harmonisation job.
    md5 = None
    try:
        req = urllib.request.Request(url.replace(".zip", ".md5"),
                                     headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            md5 = r.read().decode().split()[0].lower()
    except Exception:
        log("    (no md5 published; skipping the checksum)")
    if md5:
        got = _md5(p)
        if got != md5:
            p.rename(p.with_suffix(".zip.bad"))
            raise SystemExit(f"BindingDB md5 mismatch ({got} != {md5}); the file was "
                             "moved aside as .zip.bad - re-run to download it again")
        log("    md5 verified")
    record("bindingdb", file=str(p), bytes=p.stat().st_size, url=url, md5=md5)
    log(f"  BindingDB: {human(p.stat().st_size)} at {p}")


# ------------------------------------------------------------------ Papyrus

## Papyrus 05.7 file names never contain "papyrus" - the old filter matched nothing.
## Match by role instead. Two files are needed: the bioactivity table carries only
## `target_id`/`accession`, and the sequences live in the separate targets table.
##
## Papyrus++ (57 MB), not the full set (750 MB): it is what PLAN.md Section 4.1
## names, it is High quality only with the 0.5-log reproducibility filter applied,
## and it never averages Ki/Kd with IC50 in one row. Measured: 250,874 Kd/Ki pairs.
PAPYRUS_FILES = {
    "bioactivity": re.compile(r"^\d+\.\d+\+\+_combined_set_without_stereochemistry\.tsv\.xz$"),
    "targets": re.compile(r"^\d+\.\d+_combined_set_protein_targets\.tsv\.xz$"),
}


def acquire_papyrus() -> None:
    js = get_json(f"https://zenodo.org/api/records/{ZENODO_PAPYRUS}")
    avail = js.get("files", [])
    got = {}
    for role, pat in PAPYRUS_FILES.items():
        f = next((x for x in avail if pat.match(x["key"])), None)
        if f is None:
            raise SystemExit(
                f"Papyrus: no {role} file matching {pat.pattern} in Zenodo record "
                f"{ZENODO_PAPYRUS}. Available:\n  "
                + "\n  ".join(x["key"] for x in avail))
        log(f"  Papyrus {role} <- {f['key']} ({human(f.get('size', 0))})")
        got[role] = download(f["links"]["self"], RAW / "papyrus" / f["key"],
                             expect_min_mb=1)
    record("papyrus", file=str(got["bioactivity"]), targets=str(got["targets"]),
           bytes=sum(p.stat().st_size for p in got.values()), zenodo=ZENODO_PAPYRUS)
    log(f"  Papyrus: {human(got['bioactivity'].stat().st_size)} + targets "
        f"at {RAW / 'papyrus'}")


# ------------------------------------------------------------------- IUPHAR

def acquire_iuphar() -> None:
    """Guide to PHARMACOLOGY: small, hand-curated, best GPCR/ion-channel coverage.

    Worth the two minutes it costs: ChEMBL and BindingDB are both literature
    scrapes, and IUPHAR is the one source in Section 4.1 that is expert-curated
    end to end - a useful independent check on the others rather than more of the
    same.
    """
    got = {}
    for name in IUPHAR_FILES:
        url = f"{IUPHAR_BASE}/{name}"
        log(f"  IUPHAR <- {url}")
        q = download(url, RAW / "iuphar" / name, expect_min_mb=0.5)
        got[name] = q.stat().st_size
    record("iuphar", files=got, bytes=sum(got.values()), base=IUPHAR_BASE)
    log(f"  IUPHAR: {human(sum(got.values()))} in {RAW / 'iuphar'}")


# ------------------------------------------------- RCSB crystal complexes
# An OPEN substitute for PDBbind: no registration, no licence gate, and it can be
# fetched automatically. See the `--rcsb-crystals` note in check_pdbbind().

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FILES = "https://files.rcsb.org/download"
RCSB_LIGAND = "https://files.rcsb.org/ligands/download"
UNIT_TO_NM = {"mM": 1e6, "uM": 1e3, "nM": 1.0, "pM": 1e-3, "fM": 1e-6, "M": 1e9}


def rcsb_entries_with_affinity(types=("Kd", "Ki")) -> list[str]:
    """Every PDB entry carrying a Kd or Ki binding-affinity annotation.

    Verified live: 4,562 entries. RCSB redistributes these annotations (provenance
    is mostly BindingDB and PDBbind) under its own open terms, which is what makes
    this route usable where PDBbind's own download is not.
    """
    ids, start, page = [], 0, 1000
    while True:
        q = {
            "query": {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_binding_affinity.type",
                "operator": "in", "value": list(types)}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": page},
                                "results_content_type": ["experimental"]},
        }
        js = get_json(RCSB_SEARCH + "?json=" + urllib.parse.quote(json.dumps(q)))
        hits = js.get("result_set", [])
        if not hits:
            break
        ids += [h["identifier"] for h in hits]
        total = js.get("total_count", 0)
        log(f"    {len(ids):,} / {total:,} entry ids")
        start += page
        if start >= total:
            break
        time.sleep(0.1)
    return ids


RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
_GQL = """query($ids:[String!]!){ entries(entry_ids:$ids){
  rcsb_id
  rcsb_binding_affinity { comp_id type value unit provenance_code }
  rcsb_entry_info { resolution_combined }
  rcsb_accession_info { initial_release_date }
}}"""


def _affinity_rows(entry_ids, types=("Kd", "Ki"), batch: int = 500) -> list[dict]:
    """Affinity records for many entries at once, via GraphQL.

    Batched because the per-entry REST route is one HTTP round trip per entry:
    4,562 of them is 15-20 minutes of almost pure latency, against about five
    requests here.
    """
    out = []
    ids = [str(i).upper() for i in entry_ids]
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        body = json.dumps({"query": _GQL, "variables": {"ids": chunk}}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    RCSB_GRAPHQL, data=body,
                    headers={"User-Agent": UA, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as fh:
                    js = json.load(fh)
                break
            except Exception as e:
                if attempt == 4:
                    log(f"    batch at {i} failed: {type(e).__name__}")
                    js = {"data": {"entries": []}}
                    break
                time.sleep(min(2 ** attempt, 30))
        for d in ((js.get("data") or {}).get("entries") or []):
            if not d:
                continue
            res = ((d.get("rcsb_entry_info") or {}).get("resolution_combined")
                   or [None])[0]
            yr = str((d.get("rcsb_accession_info") or {}).get(
                "initial_release_date", ""))[:4]
            for a in (d.get("rcsb_binding_affinity") or []):
                if a.get("type") not in types or a.get("unit") not in UNIT_TO_NM:
                    continue
                out.append({
                    "pdb_id": d["rcsb_id"].lower(), "comp_id": a.get("comp_id"),
                    "type": a["type"],
                    "value_nM": float(a["value"]) * UNIT_TO_NM[a["unit"]],
                    "provenance": a.get("provenance_code"),
                    "resolution": res, "year": yr,
                })
        log(f"    affinities: {min(i + batch, len(ids)):,}/{len(ids):,} entries")
        time.sleep(0.1)
    return out


_local = threading.local()
_ideal_cache: dict[str, str] = {}
_ideal_lock = threading.Lock()


def _conn(host: str):
    """One kept-alive HTTPS connection per thread.

    `urllib.request.urlopen` opens a NEW connection for every call, so fetching
    4,524 complexes means ~9,000 TLS handshakes. On a fast link that is merely
    wasteful; through an HPC's shared egress it is the entire cost - measured at
    27 s per complex against 0.7 s locally, a 40x gap that is nearly all handshake.
    """
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    c = conns.get(host)
    if c is None:
        c = conns[host] = http.client.HTTPSConnection(host, timeout=120)
    return c


def _http_get(host: str, path: str, gzipped: bool = False, retries: int = 3):
    """GET over the thread's persistent connection, with reconnect-and-retry.

    A kept-alive connection WILL be closed by the server eventually; that surfaces
    as a random exception mid-run, so a retry that rebuilds the connection is not
    optional here.
    """
    for attempt in range(retries):
        try:
            c = _conn(host)
            c.request("GET", path, headers={"User-Agent": UA,
                                            "Connection": "keep-alive"})
            resp = c.getresponse()
            body = resp.read()
            if resp.status != 200:
                return None
            return gzip.decompress(body) if gzipped else body
        except Exception:
            ## Drop the dead connection so the next attempt makes a fresh one.
            try:
                _local.conns.pop(host, None).close()
            except Exception:
                pass
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def _fetch_complex(r):
    """(row, pdb_text, ideal_sdf). Runs in a worker thread - no RDKit in here."""
    ## .pdb.gz, not .pdb: ~4.2x less data (2.38 MB -> 0.56 MB over 8 entries
    ## measured), which matters on a bandwidth-constrained compute node.
    raw = _http_get("files.rcsb.org", f"/download/{r.pdb_id.upper()}.pdb.gz",
                    gzipped=True)
    if raw is None:
        return r, None, None
    ideal = _ideal_cache.get(r.comp_id)
    if ideal is None:
        ## Ligands repeat across entries (a congeneric series shares one comp_id),
        ## so cache by comp_id rather than re-fetching per complex.
        b = _http_get("files.rcsb.org", f"/ligands/download/{r.comp_id}_ideal.sdf")
        if b is None:
            return r, None, None
        ideal = b.decode(errors="replace")
        with _ideal_lock:
            _ideal_cache[r.comp_id] = ideal
    return r, raw.decode(errors="replace"), ideal


def _write_complex(out_dir: Path, pdb_id: str, comp_id: str, pdb_text: str,
                   ideal_sdf: str) -> bool:
    """Split an entry into PDBbind's on-disk layout.

    Writing `<id>/<id>_protein.pdb` + `<id>/<id>_ligand.sdf` means the EXISTING
    pdbbind loader and phase2_geometry.py consume this with no changes at all -
    no second loader, no second geometry path, no second thing to keep in sync.

    Bond orders come from the chemical component's ideal SDF and coordinates from
    the entry's HETATM records, joined with RDKit's AssignBondOrdersFromTemplate.
    Inferring bond orders from coordinates alone is the classic way to get
    aromaticity and charges wrong on exactly the ligands that matter.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    d = out_dir / pdb_id
    d.mkdir(parents=True, exist_ok=True)
    lines = pdb_text.splitlines()

    ## protein = ATOM records only; waters, ions and other heteroatoms are not the
    ## binding partner and would be parsed as residues.
    prot = [ln for ln in lines if ln.startswith(("ATOM", "TER"))]
    het = [ln for ln in lines if ln.startswith("HETATM") and ln[17:20].strip() == comp_id
           ## Drop explicit hydrogens HERE, by element column, rather than relying
           ## on RDKit's removeHs: that only works on a SANITIZED molecule, and the
           ## fragment cannot be sanitized before its bond orders are assigned. Some
           ## entries model every hydrogen (7LEG has 53 of them), which made the
           ## fragment 104 atoms against a 51-atom template and failed every match.
           and ln[76:78].strip() != "H"
           ## First altloc only; a second conformation would double the atom count
           ## in exactly the same way.
           and ln[16] in (" ", "A")]
    if len(prot) < 100 or not het:
        return False
    ## One copy of the ligand: an entry often has the same ligand in several chains,
    ## and concatenating them makes one impossible 2x-size molecule.
    chain, resseq = het[0][21], het[0][22:27]
    het = [ln for ln in het if ln[21] == chain and ln[22:27] == resseq]

    (d / f"{pdb_id}_protein.pdb").write_text("\n".join(prot) + "\nEND\n")
    try:
        frag = Chem.MolFromPDBBlock("\n".join(het) + "\nEND\n", sanitize=False,
                                    removeHs=True)
        ## sanitize=True AND removeHs=True, both load-bearing. The RCSB ideal SDF
        ## carries explicit hydrogens (24 atoms for a 12-heavy-atom sugar), and
        ## AssignBondOrdersFromTemplate needs the template and the fragment to have
        ## the SAME heavy-atom count. Loading with sanitize=False leaves the Hs in,
        ## every match fails, and the fallback silently writes ligands with no
        ## double bonds at all - `S(N)(O)O` instead of `S(=O)(=O)N`, which wrecks
        ## TPSA, HBD/HBA and Crippen logP for the whole Tier-A feature block.
        ## Measured: 13 of 20 complexes fell back before this was fixed.
        tmpl = Chem.MolFromMolBlock(ideal_sdf, sanitize=True, removeHs=True)
        if frag is None or tmpl is None:
            return False
        mol = AllChem.AssignBondOrdersFromTemplate(tmpl, frag)
        Chem.MolToMolFile(mol, str(d / f"{pdb_id}_ligand.sdf"))
        return True
    except Exception:
        pass
    ## Genuine fallback: covalent adducts, alternate locations and partially
    ## modelled ligands where the template cannot match. The geometry is still
    ## correct; the bond orders are inferred. Recorded separately so the rate is
    ## visible rather than assumed.
    try:
        frag = Chem.MolFromPDBBlock("\n".join(het) + "\nEND\n", sanitize=False)
        if frag is None:
            return False
        Chem.MolToMolFile(frag, str(d / f"{pdb_id}_ligand.sdf"))
        (d / "BOND_ORDERS_INFERRED").write_text(
            "template matching failed; bond orders come from geometry, not the "
            "chemical component definition\n")
        return True
    except Exception:
        return False


def acquire_rcsb_crystals(limit: int | None = None, types=("Kd", "Ki"),
                          workers: int = 8) -> None:
    out = RAW / "rcsb_crystals"
    out.mkdir(parents=True, exist_ok=True)
    aff_p = out / "affinities.csv"

    if aff_p.exists():
        aff = pd.read_csv(aff_p)
        log(f"  {len(aff):,} affinity rows cached")
    else:
        log("  querying RCSB for entries with a Kd/Ki annotation")
        ids = rcsb_entries_with_affinity(types)
        log(f"  {len(ids):,} entries; fetching affinity records")
        aff = pd.DataFrame(_affinity_rows(ids, types))
        ## Atomic: write then rename. A crash part-way through a direct write leaves
        ## a TRUNCATED affinities.csv, and the `if aff_p.exists()` branch above would
        ## silently trust it on the next run - a quietly smaller crystal set with no
        ## error anywhere.
        tmp = aff_p.with_suffix(".csv.part")
        aff.to_csv(tmp, index=False)
        tmp.replace(aff_p)
    if aff.empty:
        raise SystemExit("RCSB returned no Kd/Ki affinity rows")

    ## Drop physically impossible values BEFORE choosing the best per entry.
    ## Sorting ascending to keep the tightest binder puts any zero or absurdly
    ## small annotation at the very top, so an unfiltered `0 nM` silently becomes
    ## the representative measurement for its structure - which is how all 26
    ## complexes of the first test run ended up written as `Kd=0nM` at pK 18 and
    ## then dropped by the corpus filter.
    n0 = len(aff)
    aff = aff.dropna(subset=["comp_id", "value_nM"])
    ## pK in (0, 14) is the same window the corpus enforces: 1e-5 nM (pK 14, about
    ## the biotin-streptavidin limit) to 1e9 nM (pK 0, i.e. 1 M).
    aff = aff[(aff.value_nM > 1e-5) & (aff.value_nM < 1e9)]
    if len(aff) < n0:
        log(f"    dropped {n0 - len(aff):,} annotations outside 0 < pK < 14 "
            "(zero, negative or unit-error values)")
    ## One row per entry: the tightest-binding annotated ligand. An entry with two
    ## annotated ligands is two different measurements sharing one structure, and
    ## the loader keys on pdb_id.
    aff = aff.sort_values("value_nM").drop_duplicates("pdb_id")
    todo = [r for r in aff.itertuples()
            if not (out / r.pdb_id / f"{r.pdb_id}_ligand.sdf").exists()]
    if limit:
        todo = todo[:limit]
    log(f"  {len(aff):,} complexes, {len(todo):,} still to download")

    ok, fail = 0, 0
    t0 = last_log = time.time()
    last_n = 0
    ## Downloads run in a thread pool; every RDKit call stays on the main thread.
    ## ex.map preserves order and streams, so writes overlap the next fetches
    ## without putting RDKit anywhere near a worker thread.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, got in enumerate(ex.map(_fetch_complex, todo), 1):
            r, pdb_text, ideal = got
            if pdb_text is None or ideal is None:
                fail += 1
            else:
                try:
                    ok += _write_complex(out, r.pdb_id, r.comp_id, pdb_text, ideal)
                except Exception:
                    fail += 1
            ## Time-based, not every-N-items. With `if n % 100 == 0` a job that
            ## slows to a crawl prints nothing for hours and the last ETA it did
            ## print is stale by exactly that long - which is how a 31-hour run
            ## looked like "eta 35 min" for three hours.
            now = time.time()
            if now - last_log > 30 or n == len(todo):
                recent = (n - last_n) / max(now - last_log, 1e-6)
                overall = n / max(now - t0, 1e-6)
                ## ETA from the RECENT rate: throughput on a shared link changes,
                ## and a cumulative average hides a slowdown for a long time.
                eta = (len(todo) - n) / max(recent, 1e-6) / 60
                log(f"    {n:,}/{len(todo):,}  {recent:.2f}/s now "
                    f"({overall:.2f}/s avg)  eta {eta:.0f} min  "
                    f"[{ok} ok, {fail} failed]")
                last_log, last_n = now, n

    ## A PDBbind-format index, so the existing loader parses this unchanged.
    built = sorted(p.parent.name for p in out.glob("*/*_ligand.sdf"))
    have = aff[aff.pdb_id.isin(built)]
    idx = out / "INDEX_general_PL_data.rcsb"
    lines, skipped_fmt = [], 0
    for r in have.itertuples():
        res = f"{r.resolution:.2f}" if pd.notna(r.resolution) else "2.00"
        yr = r.year if str(r.year).isdigit() else "2000"
        pk = 9.0 - math.log10(float(r.value_nM))
        ## `:g` switches to scientific notation for small numbers ("1e-09"), and the
        ## index grammar's value field is [\d.]+ - an exponent makes the line
        ## unparseable. `.6g` keeps ordinary decimals exact; anything that still
        ## needs an exponent is out of the sane range anyway and is skipped.
        val = f"{float(r.value_nM):.6g}"
        if not val.replace(".", "").isdigit():
            skipped_fmt += 1
            continue
        lines.append(f"{r.pdb_id}  {res}  {yr}  {pk:.2f}  "
                     f"{r.type}={val}nM  // RCSB/{r.provenance}")
    idx.write_text("# RCSB open crystal set, built by scripts/phase1a_acquire.py\n"
                   "# PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference\n"
                   + "\n".join(lines) + "\n")

    ## Prove the index we just wrote is readable by the loader that will read it.
    ## An index that parses to zero rows looks identical to "no data" downstream.
    sys.path.insert(0, str(ROOT / "src"))
    from pfpl.data.sources.pdbbind import parse_index
    parsed = parse_index(idx)
    if len(parsed) < len(lines):
        log(f"    WARNING: wrote {len(lines):,} index lines but only "
            f"{len(parsed):,} parse back")
    inferred = len(list(out.glob("*/BOND_ORDERS_INFERRED")))
    record("rcsb_crystals", dir=str(out), complexes=len(built),
           index_parses=int(len(parsed)),
           bond_orders_inferred=inferred,
           affinity_rows=int(len(aff)), index=str(idx), failed=fail,
           types=aff["type"].value_counts().to_dict(),
           provenance=aff["provenance"].value_counts().to_dict(),
           note="open substitute for PDBbind; PDBbind-format layout so the same "
                "loader and geometry code consume it unchanged")
    log(f"  RCSB crystals: {len(built):,} complexes -> {out}")
    if inferred:
        log(f"    {inferred:,} ({inferred / max(len(built), 1):.1%}) have INFERRED bond "
            "orders (template match failed); their ligand chemistry is approximate")
    log(f"  point the loader at it:  -o data.sources.pdbbind.path={out}")


# ------------------------------------------------------------------ PDBbind

def stage_pdbbind(tarballs, dest: str | None = None) -> None:
    """Unpack PDBbind tarballs YOU downloaded into the layout the loader expects.

    This is the legitimate half of the automation. PDBbind is behind a registration
    gate, so the download itself is yours to do under your own licence - but once
    the tarballs are on disk, working out that the index goes at the top level
    while the complexes need `--strip-components=1` is just fiddly, and getting it
    wrong produces a tree the loader silently finds nothing in.
    """
    import tarfile
    d = Path(dest).expanduser() if dest else RAW / "pdbbind"
    d.mkdir(parents=True, exist_ok=True)
    for t in tarballs:
        tp = Path(t).expanduser()
        if not tp.exists():
            raise SystemExit(f"{tp} not found")
        log(f"  unpacking {tp.name} ({human(tp.stat().st_size)})")
        with tarfile.open(tp) as tf:
            names = tf.getnames()[:50]
            ## The complex tarballs nest everything under a single top directory
            ## (v2020-refined/, refined-set/); the index tarball does not.
            tops = {n.split("/")[0] for n in names if "/" in n}
            strip = len(tops) == 1 and not any(
                n.startswith("INDEX") for n in names)
            if strip:
                top = tops.pop()
                for m in tf.getmembers():
                    if m.name.startswith(top + "/"):
                        m.name = m.name[len(top) + 1:]
                        if m.name:
                            tf.extract(m, d, filter="data")
            else:
                tf.extractall(d, filter="data")
    check_pdbbind(str(d))


def check_pdbbind(pdbbind_dir: str | None) -> None:
    """Never downloaded - licensed, registration required, not redistributable.

    PDBbind does three jobs here and it is worth being explicit about all three,
    because the user's plan is to pretrain on the big corpus and fine-tune on it:

      1. Kd/Ki LABELS for the corpus, at source weight 1.0 (Section 5.2)
      2. distogram + pocket GEOMETRY for Stage 1 and the anchor (Section 2.1)
      3. the Stage 3 crystal fine-tune set (Section 8)
    """
    d = Path(pdbbind_dir).expanduser() if pdbbind_dir else RAW / "pdbbind"
    ## Use the loader's own finder, not a second copy of the glob here: the two
    ## drifted apart once already, so a correctly staged 2020R1 package reported
    ## "NOT staged" while the loader would have read it fine.
    sys.path.insert(0, str(ROOT / "src"))
    from pfpl.data.sources.pdbbind import find_index
    idx = find_index(d) if d.exists() else None
    if idx is None and d.exists():
        ## Tarballs you downloaded but did not unpack (e.g. P-L.tar + index.tar.gz
        ## from PDBbind+ 2020R1). Unpack them in place so the config default
        ## `data/raw/pdbbind` just works. Complexes first, index LAST: the index is
        ## what marks the directory as staged, so an interrupted unpack is retried
        ## rather than half-used.
        tars = sorted((p for p in d.iterdir()
                       if p.is_file() and p.name.endswith((".tar", ".tar.gz", ".tgz"))),
                      key=lambda p: ("index" in p.name.lower(), p.name))
        if tars:
            log(f"  found {len(tars)} unpacked tarball(s) in {d}: "
                + ", ".join(p.name for p in tars))
            return stage_pdbbind([str(p) for p in tars], str(d))
    if idx is None:
        print(f"""
  PDBbind is NOT staged at {d}

  It sits behind a registration gate and its terms restrict redistribution, so
  this script does not fetch it (PLAN.md Section 0.4.9). THREE WAYS FORWARD:

  (a) OPEN SUBSTITUTE - no registration, but slow (thousands of structure
      downloads) and about a third the size. Only if (b) is impossible.
      RCSB annotates ~4,562 entries with a Kd or Ki value (provenance mostly
      BindingDB and PDBbind) and redistributes those annotations openly, and the
      structures themselves have always been open:

          python scripts/phase1a_acquire.py --rcsb-crystals

      It is written in PDBbind's own on-disk layout, so the same loader and the
      same geometry code consume it unchanged:

          -o data.sources.pdbbind.path={RAW}/rcsb_crystals

      Smaller and curated differently from PDBbind v2020 (~5.3k refined /
      ~19k general), so record the counts in DATA_MANIFEST.md rather than
      assuming they match.

  (b) STAGE A COPY YOU DOWNLOADED. RECOMMENDED. Register at pdbbind-plus.org.cn,
      download the protein-ligand set and the index, and drop the tarballs in
      {d} - the next run of this script unpacks them in place.

  (c) SKIP IT. WHAT THAT COSTS - three separate things, not one:
    1. its Kd/Ki labels (~5k crystal-linked pairs at source weight 1.0)
    2. the distogram ANCHOR. Section 2.1 lists it as a REQUIRED mitigation;
       without it the affinity loss converges on an over-attractive,
       geometrically impossible solution and the physics claim is unsupported.
    3. the Stage 3 crystal fine-tune, which is the last stage of the curriculum.

  Wave 1 and every baseline run fine without it.""")
        record("pdbbind", staged=False, expected_at=str(d))
        return
    n = sum(1 for _ in idx.open()) if idx.exists() else 0
    record("pdbbind", staged=True, dir=str(d), index=str(idx), index_lines=n)
    log(f"  PDBbind: staged at {d}, index {idx.name} ({n:,} lines)")


# -------------------------------------------------------------------- status

def status() -> None:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    print("=" * 78)
    print(f"RAW DATA  {RAW}")
    print("=" * 78)
    for name in ("chembl", "uniprot", "bindingdb", "papyrus", "iuphar",
                 "rcsb_crystals", "pdbbind"):
        e = m.get(name)
        if not e:
            print(f"  {name:<12} not acquired")
            continue
        bits = [f"{k}={v}" for k, v in e.items()
                if k in ("activities", "bytes", "targets_with_sequence", "staged",
                         "index_lines", "sequences", "unresolved", "complexes")]
        print(f"  {name:<12} {e['acquired']}  " + "  ".join(bits))
    ck = RAW / "chembl" / "checkpoint.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        if not c.get("done"):
            print(f"\n  ChEMBL is PARTIAL ({c.get('offset', 0):,} of "
                  f"{c.get('total', '?')}). Re-run --chembl to resume.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every network source")
    ap.add_argument("--chembl", action="store_true")
    ap.add_argument("--bindingdb", action="store_true")
    ap.add_argument("--papyrus", action="store_true")
    ap.add_argument("--iuphar", action="store_true")
    ap.add_argument("--uniprot", action="store_true",
                    help="resolve accessions -> sequences (ChEMBL and IUPHAR carry "
                         "accessions only)")
    ap.add_argument("--rcsb-crystals", action="store_true",
                    help="OPEN substitute for PDBbind: crystal complexes with Kd/Ki "
                         "from RCSB, no registration, written in PDBbind layout")
    ap.add_argument("--rcsb-limit", type=int, default=None,
                    help="debug: cap how many complexes to download")
    ap.add_argument("--rcsb-workers", type=int, default=8,
                    help="parallel download workers (default 8; downloads are "
                         "pure I/O, and RCSB is a shared public resource)")
    ap.add_argument("--pdbbind-tarball", action="append", default=[],
                    metavar="TGZ",
                    help="unpack a PDBbind tarball YOU downloaded (repeatable)")
    ap.add_argument("--pdbbind-dir", default=None,
                    help="where a licensed PDBbind copy is staged (checked, not fetched)")
    ap.add_argument("--bindingdb-url", default=BINDINGDB_URL)
    ap.add_argument("--limit-rows", type=int, default=None, help="debug: cap ChEMBL")
    ap.add_argument("--force", action="store_true", help="ignore the ChEMBL checkpoint")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    paths.ensure()
    if a.status:
        return status()

    any_net = (a.all or a.chembl or a.bindingdb or a.papyrus or a.iuphar
               or a.uniprot or a.rcsb_crystals)
    if not any_net and not a.pdbbind_dir and not a.pdbbind_tarball:
        ap.print_help()
        print("\nNothing selected. Start with:  --all")
        return

    print("=" * 78)
    print("PHASE 1a - acquire raw Kd/Ki sources")
    print(f"  into {RAW}")
    print("=" * 78)

    ## Each source runs in isolation. Before this, one failure (Papyrus renaming its
    ## Zenodo files) aborted --all before ChEMBL, BindingDB or IUPHAR even started,
    ## and cost a whole allocation. Now the rest still run and the job fails at the
    ## END, naming every source that did not make it.
    ##
    ## --rcsb-crystals is deliberately NOT part of --all: several thousand structure
    ## downloads from a shared public server, and redundant once a licensed PDBbind
    ## copy is staged (12,000 crystal Kd/Ki pairs vs ~4,500).
    steps = [
        (a.all or a.papyrus, "papyrus", acquire_papyrus),
        (a.all or a.iuphar, "iuphar", acquire_iuphar),
        (a.all or a.bindingdb, "bindingdb", lambda: acquire_bindingdb(a.bindingdb_url)),
        (a.all or a.chembl, "chembl (resumable; this is the long one)",
         lambda: acquire_chembl(a.limit_rows, a.force)),
        (a.all or a.uniprot, "uniprot (sequences for every accession seen so far)",
         acquire_uniprot),
        (a.rcsb_crystals, "rcsb crystal complexes (open PDBbind substitute)",
         lambda: acquire_rcsb_crystals(a.rcsb_limit, workers=a.rcsb_workers)),
        (True, "pdbbind (licensed: staged from YOUR copy, never downloaded)",
         lambda: stage_pdbbind(a.pdbbind_tarball, a.pdbbind_dir) if a.pdbbind_tarball
         else check_pdbbind(a.pdbbind_dir)),
    ]
    t0 = time.time()
    failed = []
    for on, name, fn in steps:
        if not on:
            continue
        log(name)
        try:
            fn()
        except (Exception, SystemExit) as e:   # noqa: BLE001 - report, keep going
            failed.append(name.split()[0])
            log(f"  FAILED {name.split()[0]}: {e}")

    print(f"\n  acquired in {(time.time() - t0) / 60:.1f} min")
    status()
    if failed:
        raise SystemExit(f"\n{len(failed)} source(s) failed: {', '.join(failed)}. "
                         "Everything else is on disk; re-run just those, e.g.\n"
                         f"    python scripts/phase1a_acquire.py --{failed[0]}")
    print("\nNext:  python scripts/phase1_harmonize.py")
    print("  Phase 1b reads ONLY these local files - no loader touches the network.")


if __name__ == "__main__":
    main()
