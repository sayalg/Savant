#!/usr/bin/env python
"""
Step 02 - Ligand taxonomy for the candidate pool.

Two annotations per unique candidate ligand:
  * ClassyFire / ChemOnt structural lineage (kingdom > superclass > class > subclass)
  * ChEMBL identity: preferred name, max clinical phase, ATC code -> drug class

Mirrors ML_model/taxonomy/02_classify_ligands.py: every response is cached under
cache/, 200 and 404 are cached but transient errors are not, so re-running only
retries failures. ClassyFire hard-rate-limits (~70/min/IP), so this is slow and
meant to be run as a background job.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from rdkit import Chem, RDLogger

from common import CACHE, CANDIDATES, LIG_ANNOT, ensure_dirs, log

RDLogger.DisableLog("rdApp.*")

CF_CACHE = os.path.join(CACHE, "classyfire")
CHEMBL_CACHE = os.path.join(CACHE, "chembl_mol")
CF_BASES = ["https://cfb.fiehnlab.ucdavis.edu/entities/{}.json",
            "http://classyfire.wishartlab.com/entities/{}.json"]
CHEMBL_URL = ("https://www.ebi.ac.uk/chembl/api/data/molecule.json"
              "?molecule_structures__standard_inchi_key={}")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "savant-benchmark/1.0 (research)"})
    return s


# --------------------------------------------------------------------------- #
def fetch_classyfire(ik: str, s: requests.Session) -> dict:
    cf = os.path.join(CF_CACHE, ik + ".json")
    if os.path.exists(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    for attempt in range(6):
        base = CF_BASES[0] if attempt < 5 else CF_BASES[1]
        try:
            r = s.get(base.format(ik), timeout=30)
            if r.status_code == 200:
                data = {"status": "ok", "json": r.json()}
                json.dump(data, open(cf, "w"))
                return data
            if r.status_code == 404:
                data = {"status": "not_found"}
                json.dump(data, open(cf, "w"))
                return data
            if r.status_code == 429:
                time.sleep((2 ** attempt) + random.random() * 2)
                continue
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {"status": "error"}


def fetch_chembl(ik: str, s: requests.Session) -> dict:
    cf = os.path.join(CHEMBL_CACHE, ik + ".json")
    if os.path.exists(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    for attempt in range(4):
        try:
            r = s.get(CHEMBL_URL.format(ik), timeout=30)
            if r.status_code == 200:
                mols = r.json().get("molecules", [])
                m = mols[0] if mols else None
                data = {"status": "ok" if m else "not_found"}
                if m:
                    atc = (m.get("atc_classifications") or [])
                    data.update({
                        "molecule_chembl_id": m.get("molecule_chembl_id"),
                        "pref_name": m.get("pref_name"),
                        "max_phase": m.get("max_phase"),
                        "atc_code": atc[0] if atc else None,
                    })
                json.dump(data, open(cf, "w"))
                return data
            if r.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue
            if r.status_code == 404:
                json.dump({"status": "not_found"}, open(cf, "w"))
                return {"status": "not_found"}
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {"status": "error"}


# --------------------------------------------------------------------------- #
def parse_cf(d: dict) -> dict:
    if d.get("status") != "ok":
        return {"cf_status": d.get("status", "error")}
    j = d["json"]

    def name(k):
        v = j.get(k)
        return v.get("name") if isinstance(v, dict) else None

    inter = [n.get("name") for n in (j.get("intermediate_nodes") or []) if isinstance(n, dict)]
    lineage = [x for x in [name("kingdom"), name("superclass"), name("class"),
                           name("subclass"), *inter] if x]
    return {
        "cf_status": "ok",
        "kingdom": name("kingdom"),
        "superclass": name("superclass"),
        "cf_class": name("class"),
        "subclass": name("subclass"),
        "direct_parent": name("direct_parent"),
        "molecular_framework": j.get("molecular_framework"),
        "lineage": " > ".join(lineage),
    }


def main(threads: int, assemble_only: bool, from_selected: bool) -> None:
    ensure_dirs()
    for d in (CF_CACHE, CHEMBL_CACHE):
        os.makedirs(d, exist_ok=True)

    src = os.path.join(os.path.dirname(CANDIDATES), "selected.csv") if from_selected else CANDIDATES
    cand = pd.read_csv(src)
    smi = cand["canonical_smiles"].dropna().unique()
    log(f"unique ligands to annotate ({os.path.basename(src)}): {len(smi)}")

    rows = []
    for s in smi:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            ik = Chem.MolToInchiKey(m)
        except Exception:
            ik = None
        if ik:
            rows.append({"canonical_smiles": s, "inchikey": ik})
    keys = pd.DataFrame(rows).drop_duplicates("inchikey")
    log(f"resolved InChIKeys: {len(keys)}")

    iks = keys["inchikey"].tolist()
    cf_res, ch_res = {}, {}

    if not assemble_only:
        s = _session()
        # ClassyFire: modest concurrency, it rate-limits hard
        with ThreadPoolExecutor(max_workers=threads) as ex:
            fut = {ex.submit(fetch_classyfire, ik, s): ik for ik in iks}
            for n, f in enumerate(as_completed(fut), 1):
                cf_res[fut[f]] = f.result()
                if n % 250 == 0:
                    ok = sum(1 for v in cf_res.values() if v.get("status") == "ok")
                    log(f"  classyfire {n}/{len(iks)} (ok={ok})")
        log("classyfire done")

        with ThreadPoolExecutor(max_workers=min(8, threads * 2)) as ex:
            fut = {ex.submit(fetch_chembl, ik, s): ik for ik in iks}
            for n, f in enumerate(as_completed(fut), 1):
                ch_res[fut[f]] = f.result()
                if n % 500 == 0:
                    log(f"  chembl {n}/{len(iks)}")
        log("chembl done")
    else:
        for ik in iks:
            p = os.path.join(CF_CACHE, ik + ".json")
            cf_res[ik] = json.load(open(p)) if os.path.exists(p) else {"status": "missing"}
            p = os.path.join(CHEMBL_CACHE, ik + ".json")
            ch_res[ik] = json.load(open(p)) if os.path.exists(p) else {"status": "missing"}

    recs = []
    for ik in iks:
        r = {"inchikey": ik}
        r.update(parse_cf(cf_res.get(ik, {})))
        c = ch_res.get(ik, {})
        r.update({
            "molecule_chembl_id": c.get("molecule_chembl_id"),
            "chembl_pref_name": c.get("pref_name"),
            "max_phase": c.get("max_phase"),
            "atc_code": c.get("atc_code"),
            "atc_l1": (c.get("atc_code") or "")[:1] or None,
            "atc_l2": (c.get("atc_code") or "")[:3] or None,
        })
        recs.append(r)

    out = keys.merge(pd.DataFrame(recs), on="inchikey", how="left")
    out.to_csv(LIG_ANNOT, index=False)
    log(f"wrote {LIG_ANNOT} ({len(out)} ligands)")
    log(f"  ClassyFire ok: {(out.cf_status == 'ok').sum()} / {len(out)}")
    log(f"  in ChEMBL:     {out.molecule_chembl_id.notna().sum()} / {len(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--assemble", action="store_true", help="build CSV from cache only")
    ap.add_argument("--from-selected", action="store_true",
                    help="annotate only the ligands in work/selected.csv")
    a = ap.parse_args()
    main(a.threads, a.assemble, a.from_selected)
