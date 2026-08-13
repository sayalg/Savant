#!/usr/bin/env python
"""
Step 03 - Protein taxonomy for the candidate targets.

BindingDB hands us UniProt accessions directly, so unlike the training-set
taxonomy run (which needed an RCSB sequence search to get from PDB-construct
sequences to accessions) this is a straight lookup:

  * UniProt REST  -> protein name, gene, organism, EC, GO-MF, protein family
  * ChEMBL        -> target class tree (chembl_l1..l6), walked parent-by-parent

Field names and level derivations match ML_model/taxonomy/04+05 so benchmark
labels are directly comparable to the training-set labels.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from common import CACHE, CANDIDATES, PROT_ANNOT, ensure_dirs, log

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
UFIELDS = "accession,protein_name,organism_name,gene_primary,ec,go_f,protein_families"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

U_CACHE = os.path.join(CACHE, "uniprot")
C_CACHE = os.path.join(CACHE, "chembl_target")
CLS_CACHE = os.path.join(CACHE, "chembl_classes")

EC_L1 = {"1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
         "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases"}


def _sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "savant-benchmark/1.0 (research)"})
    return s


# ---------------- UniProt ----------------
def annotate_uniprot(accs: list[str]) -> pd.DataFrame:
    s = _sess()
    rows, CH = [], 80
    for i in range(0, len(accs), CH):
        chunk = accs[i:i + CH]
        cf = os.path.join(U_CACHE, f"batch_{i}.tsv")
        if os.path.exists(cf):
            txt = open(cf, encoding="utf-8").read()
        else:
            q = " OR ".join(f"accession:{a}" for a in chunk)
            txt = ""
            for attempt in range(4):
                try:
                    r = s.get(UNIPROT, params={"query": q, "fields": UFIELDS,
                                               "format": "tsv", "size": 500}, timeout=60)
                    r.raise_for_status()
                    txt = r.text
                    open(cf, "w", encoding="utf-8").write(txt)
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
        if not txt.strip():
            continue
        lines = txt.strip().split("\n")
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            d = dict(zip(hdr, ln.split("\t")))
            rows.append({"uniprot_id": d.get("Entry"),
                         "protein_name": d.get("Protein names"),
                         "organism": d.get("Organism"),
                         "gene": d.get("Gene Names (primary)"),
                         "ec": d.get("EC number"),
                         "go_mf": d.get("Gene Ontology (molecular function)"),
                         "protein_families": d.get("Protein families")})
        log(f"  uniprot {min(i + CH, len(accs))}/{len(accs)}")
    return pd.DataFrame(rows).drop_duplicates("uniprot_id")


# ---------------- ChEMBL target class ----------------
def _class_node(pcid, s):
    cf = os.path.join(CLS_CACHE, f"{pcid}.json")
    if os.path.exists(cf):
        d = json.load(open(cf))
        return d.get("pref_name"), d.get("parent_id")
    r = s.get(f"{CHEMBL}/protein_classification/{pcid}.json", timeout=30)
    r.raise_for_status()
    j = r.json()
    json.dump({"pref_name": j.get("pref_name"), "parent_id": j.get("parent_id")}, open(cf, "w"))
    return j.get("pref_name"), j.get("parent_id")


def _class_path(pcid, s):
    names, seen, cur = [], set(), pcid
    while cur and cur not in seen:
        seen.add(cur)
        name, parent = _class_node(cur, s)
        if name and name.lower() != "protein classification":
            names.append(name)
        cur = parent
    return list(reversed(names))


def _chembl_class(acc, s):
    cf = os.path.join(C_CACHE, acc + ".json")
    if os.path.exists(cf):
        return json.load(open(cf))
    out = {"uniprot_id": acc, "chembl_target_id": None,
           "chembl_pref_name": None, "chembl_class_path": None}
    try:
        r = s.get(f"{CHEMBL}/target.json",
                  params={"target_components__accession": acc, "limit": 1}, timeout=30)
        tg = r.json().get("targets", [])
        if tg:
            out["chembl_target_id"] = tg[0]["target_chembl_id"]
            out["chembl_pref_name"] = tg[0].get("pref_name")
        rc = s.get(f"{CHEMBL}/target_component.json", params={"accession": acc}, timeout=30)
        pcids = [pc["protein_classification_id"]
                 for c in rc.json().get("target_components", [])
                 for pc in c.get("protein_classifications", [])]
        paths = []
        for pcid in pcids:
            p = _class_path(pcid, s)
            if p:
                paths.append(" > ".join(p))
        if paths:
            paths = sorted(set(paths), key=len, reverse=True)
            out["chembl_class_path"] = paths[0]
    except Exception:
        return out                      # transient failure: don't cache
    json.dump(out, open(cf, "w"))
    return out


def annotate_chembl(accs: list[str], workers: int) -> pd.DataFrame:
    sessions = [_sess() for _ in range(workers)]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_chembl_class, a, sessions[i % workers]): a
                for i, a in enumerate(accs)}
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if n % 100 == 0:
                log(f"  chembl {n}/{len(accs)}")
    return pd.DataFrame(rows).drop_duplicates("uniprot_id")


# ---------------- assemble ----------------
def main(workers: int) -> None:
    ensure_dirs()
    for d in (U_CACHE, C_CACHE, CLS_CACHE):
        os.makedirs(d, exist_ok=True)

    cand = pd.read_csv(CANDIDATES)
    accs = sorted({str(a).split("-")[0].strip().upper()
                   for a in cand["uniprot_id"].dropna().unique()})
    log(f"unique candidate accessions: {len(accs)}")

    up = annotate_uniprot(accs)
    log(f"uniprot annotated: {len(up)}")
    ch = annotate_chembl(accs, workers)
    log(f"chembl annotated: {ch.chembl_class_path.notna().sum()} with a class path")

    df = up.merge(ch, on="uniprot_id", how="outer")

    def split_ch(p, i):
        if not isinstance(p, str):
            return None
        parts = [x.strip() for x in p.split(">")]
        return parts[i] if i < len(parts) else None

    for i in range(6):
        df[f"chembl_l{i+1}"] = df["chembl_class_path"].map(lambda p, i=i: split_ch(p, i))

    def first_ec(e):
        if not isinstance(e, str) or not e.strip():
            return None
        return e.split(";")[0].strip()

    df["ec_primary"] = df["ec"].map(first_ec)
    df["ec_l1_name"] = df["ec_primary"].map(
        lambda e: EC_L1.get(e.split(".")[0]) if isinstance(e, str) else None)

    def fam(p, i):
        if not isinstance(p, str):
            return None
        parts = [x.strip() for x in p.split(",")]
        return parts[i] if i < len(parts) else None

    df["family_l1"] = df["protein_families"].map(lambda p: fam(p, 0))
    df["family_l2"] = df["protein_families"].map(lambda p: fam(p, 1))

    df.to_csv(PROT_ANNOT, index=False)
    log(f"wrote {PROT_ANNOT} ({len(df)} accessions)")
    log(f"  chembl_l1 coverage: {df.chembl_l1.notna().sum()}/{len(df)}")
    log(f"  family_l1 coverage: {df.family_l1.notna().sum()}/{len(df)}")
    log(f"  organism coverage:  {df.organism.notna().sum()}/{len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args().workers)
