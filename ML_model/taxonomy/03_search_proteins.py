#!/usr/bin/env python
"""
Step 03 - Map each unique protein sequence to a PDB polymer entity via the RCSB
sequence-search service (server-side mmseqs2). Tolerant to point mutations and
processed chains, so drug-target variants (e.g. HIV-protease mutants) still map.

Tries identity cutoff 0.9, then falls back to 0.7, then 0.4. Caches one JSON per
sequence (keyed by crc64) so the run is fully resumable and concurrent.

Writes output/protein_pdb_map.csv:
    protein_id, crc64, seq_len, n_complexes, pdb_entity, identity, cutoff, status
"""
import os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
CACHE = os.path.join(HERE, "cache", "rcsb_search")
os.makedirs(CACHE, exist_ok=True)

URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CUTOFFS = [0.9, 0.7, 0.4]
MIN_LEN = 20


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "savant-taxonomy/1.0 (research)"})
    return s


def _query(seq, cutoff, s):
    q = {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 1, "identity_cutoff": cutoff,
                                  "sequence_type": "protein", "value": seq}},
         "return_type": "polymer_entity",
         "request_options": {"scoring_strategy": "sequence",
                             "results_verbosity": "verbose",
                             "paginate": {"start": 0, "rows": 1}}}
    r = s.get(URL, params={"json": json.dumps(q)}, timeout=60)
    if r.status_code == 204:
        return None
    r.raise_for_status()
    js = r.json()
    rs = js.get("result_set") or []
    if not rs:
        return None
    top = rs[0]
    ident = None
    try:
        ident = top["services"][0]["nodes"][0]["match_context"][0]["sequence_identity"]
    except Exception:
        pass
    return {"pdb_entity": top["identifier"], "identity": ident}


def search_one(row, s):
    crc = row["crc64"]
    cf = os.path.join(CACHE, crc + ".json")
    if os.path.exists(cf):
        return
    seq = row["sequence"]
    if not isinstance(seq, str) or len(seq) < MIN_LEN:
        json.dump({"status": "too_short"}, open(cf, "w")); return
    for cutoff in CUTOFFS:
        for attempt in range(3):
            try:
                hit = _query(seq, cutoff, s)
                if hit:
                    hit.update({"status": "ok", "cutoff": cutoff})
                    json.dump(hit, open(cf, "w"))
                    return
                break  # valid empty response -> lower the cutoff
            except Exception:
                time.sleep(2 * (attempt + 1))
    json.dump({"status": "no_hit"}, open(cf, "w"))


def run(workers):
    prot = pd.read_csv(os.path.join(OUT, "unique_proteins.csv"))
    todo = [r for _, r in prot.iterrows()
            if not os.path.exists(os.path.join(CACHE, r["crc64"] + ".json"))]
    print(f"[03] {len(prot)} unique proteins; {len(todo)} to search, "
          f"{len(prot) - len(todo)} cached", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sessions = [_session() for _ in range(workers)]
        futs = {ex.submit(search_one, r, sessions[i % workers]): i
                for i, r in enumerate(todo)}
        for _ in as_completed(futs):
            done += 1
            if done % 100 == 0:
                print(f"[03]   {done}/{len(todo)} searched", flush=True)
    print("[03] search pass complete", flush=True)
    assemble()


def assemble():
    prot = pd.read_csv(os.path.join(OUT, "unique_proteins.csv"))
    rows = []
    from collections import Counter
    cnt = Counter()
    for _, r in prot.iterrows():
        cf = os.path.join(CACHE, r["crc64"] + ".json")
        rec = {"protein_id": r["protein_id"], "crc64": r["crc64"],
               "seq_len": r["seq_len"], "n_complexes": r["n_complexes"]}
        if os.path.exists(cf):
            d = json.load(open(cf))
            rec.update({"pdb_entity": d.get("pdb_entity"), "identity": d.get("identity"),
                        "cutoff": d.get("cutoff"), "status": d.get("status")})
            cnt[d.get("status")] += 1
        else:
            rec["status"] = "pending"; cnt["pending"] += 1
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "protein_pdb_map.csv"), index=False)
    print(f"[03] wrote protein_pdb_map.csv  status counts: {dict(cnt)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    assemble() if a.assemble else run(a.workers)
