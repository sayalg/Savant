#!/usr/bin/env python
"""
Step 02 - Classify ligands with ClassyFire (ChemOnt structural taxonomy).

For each unique ligand InChIKey, query the ClassyFire entity endpoint (a cached
lookup of already-classified structures) and store the full ChemOnt lineage:

    kingdom > superclass > class > subclass > [intermediate nodes] > direct_parent

Resumable & concurrent:
  * every InChIKey response is cached to cache/classyfire/<inchikey>.json
  * a 200 (classified) or 404 (not in ClassyFire) is cached; transient errors are
    NOT cached, so re-running retries only the failures.

Writes output/ligand_taxonomy.csv when all lookups are resolved (or on --assemble).
"""
import os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
CACHE = os.path.join(HERE, "cache", "classyfire")
os.makedirs(CACHE, exist_ok=True)

# The Fiehn Lab mirror serves the same ChemOnt classification as the original
# ClassyFire server (classyfire.wishartlab.com) but is not aggressively rate
# limited, so it is used as the primary endpoint with the original as fallback.
BASES = ["https://cfb.fiehnlab.ucdavis.edu/entities/{}.json",
         "http://classyfire.wishartlab.com/entities/{}.json"]
LEVELS = ["kingdom", "superclass", "class", "subclass"]


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "savant-taxonomy/1.0 (research)"})
    return s


def fetch(ik, s):
    """Return dict from cache or network. Caches 200/404, not transient errors."""
    cf = os.path.join(CACHE, ik + ".json")
    if os.path.exists(cf):
        try:
            return json.load(open(cf))
        except Exception:
            pass
    import random
    # ClassyFire (both the Fiehn mirror and the origin) rate-limits with HTTP 429.
    # Retry the SAME endpoint with exponential backoff + jitter; the origin server
    # is only used as a last-ditch fallback.
    for attempt in range(6):
        base = BASES[0] if attempt < 5 else BASES[1]
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
            # 429 / 5xx -> backoff
        except Exception:
            pass
        time.sleep(min(5.0, 1.0 * (attempt + 1)) + random.uniform(0, 0.5))
    return {"status": "error"}


def run(workers):
    lig = pd.read_csv(os.path.join(OUT, "unique_ligands.csv"))
    iks = [x for x in lig["inchikey"].dropna().unique().tolist()]
    todo = [ik for ik in iks if not os.path.exists(os.path.join(CACHE, ik + ".json"))]
    print(f"[02] {len(iks)} unique InChIKeys; {len(todo)} to fetch, "
          f"{len(iks) - len(todo)} cached", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sessions = [_session() for _ in range(workers)]
        futs = {ex.submit(fetch, ik, sessions[i % workers]): ik
                for i, ik in enumerate(todo)}
        for fu in as_completed(futs):
            done += 1
            if done % 250 == 0:
                print(f"[02]   {done}/{len(todo)} fetched", flush=True)
    print("[02] fetch pass complete", flush=True)
    assemble()


def _lineage(j):
    path = []
    for lvl in LEVELS:
        node = j.get(lvl)
        path.append(node.get("name") if node else None)
    inter = [n.get("name") for n in (j.get("intermediate_nodes") or [])]
    dp = j.get("direct_parent")
    dp = dp.get("name") if dp else None
    full = [p for p in path if p] + inter + ([dp] if dp else [])
    return path, dp, inter, full


def assemble():
    lig = pd.read_csv(os.path.join(OUT, "unique_ligands.csv"))
    rows = []
    n_ok = n_nf = n_err = n_missing = 0
    for _, r in lig.iterrows():
        ik = r["inchikey"]
        rec = {"ligand_id": r["ligand_id"], "inchikey": ik,
               "canonical_smiles": r["canonical_smiles"], "n_complexes": r["n_complexes"]}
        cf = os.path.join(CACHE, str(ik) + ".json") if isinstance(ik, str) else None
        if not cf or not os.path.exists(cf):
            rec["cf_status"] = "no_lookup"; n_missing += 1
        else:
            data = json.load(open(cf))
            st = data.get("status")
            if st == "ok":
                j = data["json"]
                path, dp, inter, full = _lineage(j)
                rec.update({"cf_status": "ok",
                            "kingdom": path[0], "superclass": path[1],
                            "cf_class": path[2], "subclass": path[3],
                            "direct_parent": dp,
                            "molecular_framework": (j.get("molecular_framework")),
                            "lineage": " > ".join(full)})
                n_ok += 1
            elif st == "not_found":
                rec["cf_status"] = "not_found"; n_nf += 1
            else:
                rec["cf_status"] = "error"; n_err += 1
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ligand_taxonomy.csv"), index=False)
    print(f"[02] wrote ligand_taxonomy.csv  ok={n_ok} not_found={n_nf} "
          f"error={n_err} no_lookup={n_missing} total={len(df)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--assemble", action="store_true", help="only rebuild CSV from cache")
    a = ap.parse_args()
    if a.assemble:
        assemble()
    else:
        run(a.workers)
