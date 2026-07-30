#!/usr/bin/env python
"""
Step 08 - Curated ligand mechanism / drug class from ChEMBL (high precision,
partial coverage). For every unique ligand InChIKey:

  molecule lookup  -> molecule_chembl_id, pref_name, max_phase, ATC code(s)
  mechanism lookup -> action_type (INHIBITOR/AGONIST/ANTAGONIST/...), MoA text
  ATC resolve      -> WHO ATC drug-class names (level 2/3/4)

All cached & resumable. Batched (InChIKeys / molecule ids in groups) to keep the
request count low. Writes output/ligand_moa.csv and merges the curated columns
into complexes_annotated.csv.
"""
import os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
CDIR = os.path.join(HERE, "cache")
MOL = os.path.join(CDIR, "chembl_mol"); os.makedirs(MOL, exist_ok=True)
MECH = os.path.join(CDIR, "chembl_mech"); os.makedirs(MECH, exist_ok=True)
ATC = os.path.join(CDIR, "chembl_atc"); os.makedirs(ATC, exist_ok=True)
API = "https://www.ebi.ac.uk/chembl/api/data"


def _sess():
    s = requests.Session(); s.headers.update({"User-Agent": "savant-taxonomy/1.0"}); return s


def _get(s, url, params):
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


# ---------- phase A: InChIKey -> molecule ----------
def fetch_molecules(iks, workers):
    todo = [ik for ik in iks if not os.path.exists(os.path.join(MOL, ik + ".json"))]
    print(f"[08A] {len(iks)} IKs, {len(todo)} to fetch", flush=True)
    groups = [todo[i:i + 30] for i in range(0, len(todo), 30)]

    def work(grp):
        s = _sess()
        js = _get(s, f"{API}/molecule.json",
                  {"molecule_structures__standard_inchi_key__in": ",".join(grp),
                   "limit": 100})
        found = {}
        for m in (js or {}).get("molecules", []):
            ik = (m.get("molecule_structures") or {}).get("standard_inchi_key")
            if ik:
                found[ik] = {"molecule_chembl_id": m.get("molecule_chembl_id"),
                             "pref_name": m.get("pref_name"),
                             "max_phase": m.get("max_phase"),
                             "atc": m.get("atc_classifications") or []}
        # write one cache file per requested IK (present or not)
        for ik in grp:
            json.dump(found.get(ik, {"molecule_chembl_id": None}),
                      open(os.path.join(MOL, ik + ".json"), "w"))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in as_completed({ex.submit(work, g): g for g in groups}):
            done += 1
            if done % 20 == 0:
                print(f"[08A]   {done}/{len(groups)} batches", flush=True)


# ---------- phase B: molecule -> mechanism ----------
def fetch_mechanisms(cids, workers):
    todo = [c for c in cids if not os.path.exists(os.path.join(MECH, c + ".json"))]
    print(f"[08B] {len(cids)} molecules with ChEMBL id, {len(todo)} to fetch MoA", flush=True)
    groups = [todo[i:i + 30] for i in range(0, len(todo), 30)]

    def work(grp):
        s = _sess()
        js = _get(s, f"{API}/mechanism.json",
                  {"molecule_chembl_id__in": ",".join(grp), "limit": 200})
        by = {}
        for m in (js or {}).get("mechanisms", []):
            cid = m.get("molecule_chembl_id")
            by.setdefault(cid, []).append({"action_type": m.get("action_type"),
                                           "moa": m.get("mechanism_of_action"),
                                           "target": m.get("target_chembl_id")})
        for c in grp:
            json.dump(by.get(c, []), open(os.path.join(MECH, c + ".json"), "w"))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in as_completed({ex.submit(work, g): g for g in groups}):
            done += 1
            if done % 10 == 0:
                print(f"[08B]   {done}/{len(groups)} batches", flush=True)


# ---------- phase C: ATC code -> class names ----------
def fetch_atc(codes, workers):
    todo = [c for c in codes if not os.path.exists(os.path.join(ATC, c + ".json"))]
    print(f"[08C] {len(codes)} ATC codes, {len(todo)} to resolve", flush=True)

    def work(code):
        s = _sess()
        js = _get(s, f"{API}/atc_class/{code}.json", {})
        d = {}
        if js:
            d = {"l2": js.get("level2_description"), "l3": js.get("level3_description"),
                 "l4": js.get("level4_description"), "who_name": js.get("who_name")}
        json.dump(d, open(os.path.join(ATC, code + ".json"), "w"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed({ex.submit(work, c): c for c in todo}))


def assemble():
    lig = pd.read_csv(os.path.join(OUT, "ligand_taxonomy.csv"))
    rows = []
    for _, r in lig.iterrows():
        ik = r["inchikey"]
        rec = {"ligand_id": r["ligand_id"], "inchikey": ik, "n_complexes": r["n_complexes"]}
        mf = os.path.join(MOL, str(ik) + ".json") if isinstance(ik, str) else None
        if mf and os.path.exists(mf):
            mol = json.load(open(mf))
            cid = mol.get("molecule_chembl_id")
            rec["molecule_chembl_id"] = cid
            rec["chembl_pref_name"] = mol.get("pref_name")
            rec["max_phase"] = mol.get("max_phase")
            # mechanism
            if cid and os.path.exists(os.path.join(MECH, cid + ".json")):
                mech = json.load(open(os.path.join(MECH, cid + ".json")))
                if mech:
                    rec["action_type"] = ";".join(sorted({m["action_type"] for m in mech if m.get("action_type")}))
                    rec["moa_text"] = " | ".join(sorted({m["moa"] for m in mech if m.get("moa")}))
            # ATC (first code)
            atcs = mol.get("atc") or []
            if atcs:
                code = atcs[0]
                rec["atc_code"] = code
                af = os.path.join(ATC, code + ".json")
                if os.path.exists(af):
                    a = json.load(open(af))
                    rec["atc_l2"] = a.get("l2"); rec["atc_l3"] = a.get("l3"); rec["atc_l4"] = a.get("l4")
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ligand_moa.csv"), index=False)
    n = len(df)
    print(f"[08] wrote ligand_moa.csv ({n} ligands)")
    for col in ["molecule_chembl_id", "action_type", "atc_l4", "max_phase"]:
        if col in df:
            c = df[col].notna().sum()
            print(f"     {col:20s}: {c} ligands ({100*c/n:.1f}%)")

    # merge into complexes_annotated
    ca = pd.read_csv(os.path.join(OUT, "complexes_annotated.csv"))
    keep = [c for c in ["ligand_id", "molecule_chembl_id", "chembl_pref_name",
                        "max_phase", "action_type", "moa_text", "atc_l2", "atc_l3", "atc_l4"]
            if c in df.columns]
    ca = ca.drop(columns=[c for c in keep if c != "ligand_id" and c in ca.columns], errors="ignore")
    ca = ca.merge(df[keep], on="ligand_id", how="left")
    ca.to_csv(os.path.join(OUT, "complexes_annotated.csv"), index=False)
    both = ((ca["action_type"].notna())).sum()
    print(f"[08] merged into complexes_annotated.csv; curated action_type on "
          f"{both}/{len(ca)} complexes ({100*both/len(ca):.1f}%)")


def run(workers):
    lig = pd.read_csv(os.path.join(OUT, "ligand_taxonomy.csv"))
    iks = lig["inchikey"].dropna().unique().tolist()
    fetch_molecules(iks, workers)
    # collect chembl ids + atc codes
    cids, codes = set(), set()
    for ik in iks:
        mf = os.path.join(MOL, ik + ".json")
        if os.path.exists(mf):
            m = json.load(open(mf))
            if m.get("molecule_chembl_id"):
                cids.add(m["molecule_chembl_id"])
            for c in (m.get("atc") or []):
                codes.add(c)
    fetch_mechanisms(sorted(cids), workers)
    fetch_atc(sorted(codes), workers)
    assemble()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    assemble() if a.assemble else run(a.workers)
