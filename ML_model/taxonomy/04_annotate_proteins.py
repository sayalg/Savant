#!/usr/bin/env python
"""
Step 04 - Annotate the mapped PDB entities with taxonomy.

Consumes output/protein_pdb_map.csv (from step 03) and produces three cached,
resumable intermediate tables under output/:

  entity_annotation.csv   per unique PDB polymer entity:
        pdb_entity, uniprot_id, description, organism, pfam, pfam_names,
        interpro, interpro_names, go_all
  uniprot_annotation.csv  per unique UniProt accession:
        uniprot_id, protein_name, organism, gene, ec, go_mf, protein_families
  chembl_class.csv        per unique UniProt accession:
        uniprot_id, chembl_target_id, chembl_pref_name, chembl_class_path

Run after step 03 has produced protein_pdb_map.csv (can be re-run on partial
results). Everything is cached so re-runs only fetch what is missing.
"""
import os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
CDIR = os.path.join(HERE, "cache")
E_CACHE = os.path.join(CDIR, "rcsb_entity"); os.makedirs(E_CACHE, exist_ok=True)
U_CACHE = os.path.join(CDIR, "uniprot"); os.makedirs(U_CACHE, exist_ok=True)
C_CACHE = os.path.join(CDIR, "chembl"); os.makedirs(C_CACHE, exist_ok=True)

DATA_GQL = "https://data.rcsb.org/graphql"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def _sess():
    s = requests.Session(); s.headers.update({"User-Agent": "savant-taxonomy/1.0"}); return s


# ---------------- 4a: RCSB entity annotation ----------------
_FRAG = """{ rcsb_polymer_entity{ pdbx_description }
  rcsb_polymer_entity_container_identifiers{ uniprot_ids }
  rcsb_entity_source_organism{ scientific_name }
  rcsb_polymer_entity_annotation{ type annotation_id name } }"""


def _gql_batch(entities, s):
    """entities: list of 'XXXX_N'. Returns {entity_id: parsed_dict}."""
    aliases = []
    for i, e in enumerate(entities):
        entry, ent = e.split("_")
        aliases.append(f'e{i}: polymer_entity(entry_id:"{entry}", entity_id:"{ent}") {_FRAG}')
    q = "{ " + " ".join(aliases) + " }"
    r = s.post(DATA_GQL, json={"query": q}, timeout=60)
    r.raise_for_status()
    data = r.json().get("data", {}) or {}
    out = {}
    for i, e in enumerate(entities):
        d = data.get(f"e{i}")
        out[e] = _parse_entity(d)
    return out


def _parse_entity(d):
    if not d:
        return {"status": "no_data"}
    desc = (d.get("rcsb_polymer_entity") or {}).get("pdbx_description")
    ups = (d.get("rcsb_polymer_entity_container_identifiers") or {}).get("uniprot_ids") or []
    org = d.get("rcsb_entity_source_organism") or []
    org = org[0]["scientific_name"] if org else None
    ann = d.get("rcsb_polymer_entity_annotation") or []
    pfam = [(a["annotation_id"], a.get("name")) for a in ann if a["type"] == "Pfam"]
    ipr = [(a["annotation_id"], a.get("name")) for a in ann if a["type"] == "InterPro"]
    go = [(a["annotation_id"], a.get("name")) for a in ann if a["type"] == "GO"]
    return {"status": "ok", "description": desc,
            "uniprot_id": ups[0] if ups else None, "uniprot_ids": ups,
            "organism": org,
            "pfam": ";".join(p[0] for p in pfam),
            "pfam_names": ";".join(p[1] or "" for p in pfam),
            "interpro": ";".join(p[0] for p in ipr),
            "interpro_names": ";".join(p[1] or "" for p in ipr),
            "go_all": ";".join(f"{g[0]}|{g[1]}" for g in go)}


def annotate_entities(workers):
    pm = pd.read_csv(os.path.join(OUT, "protein_pdb_map.csv"))
    ents = sorted(pm.loc[pm["status"] == "ok", "pdb_entity"].dropna().unique().tolist())
    todo = [e for e in ents if not os.path.exists(os.path.join(E_CACHE, e + ".json"))]
    print(f"[04a] {len(ents)} unique entities, {len(todo)} to fetch", flush=True)
    # batch into groups of 40
    groups = [todo[i:i + 40] for i in range(0, len(todo), 40)]

    def work(grp):
        s = _sess()
        for attempt in range(3):
            try:
                res = _gql_batch(grp, s)
                for e, d in res.items():
                    json.dump(d, open(os.path.join(E_CACHE, e + ".json"), "w"))
                return
            except Exception:
                time.sleep(2 * (attempt + 1))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in as_completed({ex.submit(work, g): g for g in groups}):
            done += 1
            if done % 10 == 0:
                print(f"[04a]   {done}/{len(groups)} batches", flush=True)
    # assemble
    rows = []
    for e in ents:
        cf = os.path.join(E_CACHE, e + ".json")
        d = json.load(open(cf)) if os.path.exists(cf) else {"status": "pending"}
        d = {"pdb_entity": e, **d}
        d.pop("uniprot_ids", None)
        rows.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "entity_annotation.csv"), index=False)
    print(f"[04a] wrote entity_annotation.csv ({len(rows)} entities)", flush=True)


# ---------------- 4b: UniProt annotation ----------------
UFIELDS = "accession,protein_name,organism_name,gene_primary,ec,go_f,protein_families"


def annotate_uniprot():
    ea = pd.read_csv(os.path.join(OUT, "entity_annotation.csv"))
    accs = sorted(ea["uniprot_id"].dropna().unique().tolist())
    print(f"[04b] {len(accs)} unique UniProt accessions", flush=True)
    s = _sess()
    rows = []
    CH = 80
    for i in range(0, len(accs), CH):
        chunk = accs[i:i + CH]
        cf = os.path.join(U_CACHE, f"batch_{i}.tsv")
        if os.path.exists(cf):
            txt = open(cf, encoding="utf-8").read()
        else:
            q = " OR ".join(f"accession:{a}" for a in chunk)
            for attempt in range(4):
                try:
                    r = s.get(UNIPROT, params={"query": q, "fields": UFIELDS,
                                               "format": "tsv", "size": 500}, timeout=60)
                    r.raise_for_status(); txt = r.text
                    open(cf, "w", encoding="utf-8").write(txt); break
                except Exception:
                    time.sleep(2 * (attempt + 1)); txt = ""
        if not txt.strip():
            continue
        lines = txt.strip().split("\n"); hdr = lines[0].split("\t")
        for ln in lines[1:]:
            d = dict(zip(hdr, ln.split("\t")))
            rows.append({"uniprot_id": d.get("Entry"),
                         "protein_name": d.get("Protein names"),
                         "organism": d.get("Organism"),
                         "gene": d.get("Gene Names (primary)"),
                         "ec": d.get("EC number"),
                         "go_mf": d.get("Gene Ontology (molecular function)"),
                         "protein_families": d.get("Protein families")})
        if (i // CH) % 10 == 0:
            print(f"[04b]   {i+len(chunk)}/{len(accs)}", flush=True)
    pd.DataFrame(rows).drop_duplicates("uniprot_id").to_csv(
        os.path.join(OUT, "uniprot_annotation.csv"), index=False)
    print(f"[04b] wrote uniprot_annotation.csv ({len(rows)} rows)", flush=True)


# ---------------- 4c: ChEMBL target class ----------------
CLS_CACHE = os.path.join(CDIR, "chembl_classes"); os.makedirs(CLS_CACHE, exist_ok=True)


def _class_node(pcid, s):
    """Return (pref_name, parent_id) for a protein_classification id, cached."""
    cf = os.path.join(CLS_CACHE, f"{pcid}.json")
    if os.path.exists(cf):
        d = json.load(open(cf))
        return d.get("pref_name"), d.get("parent_id")
    rp = s.get(f"{CHEMBL}/protein_classification/{pcid}.json", timeout=30)
    rp.raise_for_status()
    j = rp.json()
    json.dump({"pref_name": j.get("pref_name"), "parent_id": j.get("parent_id")}, open(cf, "w"))
    return j.get("pref_name"), j.get("parent_id")


def _class_path(pcid, s):
    """Walk parent_id up to root; return root->leaf list of pref_names."""
    names = []
    seen = set()
    cur = pcid
    while cur and cur not in seen:
        seen.add(cur)
        name, parent = _class_node(cur, s)
        if name and name.lower() != "protein classification":  # skip synthetic root
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
        r = s.get(f"{CHEMBL}/target.json", params={"target_components__accession": acc,
                                                   "limit": 1}, timeout=30)
        tg = r.json().get("targets", [])
        if tg:
            out["chembl_target_id"] = tg[0]["target_chembl_id"]
            out["chembl_pref_name"] = tg[0].get("pref_name")
        rc = s.get(f"{CHEMBL}/target_component.json", params={"accession": acc}, timeout=30)
        comps = rc.json().get("target_components", [])
        pcids = []
        for c in comps:
            for pc in c.get("protein_classifications", []):
                pcids.append(pc["protein_classification_id"])
        paths = []
        for pcid in pcids:
            path = _class_path(pcid, s)
            if path:
                paths.append(" > ".join(path))
        if paths:
            # keep the longest (most specific) path as primary
            paths = sorted(set(paths), key=len, reverse=True)
            out["chembl_class_path"] = paths[0]
            out["chembl_class_paths_all"] = " || ".join(paths)
    except Exception:
        return out  # don't cache transient failure
    json.dump(out, open(cf, "w"))
    return out


def annotate_chembl(workers):
    ua = pd.read_csv(os.path.join(OUT, "uniprot_annotation.csv"))
    accs = sorted(ua["uniprot_id"].dropna().unique().tolist())
    print(f"[04c] {len(accs)} accessions -> ChEMBL", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        sess = [_sess() for _ in range(workers)]
        futs = {ex.submit(_chembl_class, a, sess[i % workers]): a
                for i, a in enumerate(accs)}
        done = 0
        for fu in as_completed(futs):
            rows.append(fu.result()); done += 1
            if done % 100 == 0:
                print(f"[04c]   {done}/{len(accs)}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "chembl_class.csv"), index=False)
    print(f"[04c] wrote chembl_class.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.step in ("a", "all"):
        annotate_entities(a.workers)
    if a.step in ("b", "all"):
        annotate_uniprot()
    if a.step in ("c", "all"):
        annotate_chembl(a.workers)
