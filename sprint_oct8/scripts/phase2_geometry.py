#!/usr/bin/env python3
"""Phase 2: structure harvesting and geometry labels (PLAN.md Section 8, Phase 2).

Extracts, per crystal complex: the SEQRES-derived sequence, pocket residues,
pseudo-site coordinates, and binned distogram targets on the same grid the physics
uses.

READ THIS BEFORE SKIPPING THE PHASE. Section 2.1 lists the supervised distogram
cross-entropy anchor as a REQUIRED mitigation, and Section 0.3.4 requires it to stay
on through every later stage. Without structural data there is no anchor; with no
anchor, the affinity loss is free to push probability mass toward each pair's energy
minimum, and the model converges on an over-attractive geometry that scores well on
pK and is not physics. A run in that state is a fusion model with extra steps, and
this script says so rather than letting it happen quietly.

Sources, in preference order: a local PDBbind tree, then any directory of
`<id>_protein.pdb` + `<id>_ligand.sdf`, then RCSB fetch for an explicit PDB list
(compute nodes have outbound network).

Sharded: one array task per slice of the complex list.

    python scripts/phase2_geometry.py --pdbbind /scratch/$USER/pdbbind_v2020
    python scripts/phase2_geometry.py --merge
    python scripts/phase2_geometry.py --selfcheck --pdbbind <dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import paths  # noqa: E402
from pfpl.data import schema  # noqa: E402
from pfpl.data.sources.pdbbind import complex_paths, find_index, parse_index  # noqa: E402
from pfpl.data.standardize import sequence_hash  # noqa: E402
from pfpl.features import geom_labels as G  # noqa: E402
from pfpl.features.cache import slurm_shard  # noqa: E402

OUT = paths.WORK / "geometry"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def discover(pdbbind: Path | None, pdb_dir: Path | None,
             allow_ic50: bool) -> pd.DataFrame:
    """List the complexes to process, with their affinity type already resolved."""
    if pdbbind:
        ## The loader's finder, not a local glob: 2020R1 names its index
        ## INDEX_general_PL.2020R1.lst, which the old `INDEX_general_PL_data.*`
        ## pattern missed.
        idx = find_index(Path(pdbbind))
        if idx is None:
            raise SystemExit(f"no PDBbind protein-ligand index under {pdbbind}")
        df = parse_index(idx)
        t = schema.normalize_type(df["raw_type"])
        df["type"] = t
        df["has_affinity_label"] = t.isin(schema.ALLOWED_TYPES)
        ## Section 4.2's IC50-complex policy. Default OFF: an IC50-only complex is
        ## excluded ENTIRELY, including from geometry pretraining, so no
        ## IC50-associated record enters training in any form.
        n_ic50 = int((~df.has_affinity_label).sum())
        if not allow_ic50:
            df = df[df.has_affinity_label]
            log(f"excluded {n_ic50:,} IC50/EC50-only complexes entirely "
                "(use_ic50_only_structures=false)")
        else:
            log(f"keeping {n_ic50:,} IC50-only complexes for GEOMETRY ONLY - their "
                "affinity loss is masked and the value is never read")
        root = Path(pdbbind)
        df["protein_file"] = [str(complex_paths(root, p)["protein"]) for p in df.pdb_id]
        df["ligand_file"] = [
            str(complex_paths(root, p)["ligand_sdf"]) for p in df.pdb_id]
        return df.reset_index(drop=True)

    if pdb_dir:
        d = Path(pdb_dir)
        rows = []
        for p in sorted(d.glob("**/*_protein.pdb")):
            pid = p.name.replace("_protein.pdb", "")
            lig = next((q for q in (p.with_name(f"{pid}_ligand.sdf"),
                                    p.with_name(f"{pid}_ligand.mol2")) if q.exists()),
                       None)
            if lig:
                rows.append(dict(pdb_id=pid, protein_file=str(p), ligand_file=str(lig),
                                 type=pd.NA, has_affinity_label=False))
        return pd.DataFrame(rows)

    raise SystemExit("give --pdbbind <dir> or --pdb-dir <dir>")


def process(rows: pd.DataFrame) -> tuple[list, list]:
    out, failures = [], []
    for n, r in enumerate(rows.itertuples(), 1):
        try:
            L = G.labels_for_complex(r.protein_file, r.ligand_file)
        except Exception as e:
            failures.append((r.pdb_id, f"{type(e).__name__}: {e}"))
            continue
        if L is None:
            failures.append((r.pdb_id, "no residues or no ligand atoms"))
            continue
        L["pdb_id"] = r.pdb_id
        L["sequence_hash"] = sequence_hash(L["sequence"])
        L["has_affinity_label"] = bool(getattr(r, "has_affinity_label", False))
        L["type"] = getattr(r, "type", pd.NA)
        L["value_nM"] = getattr(r, "value_nM", np.nan)
        L["relation"] = getattr(r, "relation", "=")
        out.append(L)
        if n % 200 == 0:
            log(f"  {n}/{len(rows)}  ({len(failures)} failed)")
    return out, failures


def selfcheck(rows: pd.DataFrame) -> None:
    """Phase 2 gate: re-compute the distogram from raw coordinates on a sample."""
    from pfpl.physics import terms as T
    sample = rows.sample(min(10, len(rows)), random_state=0)
    labels, fails = process(sample)
    if not labels:
        raise SystemExit(f"every sampled complex failed: {fails[:3]}")
    L = labels[0]
    d, b = L["dist"], L["dist_bins"]
    real = np.isfinite(d) & (d < T.D_MAX)
    lo = T.D_MIN + b[real] * T.D_STEP
    assert np.all((d[real] >= lo - 1e-6) & (d[real] < lo + T.D_STEP + 1e-6)), \
        "a binned distance does not fall inside its own bin"
    far = ~np.isfinite(d) | (d >= T.D_MAX)
    assert np.all(b[far] == T.FAR_BIN), "a far distance is not in the far bin"
    log(f"re-derived bins from raw coordinates on {L['pdb_id']}: consistent")
    for L in labels[:10]:
        frac = L["n_pocket"] / max(L["n_res"], 1)
        log(f"  {L['pdb_id']}  {L['n_res']:>4} res, pocket {L['n_pocket']:>3} "
            f"({frac:.1%}), {L['n_lig_sites']:>3} ligand sites")
        ## A "pocket" covering most of the protein means the ligand was parsed as
        ## something enormous, usually a second protein chain in the SDF.
        assert 0.0 < frac < 0.5, f"{L['pdb_id']}: implausible pocket fraction {frac:.1%}"
    print(json.dumps(G.contact_statistics(labels), indent=1))
    log("SELFCHECK PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbbind", default=None)
    ap.add_argument("--pdb-dir", default=None)
    ap.add_argument("--allow-ic50", action="store_true",
                    help="Section 4.2 use_ic50_only_structures: geometry ONLY, "
                         "affinity loss masked, the IC50 value never read")
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if a.merge:
        shards = sorted(OUT.glob("shard_*.pkl"))
        if not shards:
            raise SystemExit(f"no shards in {OUT} - the array has not run")
        labels, fails = [], []
        for s in shards:
            with open(s, "rb") as fh:
                d = pickle.load(fh)
            labels += d["labels"]
            fails += d["failures"]
        with open(OUT / "geometry.pkl", "wb") as fh:
            pickle.dump(labels, fh, protocol=4)
        stats = G.contact_statistics(labels)
        meta = {"n_complexes": len(labels), "n_failed": len(fails),
                "n_shards": len(shards),
                "with_affinity_label": int(sum(L["has_affinity_label"] for L in labels)),
                "contact_statistics": stats,
                "note": "contact_prior_loss target should match "
                        "mean_contacts_per_buried_atom"}
        (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
        pd.DataFrame(fails, columns=["pdb_id", "reason"]).to_csv(
            paths.REPORTS / "geometry_failures.csv", index=False)
        log(f"merged {len(labels):,} complexes ({len(fails):,} failed)")
        print(json.dumps(meta, indent=1))
        if stats:
            print(f"\n  Set contact_prior_loss(target=...) to "
                  f"{stats['mean_contacts_per_buried_atom']:.1f} - it is currently a "
                  "constant in physics/energy.py and should come from this number.")
        return

    rows = discover(Path(a.pdbbind) if a.pdbbind else None,
                    Path(a.pdb_dir) if a.pdb_dir else None, a.allow_ic50)
    log(f"{len(rows):,} candidate complexes")
    if rows.empty:
        raise SystemExit("no complexes found")

    if a.selfcheck:
        selfcheck(rows)
        return

    sid, nsh = (a.shard_id, a.n_shards) if a.shard_id is not None else slurm_shard()
    nsh = nsh or 1
    mine = rows.iloc[np.array_split(np.arange(len(rows)), nsh)[sid]]
    sp = OUT / f"shard_{sid:04d}.pkl"
    if sp.exists() and sp.stat().st_size > 0:
        ## Idempotent, so re-running the wave after a partial failure elsewhere does
        ## not recompute geometry that already finished.
        log(f"shard {sid} already present - skipping")
        return
    log(f"shard {sid}/{nsh}: {len(mine):,} complexes")
    labels, failures = process(mine)
    tmp = sp.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"labels": labels, "failures": failures}, fh, protocol=4)
    tmp.replace(sp)
    log(f"shard {sid} done: {len(labels):,} ok, {len(failures):,} failed")


if __name__ == "__main__":
    main()
