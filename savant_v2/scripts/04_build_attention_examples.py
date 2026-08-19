#!/usr/bin/env python
"""
Step 04 - Build the per-complex tensors the cross-attention model trains on.

For every complex this stores four things:

  ligand_features   (M, 16)  RDKit atom typing
  pocket_embedding  (K, D)   ESM-2 per-residue states for the pocket residues
  pocket_type       (K, 20)  residue-type one-hot
  distance          (M, K)   ligand atom to pocket residue, in angstroms

ESM is the expensive part and is amortised by protein: 47,619 complexes cover
only 11,965 sequences, so rows are processed grouped by sequence and each
protein is embedded once. With a GPU that is minutes; the per-sequence cache
also survives across shards and re-runs.

Storage is about 61 KB per complex at the default 640-dimensional model
(48 pocket residues, float16), so roughly 3 GB for the full training set.

    python 04_build_attention_examples.py --manifest work/manifest_training.csv --num-shards 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from savant2 import LIGAND_CHAIN, PROTEIN_CHAIN                      # noqa: E402
from savant2.attention import (MAX_LIGAND_ATOMS, ligand_atom_features,  # noqa: E402
                               pairwise_distances, residue_type_onehot)
from savant2.embeddings import (DEFAULT_ESM_MODEL, MAX_POCKET_RESIDUES,  # noqa: E402
                                POCKET_CUTOFF, EsmEmbedder, pocket_residues)
from savant2.ligand import ligand_mol                                # noqa: E402
from savant2.structure import read_complex, residue_keys             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"


def pocket_coordinates(cx, positions) -> np.ndarray:
    """Representative coordinate per pocket residue: its atom nearest the ligand.

    Not the CA. The atom actually making the contact is what the distance bias
    should be measured from - a long Arg side chain reaching into the site has
    its CA 8 A further away than the guanidinium doing the work.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(cx.l_xyz)
    dist, _ = tree.query(cx.p_xyz, k=1)
    coords = np.zeros((len(positions), 3), dtype=np.float32)
    for i, pos in enumerate(positions):
        sel = np.where(cx.p_res_id == pos + 1)[0]
        if len(sel):
            coords[i] = cx.p_xyz[sel[np.argmin(dist[sel])]]
    return coords


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    ap.add_argument("--esm-cache", default=str(WORK / "esm_cache"))
    ap.add_argument("--pocket-cutoff", type=float, default=POCKET_CUTOFF)
    ap.add_argument("--max-pocket", type=int, default=MAX_POCKET_RESIDUES)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import joblib

    shard = args.shard_id
    if shard is None:
        shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    tag = Path(args.manifest).stem.replace("manifest_", "")
    out_dir = Path(args.out_dir) if args.out_dir else WORK / f"attention_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard:02d}.joblib"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to redo")
        return

    manifest = pd.read_csv(args.manifest)
    if "seq" not in manifest.columns:
        raise SystemExit("manifest has no 'seq' column - rebuild it with 00_build_manifest.py")
    rows = manifest.iloc[shard::args.num_shards] if args.num_shards > 1 else manifest
    if args.limit:
        rows = rows.head(args.limit)
    # Group by protein so each sequence is embedded once per shard.
    rows = rows.sort_values("seq", kind="stable")
    print(f"shard {shard}/{args.num_shards}: {len(rows)} complexes, "
          f"{rows.seq.nunique()} distinct proteins", flush=True)

    embedder = EsmEmbedder(args.esm_model, cache_dir=args.esm_cache)
    print(f"ESM {args.esm_model} on {embedder.device}, dim {embedder.dim}", flush=True)

    examples, failures = [], []
    started = time.perf_counter()
    for n, r in enumerate(rows.itertuples(index=False), 1):
        try:
            cx = read_complex(r.pdb_path, PROTEIN_CHAIN, LIGAND_CHAIN)
            positions, names, dists = pocket_residues(
                cx, cutoff=args.pocket_cutoff, max_residues=args.max_pocket
            )
            if len(positions) == 0:
                raise ValueError("no pocket residues within cutoff")

            per_residue, _ = embedder.pocket_embedding(str(r.seq), positions)
            if len(per_residue) != len(positions):
                # Residue numbering ran past the sequence: skip rather than
                # silently pair the wrong embedding with the wrong residue.
                raise ValueError(
                    f"pocket positions out of range for a {len(r.seq)}-residue sequence"
                )

            mol = ligand_mol(r.pdb_path, LIGAND_CHAIN, r.smiles)
            atom_features = ligand_atom_features(mol, cx.n_ligand_atoms)[:MAX_LIGAND_ATOMS]
            ligand_xyz = cx.l_xyz[:MAX_LIGAND_ATOMS]

            examples.append({
                "id": r.id,
                "ligand_features": atom_features.astype(np.float16),
                "pocket_embedding": per_residue,
                "pocket_type": residue_type_onehot(names).astype(np.float16),
                "pocket_distance": dists.astype(np.float16),
                "distance": pairwise_distances(
                    ligand_xyz, pocket_coordinates(cx, positions)
                ).astype(np.float16),
                "pkd": float(r.pkd),
                "is_binder": str(r.binding_level) == "binder",
                "protein_group": r.protein_group,
            })
        except Exception as exc:
            failures.append({"id": r.id, "error": f"{type(exc).__name__}: {exc}"})
        if n % 250 == 0:
            rate = n / (time.perf_counter() - started)
            print(f"  {n}/{len(rows)}  {rate:.1f}/s  eta {(len(rows)-n)/rate/60:.1f} min",
                  flush=True)

    joblib.dump({"examples": examples, "esm_dim": embedder.dim,
                 "esm_model": args.esm_model}, out_path, compress=3)
    print(f"wrote {out_path}: {len(examples)} examples")
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / f"failures_shard_{shard:02d}.csv", index=False)
        print(f"{len(failures)} failures -> failures_shard_{shard:02d}.csv")


if __name__ == "__main__":
    main()
