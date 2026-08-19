#!/usr/bin/env python
"""
Step 01 - Extract v2 features for every complex in a manifest.

Roughly 0.3 s per complex on one core (measured on real AF3+HADDOCK structures:
0.35 s for a 1266-atom protein, 0.23 s for 3357 atoms - cost is set by the
pocket, not the protein). The full 47,619 training set is about 4 CPU-hours, so
a 20-way array finishes in ~12 minutes.

Compare with v1, where the same table required a HADDOCK3 minimisation per
complex.

Failures are recorded per row rather than aborting the shard: a single
unparseable ligand should not cost the other 2,000 complexes in the task.

    python 01_extract_features.py --manifest work/manifest_training.csv --num-shards 20
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

from savant2.features import featurize                        # noqa: E402
from savant2 import LIGAND_CHAIN, PROTEIN_CHAIN               # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "work"


def shard_of(num_shards: int, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(task) if task is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--no-ecfp", action="store_true",
                    help="skip the 512-bit fingerprint (structural features only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Fail now, not in twelve minutes. The parquet engine is only exercised by
    # the final write, so a missing pyarrow would otherwise throw away a whole
    # shard's worth of completed extraction.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pyarrow is required to write the feature shards.\n"
            "  conda install -c conda-forge pyarrow    (or: pip install pyarrow)"
        )

    manifest = pd.read_csv(args.manifest)
    shard = shard_of(args.num_shards, args.shard_id)
    tag = Path(args.manifest).stem.replace("manifest_", "")
    out_dir = Path(args.out_dir) if args.out_dir else WORK / f"features_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard:02d}.parquet"

    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to redo")
        return

    rows = manifest.iloc[shard::args.num_shards] if args.num_shards > 1 else manifest
    if args.limit:
        rows = rows.head(args.limit)
    print(f"shard {shard}/{args.num_shards}: {len(rows)} complexes", flush=True)

    records, failures = [], []
    started = time.perf_counter()
    for n, r in enumerate(rows.itertuples(index=False), 1):
        try:
            features, meta = featurize(
                r.pdb_path, r.smiles,
                protein_chain=PROTEIN_CHAIN, ligand_chain=LIGAND_CHAIN,
                include_ecfp=not args.no_ecfp,
            )
        except Exception as exc:
            failures.append({"id": r.id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        record = {"id": r.id, "ligand_typed": meta["ligand_typed"],
                  "n_ligand_atoms": meta["n_ligand_atoms"],
                  "n_protein_atoms": meta["n_protein_atoms"]}
        record.update(features)
        records.append(record)
        if n % 500 == 0:
            rate = n / (time.perf_counter() - started)
            print(f"  {n}/{len(rows)}  {rate:.1f}/s  eta {(len(rows)-n)/rate/60:.1f} min",
                  flush=True)

    if records:
        frame = pd.DataFrame.from_records(records)
        # float32 halves the file and costs nothing: every feature is a count,
        # a distance or an area.
        for c in frame.columns:
            if frame[c].dtype == np.float64:
                frame[c] = frame[c].astype(np.float32)
        frame.to_parquet(out_path, index=False)
        print(f"wrote {out_path}: {frame.shape[0]} rows x {frame.shape[1]} cols")

    if failures:
        fail_path = out_dir / f"failures_shard_{shard:02d}.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"{len(failures)} failures -> {fail_path}")

    elapsed = time.perf_counter() - started
    print(f"shard {shard} done in {elapsed/60:.1f} min "
          f"({len(records)/max(elapsed,1e-9):.1f} complexes/s)")


if __name__ == "__main__":
    main()
