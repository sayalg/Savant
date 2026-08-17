#!/usr/bin/env python
"""
Pre-flight for the Boltz job: delete prediction directories that cannot resume.

`boltz predict` decides what to skip with two different tests:

    filter_inputs_structure   skips a record if predictions/<id>/ EXISTS AT ALL
    filter_inputs_affinity    skips a record only if affinity_<id>.json exists

A run killed part-way (walltime, OOM, a crash) leaves directories that satisfy
the first test but not the second. On the next submission the structure stage
skips those records - so it never writes the `pre_affinity_<id>.npz` that the
affinity stage needs - and the affinity stage then dies on the missing file:

    FileNotFoundError: .../predictions/bm_0101/pre_affinity_bm_0101.npz

That kills the whole array task, not just the one record. This removes the
directories in that state so both stages re-run for them, and leaves finished
work alone.

A directory is kept when it is resumable:

  * <id>_model_0.pdb present, and
  * affinity not requested, or affinity_<id>.json present (done), or
    pre_affinity_<id>.npz present (structure stage did its half; the affinity
    stage can still pick it up)

    python scripts/prune_partial_boltz.py                  # whole tree
    python scripts/prune_partial_boltz.py --shard shard_00 # one shard
    python scripts/prune_partial_boltz.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

from common import BOLTZ_DIR, MANIFEST, log


def prediction_dirs(boltz_dir: str, shard: str | None) -> list[str]:
    root = os.path.join(boltz_dir, shard) if shard else boltz_dir
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "predictions":
            found += [os.path.join(dirpath, d) for d in sorted(dirnames)]
    return found


def is_resumable(path: str, bid: str, affinity_requested: bool) -> tuple[bool, str]:
    files = set(os.listdir(path)) if os.path.isdir(path) else set()
    if not any(f.startswith(f"{bid}_model_") and f.endswith((".pdb", ".cif")) for f in files):
        return False, "no structure written"
    if not affinity_requested:
        return True, "structure only, complete"
    if f"affinity_{bid}.json" in files:
        return True, "complete"
    if f"pre_affinity_{bid}.npz" in files:
        return True, "affinity stage can resume"
    return False, "structure present but affinity cannot resume"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--boltz-dir", default=BOLTZ_DIR)
    ap.add_argument("--shard", default=None, help="e.g. shard_00 (default: every shard)")
    ap.add_argument("--dry-run", action="store_true", help="report without deleting")
    args = ap.parse_args()

    affinity = {}
    if os.path.exists(args.manifest):
        manifest = pd.read_csv(args.manifest)
        affinity = dict(zip(manifest.benchmark_id, manifest.affinity_requested))

    dirs = prediction_dirs(args.boltz_dir, args.shard)
    if not dirs:
        log("no prediction directories yet - nothing to prune")
        return

    pruned, kept = [], 0
    for path in dirs:
        bid = os.path.basename(path)
        ok, reason = is_resumable(path, bid, bool(affinity.get(bid, False)))
        if ok:
            kept += 1
            continue
        pruned.append((bid, reason))
        if not args.dry_run:
            shutil.rmtree(path, ignore_errors=True)

    verb = "would remove" if args.dry_run else "removed"
    log(f"{len(dirs)} prediction dirs: {kept} resumable, {verb} {len(pruned)}")
    for bid, reason in pruned[:20]:
        log(f"  {bid}: {reason}")
    if len(pruned) > 20:
        log(f"  ... and {len(pruned) - 20} more")


if __name__ == "__main__":
    main()
