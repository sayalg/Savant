#!/usr/bin/env python3
"""Download TabFM weights to a local directory.

Run this ONCE on a node that has internet access (an HPC login node or a
dedicated data-transfer node). Compute nodes are usually offline, so the goal
is to get every file onto shared storage here, then run everything else with
no network at all (see run_tabfm.py + HF_HUB_OFFLINE).

The full repo is ~13 GB (classification + regression). You almost always only
need one task, so by default we fetch just the subfolder you ask for.

Examples
--------
    # only the classification weights (~6.6 GB) into /scratch/$USER/tabfm
    python download_weights.py --task classification --dest /scratch/$USER/tabfm

    # both tasks
    python download_weights.py --task both --dest /scratch/$USER/tabfm

Gated repo
----------
The weights carry a non-commercial license and the repo may be gated. If the
download 401/403s, accept the license on the model page while logged in, then
authenticate on the node with:  hf auth login   (or set HF_TOKEN=...).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "google/tabfm-1.0.0-pytorch"


def build_patterns(task: str) -> list[str]:
    """Only download what we need. Root config.json is always included."""
    patterns = ["config.json", "*.md", "LICENSE"]
    if task in ("classification", "both"):
        patterns.append("classification/*")
    if task in ("regression", "both"):
        patterns.append("regression/*")
    return patterns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["classification", "regression", "both"],
                    default="classification",
                    help="Which weights to fetch (default: classification).")
    ap.add_argument("--dest", required=True,
                    help="Local directory to write weights into. Put this on "
                         "scratch/project storage, NOT your home dir.")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default=None,
                    help="Pin a specific commit hash for reproducibility.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel download workers.")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token for a gated repo (or set HF_TOKEN / run "
                         "`hf auth login`).")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. `pip install huggingface_hub`.")

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Keep the big HF cache off the home quota too, in case snapshot_download
    # still stages anything there.
    os.environ.setdefault("HF_HOME", str(dest / ".hf_home"))

    patterns = build_patterns(args.task)
    print(f"Downloading {args.repo_id} (task={args.task}) -> {dest}")
    print(f"  allow_patterns = {patterns}")
    if args.revision:
        print(f"  revision       = {args.revision}")

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=args.workers,
        token=args.token,
        # local_dir_use_symlinks left default so files are real copies on
        # local_dir for older hub versions; newer hubs ignore it.
    )
    print(f"\nDone. Weights are at: {path}")
    print("Now run offline, e.g.:")
    print(f"  export HF_HUB_OFFLINE=1")
    print(f"  python run_tabfm.py --task {args.task if args.task != 'both' else 'classification'} \\")
    print(f"      --weights {dest} --train train.csv --test test.csv \\")
    print(f"      --target label --out results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
