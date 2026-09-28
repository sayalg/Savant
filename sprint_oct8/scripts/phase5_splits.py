#!/usr/bin/env python3
"""Phase 5: build and FREEZE the splits (PLAN.md Section 8, Phase 5).

Builds every split in Section 5.3, checks the guarantee each one advertises, writes
them to `splits/` with counts and overlap statistics, and fails loudly when a split
does not deliver what its name promises.

Gate: automated leakage tests pass where the split promises them.

Why this is a gate and not a step: the previous project's protein-GROUPED split put
21 near-identical HIV-1 RT constructs in the fit fold while another RT sat in test
at 0.94 k-mer containment. Internal r came out at 0.580 and the same model scored
0.327 externally. A split that quietly fails its own guarantee produces numbers
that look fine and mean nothing, so `describe(strict=True)` raises rather than warns.

mmseqs runs in a container on the cluster (slurm/20_mmseqs.sbatch); this script only
consumes the TSV. Without it, splits fall back to protein-GROUPED, which is the
failure above - so it says so, every time.

    python scripts/phase5_splits.py
    python scripts/phase5_splits.py --viral-mask work/viral_mask.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pfpl import conf, paths  # noqa: E402
from pfpl.data import splits as SP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    conf.add_cfg_args(ap)
    ap.add_argument("--only", nargs="*", default=None, help="subset of split names")
    ap.add_argument("--viral-mask", default=None,
                    help=".npy bool per PROTEIN index, for leave_viral_out")
    ap.add_argument("--no-strict", action="store_true",
                    help="warn instead of failing on an overlap (debugging only)")
    a = ap.parse_args()
    cfg = conf.from_args(a)
    paths.ensure()

    pairs = pd.read_parquet(paths.WORK / "pairs.parquet")
    prot = pd.read_parquet(paths.WORK / "proteins.parquet")
    n_prot = len(prot)
    print(f"  {len(pairs):,} pairs, {n_prot:,} proteins")

    ## ---- protein clusters ---------------------------------------------------
    tsv = Path(cfg.data.mmseqs_tsv)
    tsv = tsv if tsv.is_absolute() else paths.WORK / tsv
    if tsv.exists():
        cl = SP.clusters_from_mmseqs(tsv, n_prot)
        print(f"  mmseqs: {len(np.unique(cl)):,} clusters at "
              f"{cfg.data.mmseqs_identity:.0%} identity from {tsv}")
        route = f"mmseqs @ {cfg.data.mmseqs_identity}"
    else:
        cl = np.arange(n_prot, dtype=np.int32)
        route = "protein-grouped FALLBACK"
        print(f"\n  *** {tsv} MISSING ***")
        print("  Falling back to one cluster per protein, i.e. a protein-GROUPED")
        print("  split. That is the split that let 21 near-identical HIV-1 RT")
        print("  constructs sit in the fit fold while another RT was 'held out',")
        print("  and it is why an internal r of 0.580 became 0.327 externally.")
        print("  Run slurm/20_mmseqs.sbatch before quoting any number from this.\n")
    np.save(paths.WORK / "protein_clusters.npy", cl)

    pair_cluster = cl[pairs.protein_idx.to_numpy()]
    print("  computing Murcko scaffolds")
    scaf = SP.murcko_scaffolds(pairs.ligand_smiles_std.tolist())
    print(f"  {len(np.unique(scaf)):,} distinct scaffolds")
    np.save(paths.WORK / "scaffolds.npy", scaf)

    viral = None
    if a.viral_mask and Path(a.viral_mask).exists():
        vm = np.load(a.viral_mask)
        if len(vm) != n_prot:
            raise SystemExit(f"viral mask has {len(vm)} entries, "
                             f"the protein table has {n_prot}")
        viral = vm[pairs.protein_idx.to_numpy()]
        print(f"  viral panel: {int(vm.sum()):,} proteins, {int(viral.sum()):,} pairs")

    names = a.only or ["random", "scaffold", "protein_cluster", "double"]
    if viral is not None and "leave_viral_out" not in names:
        names.append("leave_viral_out")

    y = pairs.pK.to_numpy(float)
    index, failed = {}, []
    for name in names:
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        try:
            tr, va, te = SP.build(name, pairs, pair_cluster, scaf, viral,
                                  float(cfg.data.val_frac), float(cfg.data.test_frac),
                                  int(cfg.seed))
        except SystemExit as e:
            print(f"  skipped: {e}")
            continue
        try:
            meta = SP.describe(name, tr, va, te, pair_cluster, scaf, y,
                               strict=not a.no_strict)
        except SystemExit as e:
            failed.append(name)
            print(f"  GUARANTEE FAILED: {e}")
            continue
        meta |= {"cluster_route": route, "seed": int(cfg.seed),
                 "config_hash": cfg.hash}
        SP.save(paths.SPLITS, name, tr, va, te, meta)
        index[name] = meta

    (paths.SPLITS / "README.md").write_text(_readme(index, route, len(pairs)))
    (paths.SPLITS / "index.json").write_text(json.dumps(index, indent=1))

    print("\n" + "=" * 78)
    print("GATE - Phase 5")
    for n, m in index.items():
        print(f"  {n:<18} train {m['n_train']:>8,}  val {m['n_val']:>7,}  "
              f"test {m['n_test']:>7,}  discarded {m['n_discarded']:>7,}  "
              f"{'OK' if m['guarantee_ok'] else 'VIOLATED'}")
    if failed:
        print(f"  FAILED: {failed}")
    if route.startswith("protein-grouped"):
        print("  clustering: FALLBACK - the headline number cannot come from this")
    print("=" * 78)
    sys.exit(1 if failed else 0)


def _readme(index: dict, route: str, n_pairs: int) -> str:
    L = ["# Frozen splits", "",
         f"Built from `work/pairs.parquet` ({n_pairs:,} pairs). Protein clustering: "
         f"**{route}**.", "",
         "These files are frozen and version-controlled. Rebuilding them with a "
         "different seed invalidates every number measured against the old ones.", "",
         "| split | train | val | test | discarded | protein overlap | scaffold overlap |",
         "|---|---|---|---|---|---|---|"]
    for n, m in index.items():
        L.append(f"| `{n}` | {m['n_train']:,} | {m['n_val']:,} | {m['n_test']:,} | "
                 f"{m['n_discarded']:,} | {m.get('protein_overlap_train_test', '-')} | "
                 f"{m.get('scaffold_overlap_train_test', '-')} |")
    L += ["", "## What each one is for", "",
          "- `random` - **sanity only**. Section 0.4.4 forbids headlining it.",
          "- `scaffold` - ligand generalisation with proteins shared across folds.",
          "- `protein_cluster` - **the headline split.** Whole mmseqs clusters move "
          "together, so a 95%-identical construct cannot be 'held out' while its "
          "twin trains. Section 9's comparison rule is defined on this split.",
          "- `double` - unseen protein cluster AND unseen Murcko scaffold. Pairs that "
          "share exactly one axis are DISCARDED, which is the point: a pair whose "
          "protein is held out but whose scaffold was trained on measures "
          "memorisation, and mixing the two is how 0.580 internal became 0.327 "
          "external.",
          "- `leave_viral_out` - the viral panel as a held-out application domain.", "",
          "## Guarantees", "",
          "`pfpl.data.splits.describe(strict=True)` raises if a split has any group "
          "in both folds. `tests/test_leakage.py` re-checks it in CI.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
