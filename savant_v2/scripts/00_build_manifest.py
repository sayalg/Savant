#!/usr/bin/env python
"""
Step 00 - Build the (id, pdb_path, smiles) manifest the extractor consumes.

Two sources, same output shape, so one extraction job serves both:

  --source training   the 47,619 AF3+HADDOCK complexes behind
                      combined_50k_filtered.csv, carrying the pKd label and the
                      binder/nonbinder level recovered from v1's prepped table
  --source benchmark  the 1000-pair benchmark, scored against the Boltz
                      structures in ML_model/benchmark/eval/work/structures/

Structure paths are resolved by pattern rather than assumed, because the
training complexes live wherever the AF3 run put them. Point --structure-dir at
the parent and --pattern at the layout; the default matches
haddock_testing/<job>_model.pdb.

    python 00_build_manifest.py --source training  --structure-dir /scratch/.../docked
    python 00_build_manifest.py --source benchmark
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # savant_v2/
REPO = ROOT.parent
WORK = ROOT / "work"

TRAIN_CSV = REPO / "ML_model" / "combined_50k_filtered.csv"
LABELS_CSV = ROOT / "work_labels.csv"   # Job_ID -> binding_level, pKd
BENCH_CSV = REPO / "ML_model" / "benchmark" / "output" / "savant_benchmark_1000.csv"
BENCH_STRUCTS = REPO / "ML_model" / "benchmark" / "eval" / "work" / "structures"


def check_structures(df, structure_dir: str, pattern: str, skip: bool = False):
    """Which complexes actually have a structure file on disk.

    One os.path.exists() per row is the obvious implementation and the wrong
    one: on a parallel filesystem each call is a metadata round trip, and 47,619
    of them measured at ~13 ms each - over ten minutes, most of it spent waiting.
    Negative lookups are worse still, because a miss cannot be served from cache.

    So ask once instead. A single listing of the parent directory answers the
    same question for every row, turning 47,619 round trips into one.
    """
    import time

    if skip:
        print("  skipping the structure existence check (--skip-exists-check)", flush=True)
        return pd.Series(True, index=df.index)

    base = Path(structure_dir)
    # A pattern with a directory component (the AF3 '{id}/{id}_model.pdb'
    # layout) is answered by whether the per-job directory is present; a flat
    # pattern by whether the file itself is listed.
    nested = "/" in pattern or os.sep in pattern
    started = time.perf_counter()
    print(f"  listing {base} to check {len(df)} structures ...", flush=True)
    try:
        entries = set(os.listdir(base))
    except OSError as exc:
        print(f"  cannot list {base} ({exc}); falling back to per-path stat, "
              "which is slow on a parallel filesystem", flush=True)
        return pd.Series([os.path.exists(p) for p in df.pdb_path], index=df.index)

    if nested:
        exists = df.id.astype(str).isin(entries)
    else:
        wanted = df.id.astype(str).map(lambda i: pattern.format(id=i))
        exists = wanted.isin(entries)

    elapsed = time.perf_counter() - started
    print(f"  found {len(entries)} entries, {int(exists.sum())}/{len(df)} matched "
          f"in {elapsed:.1f}s", flush=True)

    # The listing proves the directory is there; spot-check that the file inside
    # it is too, so a run of empty job directories is caught here rather than
    # 2,000 complexes into an extraction shard.
    if nested and exists.any():
        sample = df.loc[exists, "pdb_path"].head(3).tolist()
        bad = [p for p in sample if not os.path.exists(p)]
        if bad:
            print(f"  WARNING: job directories exist but the model file does not, e.g.\n"
                  f"    {bad[0]}\n"
                  "  check --pattern against the actual layout", flush=True)
    return exists.values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["training", "benchmark"], required=True)
    ap.add_argument("--skip-exists-check", action="store_true",
                    help="trust that every structure is present; skips all filesystem "
                         "metadata calls entirely")
    # Layout from Scripts/pipeline.py: the AF3 stage writes
    #   <output_dir>/alphafold3/predictions/<job>/<job>_model.pdb
    # with the protein on chain A and the ligand on chain B as residue LIG
    # (patch_cif_residue_name + convert_cif_to_pdb).
    #
    # Deliberately the AF3 complex, NOT haddock3/<job>/output/1_emref/emref_1.pdb.gz.
    # Two reasons: it keeps HADDOCK out of the training path entirely, which is
    # the claim v2 is testing; and it matches what inference sees, since an
    # AlphaFold Server or Boltz structure has never been energy-minimised.
    # Training on minimised poses and predicting on raw ones would be a
    # distribution shift built in by construction.
    ap.add_argument("--structure-dir", default=str(REPO / "haddock_testing"),
                    help="parent of the docked complexes, e.g. "
                         "<pipeline_output>/alphafold3/predictions")
    ap.add_argument("--pattern", default="{id}_model.pdb",
                    help="path pattern under --structure-dir with {id} substituted; "
                         "use '{id}/{id}_model.pdb' for the AF3 predictions layout")
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-exists", action="store_true",
                    help="drop rows whose structure file is missing (default: keep and flag)")
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)

    if args.source == "training":
        df = pd.read_csv(TRAIN_CSV)[["Job_ID", "smiles", "neg_log10_affinity_M"]]
        df = df.rename(columns={"Job_ID": "id", "neg_log10_affinity_M": "pkd"})
        if LABELS_CSV.exists():
            labels = pd.read_csv(LABELS_CSV).rename(columns={"Job_ID": "id"})
            df = df.merge(labels[["id", "binding_level"]], on="id", how="left")
        else:
            # Fall back to v1's recovered threshold if the export is absent.
            df["binding_level"] = ["nonbinder" if v <= 4.0 else "binder" for v in df.pkd]
        base = Path(args.structure_dir)
        df["pdb_path"] = [str(base / args.pattern.format(id=i)) for i in df.id]
        # Protein identity, required for grouped splitting. The training table
        # itself has no protein column at all - that omission is why v1 could
        # not tell two targets apart - so it is recovered from the docking input.
        source = REPO / "Input" / "input.csv"
        if source.exists():
            # Read in chunks and keep only the ~47k ids actually needed.
            # input.csv is 1.33 GB of mostly protein sequences; loading it whole
            # costs 3-4 GB of RAM (pandas holds them as Python str objects) to
            # then discard 97% of the rows. Chunking bounds it to the chunk size.
            wanted = set(df.id)
            print(f"  scanning {source.name} for {len(wanted)} sequences "
                  "(1.3 GB, chunked - a few minutes)", flush=True)
            kept = []
            for n, chunk in enumerate(pd.read_csv(source, usecols=["ID", "seq"],
                                                  chunksize=250_000), 1):
                hit = chunk[chunk.ID.isin(wanted)]
                if len(hit):
                    kept.append(hit)
                found = sum(len(k) for k in kept)
                print(f"    chunk {n}: {found}/{len(wanted)} matched", flush=True)
                if found >= len(wanted):
                    break        # every id resolved; no need to read the rest
            seqs = (pd.concat(kept, ignore_index=True).rename(columns={"ID": "id"})
                    if kept else pd.DataFrame(columns=["id", "seq"]))
            df = df.merge(seqs, on="id", how="left")
            df["protein_group"] = df.seq.fillna("").map(lambda s: hash(s) & 0xFFFFFFFF)
            # seq is kept: the ESM stage needs it, and re-joining 1.8M rows
            # per shard would cost more than carrying it here.
            print(f"  resolved {df.protein_group.nunique()} distinct proteins", flush=True)
        else:
            print(f"  WARNING: {source} absent; protein_group falls back to row id "
                  "and the split will NOT be protein-clustered")
            df["protein_group"] = range(len(df))
    else:
        df = pd.read_csv(BENCH_CSV)[["benchmark_id", "ligand_smiles", "neg_log10_affinity_M"]]
        df = df.rename(columns={"benchmark_id": "id", "ligand_smiles": "smiles",
                                "neg_log10_affinity_M": "pkd"})
        df["binding_level"] = "binder"      # every benchmark pair has a measured Kd
        df["pdb_path"] = [str(BENCH_STRUCTS / f"{i}.pdb") for i in df.id]
        meta = pd.read_csv(BENCH_CSV)[["benchmark_id", "uniprot_id", "novelty_tier"]]
        df = df.merge(meta.rename(columns={"benchmark_id": "id"}), on="id", how="left")
        df["protein_group"] = df.uniprot_id
        df = df.merge(pd.read_csv(BENCH_CSV)[["benchmark_id", "target_sequence"]]
                        .rename(columns={"benchmark_id": "id", "target_sequence": "seq"}),
                      on="id", how="left")

    df["structure_exists"] = check_structures(
        df, args.structure_dir, args.pattern, skip=args.skip_exists_check
    )
    missing = int((~df.structure_exists).sum())
    if args.require_exists:
        df = df[df.structure_exists]

    out = Path(args.out) if args.out else WORK / f"manifest_{args.source}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {missing} missing structures")
    if missing:
        print(f"  {df.pdb_path.iloc[0]}")
        print("  ^ check --structure-dir / --pattern if that path looks wrong")
    if "binding_level" in df:
        print("  binding_level:", df.binding_level.value_counts().to_dict())


if __name__ == "__main__":
    main()
