#!/usr/bin/env python
"""
Step 09 - PLAPT predictions for every benchmark pair.

    python 09_run_plapt.py --plapt-dir /opt/PLAPT
    python 09_run_plapt.py --plapt-dir /opt/PLAPT --limit 4     # smoke test

PLAPT (Rose et al., bioRxiv 2024.02.08.575577) predicts affinity from a protein
sequence and a ligand SMILES using ProtBERT + ChemBERTa embeddings and a small
ONNX head. It is the closest published peer to Savant: same inputs, same
structure-free premise, different encoders and a much smaller trained head.

WHY THIS STEP DOES NOT USE THE BOLTZ STRUCTURES
-----------------------------------------------
Unlike steps 04 and 05, PLAPT needs no structure - only the sequence and SMILES
already in the benchmark CSV. So it reads the benchmark directly rather than the
manifest, and its coverage is not capped by whether Boltz-2 produced a pose. It
should reach all 500 pairs where Boltz-2 reached 496.

WHY THIS DOES NOT SHELL OUT TO plapt_cli.py
-------------------------------------------
The upstream CLI is broken for paired input. It calls

    plapt.predict_affinity(args.target[0], args.smiles)

passing a *string* as `prot_seqs`, while `predict_affinity` compares
`len(prot_seqs) != len(mol_smiles)` - so len() counts characters of the sequence
and the call raises unless the sequence happens to be exactly as long as the
SMILES list. This step uses the Python API with two parallel lists, which is the
contract `predict_affinity` actually documents.

UNITS
-----
PLAPT returns `neg_log10_affinity_M` directly - the same quantity and the same
name as the benchmark's label column. It is written out here as `plapt_pk` to
avoid colliding with the ground truth when merged.

Internally it denormalises with mean 6.51287 and scale 1.56141, i.e. its training
labels had that mean and spread. The benchmark's own labels sit at mean 7.30,
sd 1.27, so PLAPT is expected to under-predict on this set for the same reason
every model trained on a broader corpus does.

CONTAMINATION
-------------
PLAPT was posted 2024-02-09 and trained before that. Every benchmark measurement
is from 2025 or later and absent from ChEMBL v34 (2024-04), so the temporal gate
rules out PLAPT having seen these pairs - the same protection Boltz-2 gets.

Note the ligand-similarity gate does NOT transfer: it removed ligands within
Tanimoto 0.50 of a benchmark ligand from *Savant's* training set, not PLAPT's.
If anything that biases the comparison against Savant, which is the conservative
direction.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from common import (BENCHMARK_CSV, RESULTS, ensure_dirs, log, read_json,
                    slurm_shard, take_shard, write_json)

PLAPT_RESULTS = os.path.join(RESULTS, "plapt")


def load_plapt(plapt_dir: str, onnx: str | None, device: str, cache_dir: str):
    """Import PLAPT from its source tree and construct the predictor.

    The upstream default `prediction_module_path="models/affinity_predictor.onnx"`
    is relative to the working directory, so an absolute path is passed instead
    rather than relying on a chdir.
    """
    ## expanduser before abspath: abspath alone turns a literal "~/sw/PLAPT"
    ## (a shell passes the tilde through unexpanded when the value is quoted or
    ## comes from ${VAR:-~/...}) into "<cwd>/~/sw/PLAPT", which never exists.
    raw = plapt_dir
    plapt_dir = os.path.abspath(os.path.expanduser(plapt_dir))
    if not os.path.isfile(os.path.join(plapt_dir, "plapt.py")):
        detail = ""
        if os.path.isdir(plapt_dir):
            detail = f"\nThat directory exists and contains: {sorted(os.listdir(plapt_dir))[:20]}"
        raise SystemExit(
            f"no plapt.py in {plapt_dir}\n"
            f"  (--plapt-dir was given as {raw!r})" + detail + "\n"
            "Point --plapt-dir at the PLAPT checkout - the directory holding "
            "plapt.py and models/affinity_predictor.onnx:\n"
            "  git clone https://github.com/Bindwell/PLAPT.git ~/sw/PLAPT"
        )
    sys.path.insert(0, plapt_dir)

    onnx = os.path.abspath(os.path.expanduser(onnx)) if onnx else \
        os.path.join(plapt_dir, "models", "affinity_predictor.onnx")
    if not os.path.isfile(onnx):
        raise SystemExit(f"missing ONNX head: {onnx}\nPass --onnx to override.")

    try:
        from plapt import Plapt
    except ImportError as exc:
        raise SystemExit(
            f"cannot import plapt ({exc}). Its dependencies (torch, transformers, "
            "onnxruntime, diskcache) must be importable - run this inside the "
            "PLAPT container."
        )
    log(f"PLAPT from {plapt_dir}, head {onnx}, device {device}")
    return Plapt(prediction_module_path=onnx, device=device, cache_dir=cache_dir,
                 use_tqdm=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--benchmark", default=BENCHMARK_CSV)
    ap.add_argument("--out-dir", default=PLAPT_RESULTS)
    ap.add_argument("--plapt-dir", default=os.environ.get("PLAPT_DIR", "/app"))
    ap.add_argument("--onnx", default=None, help="absolute path to affinity_predictor.onnx")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu; PLAPT falls back to cpu when cuda is absent")
    ap.add_argument("--cache-dir", default=None,
                    help="PLAPT embedding cache (default: <out-dir>/../plapt_cache)")
    ap.add_argument("--prot-batch", type=int, default=2)
    ap.add_argument("--mol-batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=100,
                    help="pairs scored per call; results are written after each chunk")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    shard_id = args.shard_id if args.shard_id is not None else slurm_shard(args.num_shards)
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out_dir)), "plapt_cache")
    ensure_dirs(args.out_dir, RESULTS, cache_dir)

    df = pd.read_csv(args.benchmark)
    need = {"benchmark_id", "target_sequence", "ligand_smiles"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{args.benchmark} is missing {sorted(missing)}")

    rows = take_shard(list(df.itertuples(index=False)), shard_id, args.num_shards)
    if args.limit:
        rows = rows[: args.limit]

    todo = [r for r in rows
            if args.overwrite
            or not os.path.exists(os.path.join(args.out_dir, f"{r.benchmark_id}.json"))]
    log(f"shard {shard_id}/{args.num_shards}: {len(rows)} pairs, {len(todo)} to score")
    if not todo:
        log("nothing to do")
        return

    plapt = load_plapt(args.plapt_dir, args.onnx, args.device, cache_dir)

    n_done, failures, t0 = 0, [], time.time()
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        seqs = [str(r.target_sequence) for r in chunk]
        smis = [str(r.ligand_smiles) for r in chunk]
        t_chunk = time.time()
        try:
            ## parallel lists of equal length - the contract predict_affinity
            ## documents and the one its own CLI gets wrong
            preds = plapt.predict_affinity(seqs, smis,
                                           prot_batch_size=args.prot_batch,
                                           mol_batch_size=args.mol_batch)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:200]}"
            log(f"  chunk {start}-{start + len(chunk)} FAILED - {msg}")
            failures.extend({"benchmark_id": r.benchmark_id, "error": msg} for r in chunk)
            continue

        if len(preds) != len(chunk):
            msg = f"PLAPT returned {len(preds)} predictions for {len(chunk)} pairs"
            log(f"  {msg}")
            failures.extend({"benchmark_id": r.benchmark_id, "error": msg} for r in chunk)
            continue

        elapsed = time.time() - t_chunk
        for r, p in zip(chunk, preds):
            write_json(os.path.join(args.out_dir, f"{r.benchmark_id}.json"), {
                "benchmark_id": r.benchmark_id,
                ## renamed: PLAPT's own key is neg_log10_affinity_M, which is the
                ## benchmark's ground-truth column name
                "plapt_pk": float(p["neg_log10_affinity_M"]),
                "plapt_affinity_uM": float(p["affinity_uM"]),
                "meta": {"device": args.device, "sec_per_pair": elapsed / len(chunk)},
            })
        n_done += len(chunk)
        log(f"  [{n_done}/{len(todo)}] {elapsed:.1f}s for {len(chunk)} pairs "
            f"({elapsed / len(chunk):.3f}s/pair)")

    if failures:
        path = os.path.join(RESULTS, f"plapt_failures_shard{shard_id:02d}.csv")
        pd.DataFrame(failures).to_csv(path, index=False)
        log(f"{len(failures)} failures -> {path}")

    log(f"shard {shard_id} done: {n_done} scored, {len(failures)} failed, "
        f"{(time.time() - t0) / 60:.1f} min")
    log("Next: python scripts/06_merge_scores.py")


if __name__ == "__main__":
    main()
