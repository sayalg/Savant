#!/usr/bin/env python
"""
Step 02 - Fetch one MSA per unique target, once, and cache it as .a3m.

The 1000 pairs cover only 355 targets, so letting `boltz predict
--use_msa_server` do this would issue ~1000 ColabFold queries for ~355 distinct
answers and would re-issue all of them on every re-run. This fetches each
sequence once into work/msa/<uniprot>.a3m, which the YAMLs from step 01 point
at, and skips anything already on disk - so an interrupted run resumes.

It reuses Boltz's own ColabFold client (`boltz.data.msa.mmseqs2.run_mmseqs2`),
so the alignments are byte-for-byte what --use_msa_server would have produced.

NOTE: this needs outbound internet. On clusters where compute nodes are
firewalled, run it on a login node (it needs no GPU and little memory).

    python 02_fetch_msas.py                  # all missing targets
    python 02_fetch_msas.py --batch-size 20  # sequences per server request
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import pandas as pd

from common import MSA_DIR, TARGETS_CSV, WORK, ensure_dirs, log


def load_client():
    try:
        from boltz.data.msa.mmseqs2 import run_mmseqs2
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "Could not import boltz's MSA client. Activate the environment that "
            f"has boltz installed, or run step 01 with --msa-mode server.\n  {exc}"
        )
    return run_mmseqs2


def fetch_batch(run_mmseqs2, sequences: list[str], prefix: str, host_url: str) -> list[str]:
    """Return one a3m string per input sequence."""
    result = run_mmseqs2(
        sequences,
        prefix=prefix,
        use_env=True,
        use_filter=True,
        use_pairing=False,
        host_url=host_url,
    )
    # Boltz returns (a3m_lines, template_paths) in some versions and a bare list
    # of a3m strings in others.
    if isinstance(result, tuple):
        result = result[0]
    if len(result) != len(sequences):
        raise RuntimeError(
            f"MSA server returned {len(result)} alignments for {len(sequences)} sequences"
        )
    return list(result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default=TARGETS_CSV, help="from step 01")
    ap.add_argument("--msa-dir", default=MSA_DIR, help="output directory for <uniprot>.a3m")
    ap.add_argument("--batch-size", type=int, default=20,
                    help="sequences per server request; keep modest to stay polite")
    ap.add_argument("--host-url", default="https://api.colabfold.com")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between batches")
    ap.add_argument("--limit", type=int, default=None, help="only N missing targets")
    args = ap.parse_args()

    ensure_dirs(WORK, args.msa_dir)
    targets = pd.read_csv(args.targets)

    missing = [
        (t.uniprot_id, t.sequence) for t in targets.itertuples(index=False)
        if not os.path.exists(os.path.join(args.msa_dir, f"{t.uniprot_id}.a3m"))
        or os.path.getsize(os.path.join(args.msa_dir, f"{t.uniprot_id}.a3m")) == 0
    ]
    log(f"{len(targets)} targets, {len(targets) - len(missing)} cached, {len(missing)} to fetch")
    if args.limit:
        missing = missing[: args.limit]
    if not missing:
        return

    # Two accessions in the benchmark share one sequence; query it once and
    # write the answer out under both names.
    by_sequence: dict[str, list[str]] = {}
    for accession, sequence in missing:
        by_sequence.setdefault(sequence, []).append(accession)
    sequences = list(by_sequence)
    log(f"{len(sequences)} distinct sequences to query")

    tmp_root = os.path.join(WORK, "msa_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    n_written, n_failed = 0, 0

    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        prefix = os.path.join(tmp_root, f"batch_{start:04d}")
        label = f"batch {start // args.batch_size + 1}/{-(-len(sequences) // args.batch_size)}"
        try:
            log(f"{label}: querying {len(batch)} sequences ...")
            alignments = fetch_batch(load_client(), batch, prefix, args.host_url)
        except Exception as exc:
            n_failed += len(batch)
            log(f"{label}: FAILED ({type(exc).__name__}: {exc}) - re-run to retry")
            continue
        finally:
            shutil.rmtree(prefix, ignore_errors=True)

        for sequence, a3m in zip(batch, alignments):
            for accession in by_sequence[sequence]:
                path = os.path.join(args.msa_dir, f"{accession}.a3m")
                tmp = f"{path}.tmp"
                with open(tmp, "w", newline="\n") as fh:
                    fh.write(a3m)
                os.replace(tmp, path)
                n_written += 1
        log(f"{label}: wrote {len(batch)} alignments ({n_written} total)")
        time.sleep(args.sleep)

    log(f"done: {n_written} a3m files in {args.msa_dir}, {n_failed} sequences failed")
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
