#!/usr/bin/env python
"""Fold-based protein embeddings, SLURM-array-shardable, up to 12 GPUs at once.

    # ALWAYS run this on ONE protein before submitting the array - see "Verification
    # status" below for exactly what is and is not confirmed working.
    python embed_structure_encoders.py --mode esmfold2 --selfcheck
    python embed_structure_encoders.py --mode saprot   --selfcheck

    # then, as a SLURM array (sbatch/embed_structure_encoders.sbatch):
    python embed_structure_encoders.py --mode esmfold2 --manifest proteins.txt
    python embed_structure_encoders.py --mode esmfold2 --manifest proteins.txt --merge

Two modes
---------
esmfold2   Synthyra/ESMFold2's embed_dataset(), pooled per-residue trunk states -> 256-d.
           VERIFIED working this session: loads, embeds, returns EmbeddingResult with
           order-preserving .records, pooling=("mean",) gives a (256,) vector per
           sequence. Was excluded from the earlier encoder sweep purely on cost
           (4.92 s/protein measured => 28.9 GPU-hours for 21,140 proteins), never on
           doubt about correctness - safe to run at scale.

saprot     Fold -> Foldseek 3Di tokens -> westlake-repl/SaProt_650M_AF2 (structure-AWARE
           sequence tokens, sidesteps ever choosing a single pocket - see the "structure
           embeddings" discussion earlier in this project). This path is NOT verified
           end-to-end - it depends on ESMFold2's fold_protein() returning a structure in
           a form this script can turn into a PDB (only .ptm/.plddt were confirmed from
           documentation, not the coordinate output) and on a Foldseek binary being on
           $PATH. `--selfcheck` runs the WHOLE chain on one protein and tells you exactly
           which step failed rather than three hours into an array job.

Manifest
--------
One protein sequence per line (build it once, e.g. from the training notebook's cached
`work/static.npz`-adjacent protein list, or from `pairs_clean.parquet`). Order is
preserved and is what `--merge` reconstructs, so do not regenerate the manifest between
submitting the array and merging - a changed manifest silently produces a misaligned
embedding matrix that nothing downstream would catch.

Sharding
--------
Shard count comes from SLURM_ARRAY_TASK_COUNT, shard index from SLURM_ARRAY_TASK_ID -
NEVER hardcode a shard count. This repo has already lost an entire benchmark run to
exactly that mistake once (savant_v2/README, the NUM_SHARDS incident): `--array=N-M`
alone must be what decides how the manifest is divided, so the two cannot disagree.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "work" / "emb"
OUT.mkdir(parents=True, exist_ok=True)

ESMFOLD2_HF = "Synthyra/ESMFold2"
SAPROT_HF = "westlake-repl/SaProt_650M_AF2"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def shard_count_and_index(args) -> tuple[int, int]:
    """Explicit --n-shards/--shard-id win; SLURM env vars are the default.

    Refusing to silently default to 1 shard is deliberate - a forgotten --array would
    otherwise "work" on a laptop and process 1/12th of the manifest on the cluster with
    no error.

    EXPLICIT BEATS SLURM, which is the opposite of what this function used to do, and the
    reason matters: re-running ONE failed shard of a 12-shard array is the single most
    likely repair, and `sbatch --array=9` sets SLURM_ARRAY_TASK_COUNT=1, not 12. Under the
    old precedence that job would quietly re-shard the whole manifest into one piece,
    embed all 23,097 proteins on one GPU, and write `<mode>_shard0of1.npz` - a file that
    merges into nothing and costs days of GPU time. Passing --n-shards/--shard-id is an
    unambiguous act; the environment is a default, so the act wins and any disagreement is
    reported rather than resolved in silence.
    """
    env_n = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    env_i = os.environ.get("SLURM_ARRAY_TASK_ID")
    have_env = env_n is not None and env_i is not None
    have_arg = args.n_shards is not None and args.shard_id is not None

    if have_arg:
        if have_env and (int(env_n), int(env_i)) != (args.n_shards, args.shard_id):
            log(f"NOTE: using explicit --n-shards {args.n_shards} --shard-id "
                f"{args.shard_id}, overriding SLURM_ARRAY_TASK_COUNT={env_n}/"
                f"SLURM_ARRAY_TASK_ID={env_i}. This is the correct way to re-run a single "
                f"failed shard of a larger array.")
        return args.n_shards, args.shard_id

    if args.n_shards is not None or args.shard_id is not None:
        sys.exit("--n-shards and --shard-id must be given together (got only one). "
                 "Half an override is never what you meant.")

    if have_env:
        if int(env_n) == 1 and OUT.exists():
            ## A 1-task array is legitimate for a tiny manifest, but it is also exactly
            ## what `sbatch --array=9` looks like when someone meant "redo shard 9 of 12".
            ## If sibling shard files from a wider run are sitting in OUT, say so.
            import re as _re
            wider = {int(m.group(2)) for f in OUT.glob("*_shard*of*.npz")
                     if (m := _re.search(r"_shard(\d+)of(\d+)\.npz$", f.name))
                     and int(m.group(2)) > 1}
            if wider:
                log(f"WARNING: SLURM_ARRAY_TASK_COUNT=1, so this job will treat the WHOLE "
                    f"manifest as one shard - but {sorted(wider)}-way shard files already "
                    f"exist in {OUT}. If you meant to re-run one shard of a "
                    f"{sorted(wider)[0]}-way array, cancel this and pass "
                    f"--n-shards {sorted(wider)[0]} --shard-id <k> explicitly.")
        return int(env_n), int(env_i)

    sys.exit("no shard info: set --array on sbatch (SLURM_ARRAY_TASK_COUNT/_ID) or pass "
             "--n-shards/--shard-id explicitly for a manual/local run")


## IUPAC one-letter codes: 20 standard, plus the ambiguity/rare codes that legitimately
## appear in UniProt sequences (X unknown, B Asx, Z Glx, J Xle, U selenocysteine,
## O pyrrolysine). Anything outside this is a malformed manifest, not a real residue.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def load_manifest(path: Path, on_invalid: str = "fail") -> tuple[list[str], np.ndarray]:
    """Read and VALIDATE the manifest before any model is loaded.

    Returns (sequences, valid_mask). With on_invalid="skip", invalid sequences are KEPT
    in the list (so positional alignment with the protein index is preserved) but marked
    False in the mask; the caller emits a zero vector for them rather than folding
    nonsense. Deleting the offending lines instead would shift every subsequent protein
    by one and silently corrupt every embedding after it - which is far worse than one
    zero row.

    Validation is here rather than left to the encoder because the failure it catches is
    both easy to cause and expensive to discover late: the manifest is the one hand-built
    input in this pipeline, and a malformed one otherwise surfaces deep inside fastplms
    ("invalid symbols: [...]") only AFTER the array job has been scheduled and the model
    weights loaded on every shard - burning GPU allocation to report a text-formatting
    mistake. The specific case that prompted this: numpy's ndarray.tofile(sep='\\n') on an
    object-dtype string array writes repr() per element on some numpy versions, wrapping
    every sequence in quotes.
    """
    raw = [l.strip() for l in open(path) if l.strip()]
    if not raw:
        sys.exit(f"{path} is empty")

    bad = []
    for i, s in enumerate(raw):
        offenders = sorted(set(s.upper()) - VALID_AA)
        if offenders:
            bad.append((i + 1, offenders, s[:60]))

    if bad and on_invalid == "skip":
        ## Genuine data contaminants (e.g. a modified-peptide annotation like
        ## "biotin-C6linker-VRRLRRLTAREAA" sitting in a sequence field) are real and rare.
        ## Skipping them keeps the array job moving without disturbing the index.
        log(f"WARNING: {len(bad):,} of {len(raw):,} sequences contain non-amino-acid "
            f"characters and will get ZERO embeddings (--on-invalid skip):")
        for ln, off, snippet in bad[:5]:
            log(f"    line {ln}: invalid {off} in {snippet!r}")
        if len(bad) > 5:
            log(f"    ... and {len(bad)-5:,} more")
        mask = np.ones(len(raw), bool)
        for ln, _, _ in bad:
            mask[ln - 1] = False
        lens = np.array([len(s) for s in raw])
        log(f"manifest: {len(raw):,} proteins, {int(mask.sum()):,} embeddable "
            f"(length min {lens.min():,} / median {int(np.median(lens)):,} / max {lens.max():,})")
        return raw, mask

    if bad:
        quoted = sum(1 for _, o, _ in bad if set(o) <= {"'", '"'})
        msg = [f"{len(bad):,} of {len(raw):,} manifest lines contain non-amino-acid "
               f"characters.", ""]
        for ln, off, snippet in bad[:5]:
            msg.append(f"  line {ln}: invalid {off} in {snippet!r}")
        if len(bad) > 5:
            msg.append(f"  ... and {len(bad)-5:,} more")
        msg += ["", "If these are genuine data contaminants rather than a formatting "
                "problem, re-run with --on-invalid skip: the sequences stay in place "
                "(preserving index alignment) and receive zero embeddings."]
        if quoted:
            msg += ["", "Every offending line's only bad characters are quotes - this is the",
                    "numpy tofile() repr() problem. Rebuild the manifest with an explicit join:",
                    "",
                    "    python -c \"",
                    "    import numpy as np",
                    "    a = np.load('work/static.npz', allow_pickle=True)['proteins']",
                    "    open('work/protein_manifest.txt','w').write("
                    "'\\n'.join(str(s) for s in a))",
                    "    \"",
                    "",
                    "Do NOT use ndarray.tofile(sep='\\n') - its behaviour on object-dtype",
                    "string arrays is numpy-version-dependent and writes repr() on some."]
        sys.exit("\n".join(msg))

    lens = np.array([len(s) for s in raw])
    log(f"manifest: {len(raw):,} proteins, all residues valid "
        f"(length min {lens.min():,} / median {int(np.median(lens)):,} / max {lens.max():,})")
    return raw, np.ones(len(raw), bool)


def my_shard(seqs: list[str], n_shards: int, shard_id: int) -> tuple[np.ndarray, list[str]]:
    idx = np.arange(shard_id, len(seqs), n_shards)     # interleaved, not contiguous blocks:
    return idx, [seqs[i] for i in idx]                 # keeps shards near-equal even if
                                                        # sequence length correlates with
                                                        # position in the manifest


# --------------------------------------------------------------------------- #
# esmfold2 - verified path
# --------------------------------------------------------------------------- #

def embed_esmfold2(seqs: list[str], batch_size: int = 2) -> np.ndarray:
    import torch
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("  WARNING: no GPU visible - esmfold2 on CPU is not the measured regime "
            "(4.92 s/protein was GPU); expect this to be far slower")
    mdl = AutoModel.from_pretrained(ESMFOLD2_HF, trust_remote_code=True).to(dev).eval()

    out = np.zeros((len(seqs), 256), np.float32)
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        res = mdl.embed_dataset(chunk, batch_size=len(chunk), full_embeddings=False,
                                pooling=("mean",))
        for j, rec in enumerate(res.records):
            t = rec.tensor if rec.tensor is not None else rec.load_tensor()
            out[i + j] = np.asarray(t.float().cpu())
        if (i // batch_size) % 20 == 0:
            log(f"    {min(i+batch_size, len(seqs)):,}/{len(seqs):,}")
    del mdl
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# saprot - best-effort path, gated by --selfcheck
# --------------------------------------------------------------------------- #

def _fold_to_pdb(seq: str, tmp_pdb: Path) -> None:
    """Fold one sequence and write a PDB. UNVERIFIED beyond what is checked here.

    ESMFold2's embed_dataset() (used above) was confirmed live this session; its
    structure-prediction entry point (documented as `fold_protein`, returning an object
    with `.ptm`/`.plddt`) was only read from documentation, never called - so the actual
    coordinate output shape is unknown. This function tries the plausible attribute/
    method names in order and raises with the object's real attribute list if none work,
    so `--selfcheck` tells you which one to add rather than failing an array job cold.
    """
    import torch
    from transformers import AutoModel

    mdl = AutoModel.from_pretrained(ESMFOLD2_HF, trust_remote_code=True)
    mdl = mdl.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if not hasattr(mdl, "fold_protein"):
        raise RuntimeError(
            f"{ESMFOLD2_HF} has no fold_protein() method. Available callables: "
            f"{[a for a in dir(mdl) if not a.startswith('_') and callable(getattr(mdl, a))]}")
    result = mdl.fold_protein(seq, num_loops=1, num_sampling_steps=200)

    for attr in ("to_pdb", "as_pdb"):
        if hasattr(result, attr):
            pdb_str = getattr(result, attr)()
            tmp_pdb.write_text(pdb_str if isinstance(pdb_str, str) else pdb_str.decode())
            return
    for attr in ("pdb", "pdb_string"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            tmp_pdb.write_text(val() if callable(val) else val)
            return
    raise RuntimeError(
        "fold_protein() returned an object with no recognised PDB export "
        f"(tried .to_pdb()/.as_pdb()/.pdb/.pdb_string). Actual attributes: "
        f"{[a for a in dir(result) if not a.startswith('_')]}\n"
        "Add whichever of those is the real coordinate/PDB accessor to _fold_to_pdb() "
        "before running saprot mode at scale.")


def _run_foldseek_3di(pdb_path: Path) -> str:
    """Foldseek's 3Di alphabet string for chain A of one PDB, via structureto3didescriptor."""
    if shutil.which("foldseek") is None:
        raise RuntimeError(
            "`foldseek` is not on $PATH. Install it before using --mode saprot:\n"
            "    conda install -c conda-forge -c bioconda foldseek\n"
            "or download a static binary from https://github.com/steineggerlab/foldseek")
    with tempfile.TemporaryDirectory() as td:
        out_tsv = Path(td) / "3di.tsv"
        r = subprocess.run(["foldseek", "structureto3didescriptor", str(pdb_path), str(out_tsv)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"foldseek exited {r.returncode}: {r.stderr[:500]}")
        if not out_tsv.exists():
            raise RuntimeError(f"foldseek reported success but wrote no {out_tsv.name} - "
                               f"inspect stdout: {r.stdout[:500]}")
        line = out_tsv.read_text().splitlines()[0]
        ## column layout per Foldseek's own docs: name, 3di-seq, aa-seq, ... - verify the
        ## column count rather than assume, since a foldseek version change would
        ## otherwise silently hand SaProt the wrong column as "3Di"
        cols = line.split("\t")
        if len(cols) < 2:
            raise RuntimeError(f"unexpected structureto3didescriptor output: {line[:200]}")
        return cols[1]


def _combine_aa_3di(aa: str, di: str) -> str:
    if len(aa) != len(di):
        raise RuntimeError(f"AA length {len(aa)} != 3Di length {len(di)} - a chain-break "
                           "or multi-chain PDB will do this; single-chain monomers only")
    ## SaProt's documented token format: each residue is AA-char + 3Di-char, e.g. "MpEv..."
    return "".join(a + d for a, d in zip(aa, di))


def embed_saprot(seqs: list[str], batch_size: int = 4) -> np.ndarray:
    import torch
    from transformers import EsmForMaskedLM, EsmTokenizer

    tok = EsmTokenizer.from_pretrained(SAPROT_HF)
    mdl = EsmForMaskedLM.from_pretrained(SAPROT_HF, output_hidden_states=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()

    combined = []
    with tempfile.TemporaryDirectory() as td:
        for i, seq in enumerate(seqs):
            pdb = Path(td) / f"p{i}.pdb"
            _fold_to_pdb(seq, pdb)
            di = _run_foldseek_3di(pdb)
            combined.append(_combine_aa_3di(seq, di))
            if (i + 1) % 50 == 0:
                log(f"    folded+3Di {i+1:,}/{len(seqs):,}")

    out = np.zeros((len(seqs), mdl.config.hidden_size), np.float32)
    with torch.inference_mode():
        for i in range(0, len(combined), batch_size):
            chunk = combined[i:i + batch_size]
            b = tok(chunk, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mdl(**b).hidden_states[-1]
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = ((h * m).sum(1) / m.sum(1).clamp_min(1)).float().cpu().numpy()
            out[i:i + len(chunk)] = pooled
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SELFCHECK_SEQ = ("MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEGRTWAQ"
                 "PGYPWPLYGNEGCGWAGWLLSPRGSRPSWGPTDPRRRSRNLGKVIDTLTCGFADLMGYIPLVGAPLGGAARALAHGVR")


def cmd_run(args):
    mode = args.mode
    if args.selfcheck:
        ## Two proteins, not one - the first call pays for the model weights loading
        ## (seconds to tens of seconds, a ONE-TIME cost per job, not per protein). Timing
        ## only the second call is what makes a GPU-hour projection meaningful; timing the
        ## first call as if it were steady-state throughput overstated a real run's cost
        ## by ~12x when this was actually run (338 GPU-hours projected vs the 28.9 GPU-hours
        ## separately measured on a warm model over 16 real sequences).
        fn = embed_esmfold2 if mode == "esmfold2" else embed_saprot
        log(f"selfcheck: {mode}, loading model (one-time cost, excluded from the rate below)")
        t0 = time.time()
        try:
            _ = fn([SELFCHECK_SEQ])
        except Exception as e:
            log(f"SELFCHECK FAILED (during model load / first call): {type(e).__name__}: {e}")
            sys.exit(1)
        load_s = time.time() - t0
        log(f"  model load + first protein: {load_s:.1f}s")

        t1 = time.time()
        try:
            emb = fn([SELFCHECK_SEQ])
        except Exception as e:
            log(f"SELFCHECK FAILED (warm call): {type(e).__name__}: {e}")
            sys.exit(1)
        warm_s = time.time() - t1
        ok = emb.shape[0] == 1 and np.isfinite(emb).all()
        log(f"selfcheck {'PASSED' if ok else 'FAILED'}: shape {emb.shape}, "
            f"finite {np.isfinite(emb).all()}, warm call {warm_s:.1f}s/protein")
        log(f"  projected for 21,140 proteins on ONE GPU: {21140*warm_s/3600:.1f} GPU-hours "
            f"(model-load overhead excluded, as it happens once per array-shard job, not "
            f"once per protein)")
        if mode == "esmfold2":
            log("  cross-check: separately measured at 4.92 s/protein on 16 real sequences "
                "of median corpus length AT BATCH SIZE 4 => ~28.9 GPU-hours total. This "
                "selfcheck necessarily runs batch size 1 (one protein), and running one at a "
                "time this session measured ~31 s/protein - roughly 6x slower, consistent "
                "with small-batch GPU underutilisation rather than a real regression. Do not "
                "plan the array's GPU-hour budget off this number: run a handful of real "
                "shards at the --batch-size you intend to submit with, and use THAT rate.")
        sys.exit(0 if ok else 1)

    if not args.manifest:
        sys.exit("--manifest is required outside --selfcheck")
    seqs, valid_mask = load_manifest(Path(args.manifest), args.on_invalid)

    if args.merge:
        ## Merge runs once, after the array finishes, typically outside any SLURM array
        ## context - it needs to know how many shards to expect, not which one it is, so
        ## it must not demand a shard-id the way embedding a single shard does.
        n_shards = (int(os.environ["SLURM_ARRAY_TASK_COUNT"]) if "SLURM_ARRAY_TASK_COUNT"
                   in os.environ else args.n_shards)
        if n_shards is None:
            sys.exit("--merge needs to know the shard count: pass --n-shards N "
                     "(SLURM_ARRAY_TASK_COUNT is also honoured if set)")
        def shard_inventory() -> str:
            """Everything on disk that looks like a shard, whatever count it claims.

            Printed on any merge failure. The failure is almost always a mismatch between
            the shard count being merged and the one the files were written with, and that
            is invisible unless the actual filenames are shown.
            """
            import re as _re
            rows = []
            for f in sorted(OUT.glob(f"{mode}_shard*of*.npz")):
                m = _re.search(r"_shard(\d+)of(\d+)\.npz$", f.name)
                if not m:
                    continue
                try:
                    n_rows = len(np.load(f)["idx"])
                except Exception as e:
                    n_rows = f"unreadable ({e})"
                rows.append((int(m.group(2)), int(m.group(1)), f.name, n_rows))
            if not rows:
                return f"  (no {mode}_shard*of*.npz files in {OUT} at all)"
            out = []
            for cnt in sorted({r[0] for r in rows}):
                have = sorted(r[1] for r in rows if r[0] == cnt)
                gaps = [s for s in range(cnt) if s not in have]
                out.append(f"  {cnt}-way run: {len(have)}/{cnt} files present"
                           + (f", MISSING shard ids {gaps}" if gaps else ", complete"))
                for _, sid, name, n in sorted((r for r in rows if r[0] == cnt),
                                              key=lambda r: r[1]):
                    out.append(f"      {name}  ({n} rows)")
            return "\n".join(out)

        parts = []
        for s in range(n_shards):
            p = OUT / f"{mode}_shard{s}of{n_shards}.npz"
            if not p.exists():
                sys.exit(f"missing {p.name} - not all shards have finished.\n\n"
                         f"what is actually in {OUT}:\n{shard_inventory()}\n\n"
                         f"If the files above say a different N than the {n_shards} being "
                         f"merged, re-run merge with --n-shards <that N>. If a shard id is "
                         f"genuinely missing, re-run just that one with\n"
                         f"    python scripts/embed_structure_encoders.py --mode {mode} "
                         f"--manifest <manifest> --n-shards {n_shards} --shard-id <k>\n"
                         f"passing --n-shards/--shard-id EXPLICITLY - `sbatch --array=k` "
                         f"alone sets SLURM_ARRAY_TASK_COUNT=1 and would re-shard the whole "
                         f"manifest into one piece.")
            z = np.load(p)
            ## shard files written before --on-invalid existed have no "valid" key
            val = z["valid"] if "valid" in z.files else np.ones(len(z["idx"]), bool)
            parts.append((z["idx"], z["emb"], val))
        dim = parts[0][1].shape[1]
        full = np.zeros((len(seqs), dim), np.float32)
        filled = np.zeros(len(seqs), bool)
        valid_all = np.ones(len(seqs), bool)
        for idx, emb, val in parts:
            full[idx] = emb
            filled[idx] = True
            valid_all[idx] = val
        if not filled.all():
            ## Work out WHICH shard the hole belongs to instead of guessing. Sharding is
            ## interleaved (shard s owns indices s, s+n, s+2n, ...), so every index a given
            ## shard should have covered has the same residue mod n_shards. If all the
            ## missing indices collapse to one or two residues, the shard files for those
            ## ids are present but did not contribute their rows - which is a different
            ## problem, with a different fix, from "a shard never ran".
            miss = np.where(~filled)[0]
            resid = np.unique(miss % n_shards)
            expected = {int(r): int(len(range(int(r), len(seqs), n_shards))) for r in resid}
            detail = ", ".join(f"shard {r}: {int((miss % n_shards == r).sum())}"
                               f"/{expected[int(r)]} rows missing" for r in resid)
            stale = []
            for s in range(n_shards):
                want = np.arange(s, len(seqs), n_shards)
                got = parts[s][0]
                if len(got) != len(want):
                    stale.append(f"      {mode}_shard{s}of{n_shards}.npz holds {len(got)} "
                                 f"indices but shard {s} of this manifest has {len(want)} "
                                 f"- written against a manifest of a different length")
                elif not np.array_equal(got, want):
                    ## Same COUNT, different indices: the file was written by a job that
                    ## thought it was a different shard, or used a different shard count
                    ## that happens to divide out the same. This is the dangerous case -
                    ## nothing about the file's name or size gives it away.
                    stale.append(f"      {mode}_shard{s}of{n_shards}.npz holds the right "
                                 f"NUMBER of indices ({len(got)}) but the wrong ones "
                                 f"(first is {int(got[0])}, shard {s} should start at {s}) "
                                 f"- this file belongs to a different run")
            sys.exit(
                f"{len(miss)} of {len(seqs)} proteins never appeared in any shard.\n\n"
                f"missing rows by shard id: {detail}\n\n"
                + ("shard files whose contents do not match this manifest:\n"
                   + "\n".join(stale) + "\n\n" if stale else
                   "every shard file's indices match this manifest individually, so the "
                   "shard COUNT is the thing that disagrees.\n\n")
                + f"what is actually in {OUT}:\n{shard_inventory()}\n\n"
                f"Most likely: the array ran with a different number of tasks than the "
                f"{n_shards} being merged, or one task failed. Re-run the affected shard(s) "
                f"with EXPLICIT flags:\n"
                f"    python scripts/embed_structure_encoders.py --mode {mode} "
                f"--manifest {args.manifest} --n-shards {n_shards} --shard-id <k>\n"
                f"`sbatch --array=k` on its own sets SLURM_ARRAY_TASK_COUNT=1 and would "
                f"re-shard the entire manifest into a single piece.")
        dest = OUT / f"{mode}.npy"
        np.save(dest, full)
        ## Write the key file alongside. Row i of `full` is manifest line i, so the
        ## manifest IS the key list. Without this, any downstream script whose protein
        ## index differs from the manifest by even one sequence can only compare row
        ## counts, has to reject the whole cache, and re-folds 23k proteins for nothing -
        ## which is exactly what run_2x2_split_test.py did the first time it saw this file.
        np.save(OUT / f"{mode}_keys.npy", np.asarray(seqs, object))
        log(f"merged {n_shards} shards -> {dest} {full.shape}")
        log(f"wrote {mode}_keys.npy ({len(seqs):,} sequences) so a different protein "
            f"index can subset this cache instead of re-folding")
        if not valid_all.all():
            ## Saved alongside so downstream code can mask these rows out rather than
            ## silently treating a zero vector as a real embedding.
            vdest = OUT / f"{mode}_valid.npy"
            np.save(vdest, valid_all)
            log(f"  {int((~valid_all).sum()):,} rows are ZERO (invalid sequences) - "
                f"mask written to {vdest.name}")
        return

    n_shards, shard_id = shard_count_and_index(args)
    log(f"shard {shard_id}/{n_shards}")
    idx, my_seqs = my_shard(seqs, n_shards, shard_id)
    my_valid = valid_mask[idx]
    log(f"this shard: {len(my_seqs):,} proteins"
        + (f", {int((~my_valid).sum())} skipped as invalid" if not my_valid.all() else ""))

    fn = embed_esmfold2 if mode == "esmfold2" else embed_saprot
    if my_valid.all():
        emb = fn(my_seqs, args.batch_size)
    else:
        ## Fold only the valid ones, then scatter back into full-shard positions so row
        ## k of this shard still corresponds to idx[k] globally. Invalid rows stay zero
        ## and are recorded in the shard file for downstream masking.
        good_pos = np.where(my_valid)[0]
        good_emb = fn([my_seqs[i] for i in good_pos], args.batch_size)
        emb = np.zeros((len(my_seqs), good_emb.shape[1]), np.float32)
        emb[good_pos] = good_emb

    dest = OUT / f"{mode}_shard{shard_id}of{n_shards}.npz"
    np.savez_compressed(dest, idx=idx, emb=emb, valid=my_valid)
    log(f"wrote {dest} {emb.shape}"
        + (f" ({int((~my_valid).sum())} zero rows)" if not my_valid.all() else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["esmfold2", "saprot"], required=True)
    ap.add_argument("--manifest", help="one protein sequence per line")
    ap.add_argument("--merge", action="store_true", help="merge all shards instead of embedding")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the whole chain on one protein and report pass/fail")
    ap.add_argument("--n-shards", type=int, help="manual override (default: SLURM env)")
    ap.add_argument("--shard-id", type=int, help="manual override (default: SLURM env)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--on-invalid", choices=["fail", "skip"], default="fail",
                    help="sequences with non-amino-acid characters: fail fast (default), "
                         "or keep them in place with a zero embedding to preserve index "
                         "alignment")
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
