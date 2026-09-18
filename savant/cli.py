"""Command line: one protein, many ligands, JSON and/or CSV out.

    savant --seq MTVPDRSEI... --smiles "CCO"
    savant --seq target.fasta --smiles library.smi --out screen.json
    savant --seq target.fasta --smiles library.smi --csv screen.csv --no-json

Progress and summaries go to stderr, so `--out -` or `--csv -` writes clean
output to stdout and stays pipeable:

    savant --seq target.fasta --smiles library.smi --out - | jq '.targets[0].predictions[:5]'
"""
from __future__ import annotations

import argparse
import sys
import time

from savant.features import parse_smiles
from savant.inputs import dedupe_ligands, read_ligands, read_targets
from savant.model import DEFAULT_MODEL, MODELS, Savant
from savant.report import build_report, build_table, write_report, write_table

DEFAULT_OUT = "savant_predictions.json"


def log(msg):
    """Progress to stderr, so stdout can carry JSON or CSV."""
    print(msg, file=sys.stderr, flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        prog="savant",
        description="Structure-free protein-ligand binding affinity (pKd) "
                    "prediction from a sequence and SMILES.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group("input")
    io.add_argument("-s", "--seq", required=True, metavar="SEQ|FILE",
                    help="amino-acid sequence, or a path to a FASTA, plain-text "
                         "or PDB file. A multi-record FASTA scores every ligand "
                         "against every record.")
    io.add_argument("-l", "--smiles", required=True, nargs="+", metavar="SMILES|FILE",
                    help="one or more SMILES, and/or paths to .smi/.txt files "
                         "(one SMILES per line, optional name after it) or "
                         ".csv/.tsv files with a SMILES column. Values may be "
                         "mixed freely.")
    io.add_argument("--protein-chain", default="A",
                    help="chain to read when --seq is a PDB (default: A)")
    io.add_argument("--keep-duplicates", action="store_true",
                    help="score repeated SMILES more than once (default: drop them)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--out", default=DEFAULT_OUT, metavar="FILE",
                     help=f"JSON output path, or '-' for stdout "
                          f"(default: {DEFAULT_OUT})")
    out.add_argument("-c", "--csv", metavar="FILE",
                     help="also write a CSV here, or '-' for stdout. One row per "
                          "(target, ligand); the sequence is not repeated per row")
    out.add_argument("--no-json", action="store_true",
                     help="skip the JSON, e.g. when you only want --csv")
    out.add_argument("--precision", type=int, default=4,
                     help="decimal places for pKd and Kd_nM (default: 4)")
    out.add_argument("--no-sequence", action="store_true",
                     help="omit the sequence from the JSON; keeps id, length and "
                          "source")
    out.add_argument("--top", type=int, default=None, metavar="N",
                     help="keep only the N best-ranked ligands per target")
    out.add_argument("-q", "--quiet", action="store_true",
                     help="suppress progress on stderr")

    mdl = p.add_argument_group("model")
    mdl.add_argument("-m", "--model", choices=list(MODELS), default=DEFAULT_MODEL,
                     help=f"protein encoder variant (default: {DEFAULT_MODEL})")
    mdl.add_argument("--models", metavar="DIR|PKL",
                     help="override the model directory or point at one bundle "
                          ".pkl (env: SAVANT_MODEL_DIR)")
    mdl.add_argument("--ligand-batch", type=int, default=64,
                     help="ligands per ChemBERTa forward pass (default: 64)")
    mdl.add_argument("--protein-batch", type=int, default=8,
                     help="proteins per ESM2 forward pass (default: 8)")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    say = (lambda _m: None) if args.quiet else log

    if args.no_json and not args.csv:
        parser.error("--no-json leaves nothing to write; add --csv FILE")
    if args.csv == "-" and not args.no_json and args.out == "-":
        parser.error("--out - and --csv - would interleave on stdout; "
                     "send one of them to a file")

    ## Read and validate every input before loading a 2.5 GB encoder: a typo in a
    ## path or an unparseable SMILES should cost a second, not a model load.
    try:
        targets = read_targets(args.seq, protein_chain=args.protein_chain)
        ligands = read_ligands(args.smiles)
    except (OSError, ValueError) as exc:
        sys.exit(f"savant: {exc}")

    if not args.keep_duplicates:
        ligands, dropped = dedupe_ligands(ligands)
        if dropped:
            say(f"dropped {dropped} duplicate SMILES "
                f"({len(ligands)} unique); --keep-duplicates to score them all")

    try:
        parse_smiles([lig.smiles for lig in ligands])
    except ValueError as exc:
        sys.exit(f"savant: {exc}")

    n_pairs = len(targets) * len(ligands)
    say(f"{len(targets)} target(s) x {len(ligands)} ligand(s) = {n_pairs} pair(s)")

    try:
        sv = Savant(model=args.model, models_dir=args.models, quiet=args.quiet)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"savant: {exc}")
    say(f"model {sv.model}: {sv.n_features_in_} features, {sv.esm_model}")

    smiles = [lig.smiles for lig in ligands]
    names = [lig.name for lig in ligands]

    results, t0 = [], time.perf_counter()
    for i, target in enumerate(targets, 1):
        label = target.id or f"target {i}"
        say(f"[{i}/{len(targets)}] {label}: {target.length} residues, "
            f"scoring {len(smiles)} ligand(s)")
        try:
            scored = sv.screen(target.sequence, smiles,
                               ligand_batch=args.ligand_batch,
                               protein_batch=args.protein_batch)
        except (ValueError, RuntimeError) as exc:
            sys.exit(f"savant: scoring {label}: {exc}")

        ## screen() reorders by rank, so names have to follow the SMILES rather
        ## than the original position. Truncation to --top is the writer's job,
        ## so it can still record how many ligands were scored.
        by_smiles = dict(zip(smiles, names))
        scored["name"] = [by_smiles.get(s) for s in scored["smiles"]]
        results.append((target, scored))

    elapsed = time.perf_counter() - t0
    say(f"scored {n_pairs} pair(s) in {elapsed:.1f}s "
        f"({elapsed / max(n_pairs, 1):.3f}s per pair)")

    if not args.no_json:
        report = build_report(sv, results, precision=args.precision,
                              include_sequence=not args.no_sequence, top=args.top)
        written = write_report(report, args.out)
        if written:
            say(f"wrote {written}")

    if args.csv:
        table = build_table(sv, results, precision=args.precision, top=args.top)
        written = write_table(table, args.csv)
        if written:
            say(f"wrote {written} ({len(table)} rows)")

    for target, scored in results:
        best = scored.iloc[0]
        say(f"  {target.id or 'target'}: best {best['smiles'][:44]} "
            f"pKd {best['pKd']:.2f} (Kd {best['Kd_nM']:.1f} nM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
