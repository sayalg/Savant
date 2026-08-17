#!/usr/bin/env python
"""
Step 01 - Turn the benchmark CSV into one Boltz-2 YAML per pair.

Each YAML asks for exactly two chains:

    protein -> id A     (sequence, with a precomputed MSA if one is available)
    ligand  -> id B     (SMILES)

That split is the whole point of running Boltz first: Savant is invoked as
`-p A -l B` and PRODIGY-LIG as `-c A B:LIG`, and neither can work from a
single-chain model. Boltz uses the ids verbatim and writes every non-polymer
chain as HETATM/LIG in PDB output, so this contract holds end to end;
03_collect_boltz.py re-checks it on the produced structures rather than
trusting it.

Ligands are desalted (largest organic fragment) before docking: a SMILES like
`...[O-].[Na+].[Na+]` would otherwise put nine sodium ions into chain B, which
breaks PRODIGY's "one ligand residue per chain" rule and pollutes the HADDOCK
energetics Savant reads. The original string is kept in the manifest, and
`docked_smiles` - what actually entered the structure - is what gets passed to
Savant so its RDKit descriptors match the coordinates it scores.

YAMLs are written into shard directories so the Boltz array job can hand one
directory to each task; `boltz predict` loads the model once per directory.

    python 01_prepare_inputs.py --num-shards 25
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from common import (BENCHMARK_CSV, LIGAND_CHAIN, MANIFEST, MAX_AFFINITY_HEAVY_ATOMS,
                    MSA_DIR, PROTEIN_CHAIN, SHARDS_CSV, TARGETS_CSV, WORK, YAML_DIR,
                    ensure_dirs, log, shard_name, yaml_quote)

RDLogger.DisableLog("rdApp.*")

_YAML = """version: 1
sequences:
  - protein:
      id: {protein_chain}
      sequence: {sequence}
{msa_line}  - ligand:
      id: {ligand_chain}
      smiles: {smiles}
{affinity}"""

_AFFINITY = """properties:
  - affinity:
      binder: {ligand_chain}
"""


def desalt(smiles: str) -> tuple[str, bool, int]:
    """Return (largest_fragment_smiles, was_stripped, heavy_atom_count).

    Falls back to the input string if RDKit cannot parse it; step 05 records
    the parse failure rather than silently docking something else.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, False, -1
    if "." not in smiles:
        return Chem.MolToSmiles(mol), False, mol.GetNumHeavyAtoms()
    largest = rdMolStandardize.LargestFragmentChooser().choose(mol)
    if largest is None:
        return Chem.MolToSmiles(mol), False, mol.GetNumHeavyAtoms()
    return Chem.MolToSmiles(largest), True, largest.GetNumHeavyAtoms()


def build_yaml(sequence: str, smiles: str, msa_path: str | None, affinity: bool) -> str:
    msa_line = f"      msa: {yaml_quote(msa_path)}\n" if msa_path else ""
    return _YAML.format(
        protein_chain=PROTEIN_CHAIN,
        ligand_chain=LIGAND_CHAIN,
        sequence=yaml_quote(sequence),
        msa_line=msa_line,
        smiles=yaml_quote(smiles),
        affinity=_AFFINITY.format(ligand_chain=LIGAND_CHAIN) if affinity else "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=BENCHMARK_CSV, help="benchmark CSV to evaluate")
    ap.add_argument("--num-shards", type=int, default=25,
                    help="number of YAML shard directories (= Boltz array size)")
    ap.add_argument("--msa-dir", default=MSA_DIR,
                    help="directory holding <uniprot>.a3m; written into each YAML. "
                         "Must be the path as seen by the compute node.")
    ap.add_argument("--msa-mode", choices=["precomputed", "server"], default="precomputed",
                    help="precomputed: reference --msa-dir/<uniprot>.a3m (run 02 first). "
                         "server: omit the msa field, requiring `boltz predict "
                         "--use_msa_server` (one server query per pair, no reuse).")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs (smoke test)")
    args = ap.parse_args()

    ensure_dirs(WORK, YAML_DIR)

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    log(f"{len(df)} pairs, {df.uniprot_id.nunique()} unique targets from {args.csv}")

    # One MSA per target, reused by every pair that shares it (1000 pairs but
    # only 355 targets, so this is ~3x fewer server queries).
    targets = (df.drop_duplicates("uniprot_id")[["uniprot_id", "target_sequence"]]
                 .rename(columns={"target_sequence": "sequence"}))
    targets["seq_len"] = targets.sequence.str.len()
    targets.to_csv(TARGETS_CSV, index=False)
    log(f"wrote {TARGETS_CSV} ({len(targets)} targets, "
        f"{targets.seq_len.min()}-{targets.seq_len.max()} aa)")

    msa_dir = (os.path.abspath(args.msa_dir).replace(os.sep, "/")
               if args.msa_mode == "precomputed" else None)

    # Pass 1: resolve the ligand chemistry, so records can be grouped by whether
    # they get an affinity request before any shard is assigned.
    prepared, n_desalted, n_unparsed = [], 0, 0
    for r in df.itertuples(index=False):
        docked_smiles, stripped, n_heavy = desalt(r.ligand_smiles)
        n_desalted += stripped
        n_unparsed += n_heavy < 0
        # Boltz-2's affinity head is undefined past 128 heavy atoms; those pairs
        # still get a structure (which is what Savant and PRODIGY need), just no
        # affinity_*.json.
        affinity = 0 <= n_heavy <= MAX_AFFINITY_HEAVY_ATOMS
        prepared.append((r, docked_smiles, stripped, n_heavy, affinity))

    # Shards must be HOMOGENEOUS in affinity, or Boltz dies partway through.
    # It enters the affinity stage when ANY record in the input directory asks
    # for affinity, but that stage then iterates EVERY record in the directory
    # and loads a pre_affinity_<id>.npz that only affinity records are given:
    #   FileNotFoundError: .../predictions/bm_0101/pre_affinity_bm_0101.npz
    # One oversized ligand therefore takes down its whole shard. Keeping the two
    # populations apart means the affinity stage either runs over records that
    # all have the file, or is never entered at all.
    with_affinity = [p for p in prepared if p[4]]
    without_affinity = [p for p in prepared if not p[4]]

    per_shard = math.ceil(len(with_affinity) / args.num_shards) if with_affinity else 0
    n_extra = math.ceil(len(without_affinity) / per_shard) if (without_affinity and per_shard) \
        else (1 if without_affinity else 0)
    n_shards_total = args.num_shards + n_extra

    assignment = [(p, i % args.num_shards) for i, p in enumerate(with_affinity)]
    assignment += [(p, args.num_shards + (i % n_extra))
                   for i, p in enumerate(without_affinity)]

    rows = []
    for (r, docked_smiles, stripped, n_heavy, affinity), shard in assignment:
        shard_dir = os.path.join(YAML_DIR, shard_name(shard))
        os.makedirs(shard_dir, exist_ok=True)

        # Forward slashes: the YAML may be written on one OS and consumed on the
        # cluster, and a backslash inside a single-quoted scalar is a literal.
        msa_path = f"{msa_dir}/{r.uniprot_id}.a3m" if msa_dir else None
        yaml_path = os.path.join(shard_dir, f"{r.benchmark_id}.yaml")
        with open(yaml_path, "w", newline="\n") as fh:
            fh.write(build_yaml(r.target_sequence, docked_smiles, msa_path, affinity))

        rows.append({
            "benchmark_id": r.benchmark_id,
            "uniprot_id": r.uniprot_id,
            "shard": shard,
            "yaml_path": yaml_path,
            "ligand_smiles": r.ligand_smiles,
            "docked_smiles": docked_smiles,
            "was_desalted": bool(stripped),
            "smiles_parsed": n_heavy >= 0,
            "docked_heavy_atoms": n_heavy,
            "affinity_requested": bool(affinity),
            "seq_len": len(r.target_sequence),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)

    shards = (manifest.groupby("shard")
                      .agg(n_pairs=("benchmark_id", "size"),
                           affinity=("affinity_requested", "all"),
                           mixed=("affinity_requested", "nunique"))
                      .reset_index())
    shards["mixed"] = shards.mixed > 1
    shards.to_csv(SHARDS_CSV, index=False)
    if shards.mixed.any():  # never expected; the split above prevents it
        raise SystemExit(f"BUG: shards {shards.loc[shards.mixed, 'shard'].tolist()} "
                         "mix affinity and non-affinity records")

    log(f"wrote {len(manifest)} YAMLs under {YAML_DIR}")
    log(f"  desalted ligands            : {n_desalted}")
    log(f"  no affinity head (>{MAX_AFFINITY_HEAVY_ATOMS} heavy): {len(without_affinity)}")
    if n_unparsed:
        log(f"  WARNING: {n_unparsed} SMILES RDKit could not parse (passed through verbatim)")
    log(f"  shards: {args.num_shards} with affinity ({len(with_affinity)} pairs) "
        f"+ {n_extra} without ({len(without_affinity)} pairs) = {n_shards_total}")
    log(f"wrote {MANIFEST} and {SHARDS_CSV}")
    if msa_dir:
        log(f"YAMLs reference MSAs at {msa_dir}/<uniprot>.a3m - run 02_fetch_msas.py next")
    else:
        log("YAMLs carry no msa field - boltz must be run with --use_msa_server")
    log(f"then: sbatch --array=0-{n_shards_total - 1} sbatch/run_boltz.sbatch")


if __name__ == "__main__":
    main()
