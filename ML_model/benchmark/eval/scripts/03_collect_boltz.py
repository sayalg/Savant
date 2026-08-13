#!/usr/bin/env python
"""
Step 03 - Harvest Boltz-2 output, enforce the chain contract, emit clean PDBs.

For every pair this:

  1. finds <id>_model_0.pdb in the boltz output tree,
  2. VERIFIES the protein is chain A (ATOM records) and the ligand is chain B
     (HETATM, residue LIG, exactly one residue) - the contract Savant
     (`-p A -l B`) and PRODIGY-LIG (`-c A B:LIG`) both depend on,
  3. writes a canonical copy to work/structures/<id>.pdb with CONECT records for
     the ligand, and
  4. records Boltz's own confidence and affinity predictions.

Why re-check (2) when step 01 asked for it: a silently mis-chained model would
not crash the downstream tools, it would quietly score the wrong thing - Savant
would hand HADDOCK a protein with no ligand, and PRODIGY would abort on a
missing residue. Cheap to verify, expensive to discover later.

Why add CONECT (3): Boltz writes ligand coordinates with no bond records at all.
HADDOCK3's `autotoppar` has to perceive ligand topology, and giving it explicit
connectivity (from RDKit, on the same coordinates) is more reliable than letting
every tool in the chain re-guess it from interatomic distances.

    python 03_collect_boltz.py
    python 03_collect_boltz.py --strict     # exit non-zero if anything failed
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
from rdkit import Chem, RDLogger

from common import (BOLTZ_DIR, BOLTZ_SCORES, LIGAND_CHAIN, LIGAND_RESNAME, MANIFEST,
                    PROTEIN_CHAIN, STRUCT_DIR, WORK, ensure_dirs, log)

RDLogger.DisableLog("rdApp.*")

CONFIDENCE_FIELDS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
]
AFFINITY_FIELDS = [
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_probability_binary1",
    "affinity_pred_value2", "affinity_probability_binary2",
]


def find_prediction(boltz_dir: str, bid: str) -> str | None:
    """Locate <id>_model_0.pdb regardless of how boltz nested the output.

    `boltz predict` writes <out_dir>/boltz_results_<input>/predictions/<id>/,
    but that layout has changed between releases, so search rather than assume.
    """
    hits = glob.glob(
        os.path.join(boltz_dir, "**", "predictions", bid, f"{bid}_model_0.pdb"),
        recursive=True,
    )
    if not hits:
        hits = glob.glob(os.path.join(boltz_dir, "**", f"{bid}_model_0.pdb"), recursive=True)
    return sorted(hits)[0] if hits else None


def _atom_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(("ATOM  ", "HETATM"))]


def _chain_of(line: str) -> str:
    return line[21]


def _residue_key(line: str) -> tuple[str, str, str]:
    return (line[17:20].strip(), line[21], line[22:27].strip())


def remap_chains(lines: list[str]) -> tuple[list[str], bool]:
    """Force ATOM chains to A and HETATM chains to B, when that is unambiguous.

    Only fires when the file holds exactly one all-ATOM chain and one all-HETATM
    chain but the ids are not A/B - i.e. the model is right and only the labels
    are off. Anything more tangled is reported as a failure instead.
    """
    atom_chains = {_chain_of(ln) for ln in lines if ln.startswith("ATOM  ")}
    het_chains = {_chain_of(ln) for ln in lines if ln.startswith("HETATM")}
    if len(atom_chains) != 1 or len(het_chains) != 1 or atom_chains & het_chains:
        return lines, False
    if atom_chains == {PROTEIN_CHAIN} and het_chains == {LIGAND_CHAIN}:
        return lines, False
    out = []
    for ln in lines:
        new = PROTEIN_CHAIN if ln.startswith("ATOM  ") else LIGAND_CHAIN
        out.append(ln[:21] + new + ln[22:])
    return out, True


def check_contract(lines: list[str], expect_residues: int, expect_heavy: int) -> list[str]:
    """Return a list of contract violations (empty means the structure is usable)."""
    problems = []
    protein = [ln for ln in lines if _chain_of(ln) == PROTEIN_CHAIN]
    ligand = [ln for ln in lines if _chain_of(ln) == LIGAND_CHAIN]
    other = {_chain_of(ln) for ln in lines} - {PROTEIN_CHAIN, LIGAND_CHAIN}

    if not protein:
        problems.append(f"no chain {PROTEIN_CHAIN} (protein)")
    if not ligand:
        problems.append(f"no chain {LIGAND_CHAIN} (ligand)")
    if other:
        problems.append(f"unexpected chains {sorted(other)}")
    if not protein or not ligand:
        return problems

    if any(ln.startswith("HETATM") for ln in protein):
        problems.append(f"chain {PROTEIN_CHAIN} contains HETATM records")
    if any(ln.startswith("ATOM  ") for ln in ligand):
        problems.append(f"chain {LIGAND_CHAIN} contains ATOM records (expected HETATM)")

    lig_residues = {_residue_key(ln) for ln in ligand}
    if len(lig_residues) != 1:
        problems.append(f"chain {LIGAND_CHAIN} has {len(lig_residues)} residues, expected 1")
    lig_names = {ln[17:20].strip() for ln in ligand}
    if lig_names != {LIGAND_RESNAME}:
        # PRODIGY-LIG selects the ligand by residue name and needs exactly 3 chars.
        problems.append(f"ligand residue name(s) {sorted(lig_names)}, expected {LIGAND_RESNAME}")

    n_res = len({_residue_key(ln) for ln in protein})
    if expect_residues and n_res != expect_residues:
        problems.append(f"protein has {n_res} residues, expected {expect_residues}")
    if expect_heavy > 0 and len(ligand) != expect_heavy:
        problems.append(f"ligand has {len(ligand)} atoms, expected {expect_heavy} heavy atoms")
    return problems


def ligand_conect(ligand_lines: list[str]) -> list[str]:
    """CONECT records for the ligand, with bonds perceived by RDKit.

    Returns [] if perception fails - the structure is still usable, HADDOCK just
    falls back to guessing connectivity itself.
    """
    block = "\n".join(ligand_lines) + "\nEND\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumAtoms() != len(ligand_lines):
        return []
    serials = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            return []
        serials.append(info.GetSerialNumber())

    partners: dict[int, list[int]] = {s: [] for s in serials}
    for bond in mol.GetBonds():
        a, b = serials[bond.GetBeginAtomIdx()], serials[bond.GetEndAtomIdx()]
        partners[a].append(b)
        partners[b].append(a)

    records = []
    for serial in serials:
        neighbours = sorted(partners[serial])
        for i in range(0, len(neighbours), 4):          # 4 partners per record
            chunk = neighbours[i : i + 4]
            records.append("CONECT" + f"{serial:5d}" + "".join(f"{n:5d}" for n in chunk))
    return records


def write_canonical(path: str, lines: list[str], conect: list[str]) -> None:
    protein = [ln for ln in lines if _chain_of(ln) == PROTEIN_CHAIN]
    ligand = [ln for ln in lines if _chain_of(ln) == LIGAND_CHAIN]
    body = protein + ["TER"] + ligand + ["TER"] + conect + ["END"]
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="\n") as fh:
        fh.write("\n".join(body) + "\n")
    os.replace(tmp, path)


def read_scores(pred_dir: str, bid: str) -> dict:
    """Boltz's own predictions: confidence for the pose, affinity for the pair."""
    scores: dict = {}
    conf = os.path.join(pred_dir, f"confidence_{bid}_model_0.json")
    if os.path.exists(conf):
        with open(conf) as fh:
            data = json.load(fh)
        scores.update({f"boltz_{k}": data.get(k) for k in CONFIDENCE_FIELDS})

    aff = os.path.join(pred_dir, f"affinity_{bid}.json")
    if os.path.exists(aff):
        with open(aff) as fh:
            data = json.load(fh)
        scores.update({f"boltz_{k}": data.get(k) for k in AFFINITY_FIELDS})
        value = data.get("affinity_pred_value")
        # affinity_pred_value is log10(IC50) with IC50 in uM, so the pIC50-style
        # figure comparable to the benchmark's pKd label is 6 - value.
        if value is not None:
            scores["boltz_affinity_pic50"] = 6.0 - float(value)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--boltz-dir", default=BOLTZ_DIR)
    ap.add_argument("--struct-dir", default=STRUCT_DIR)
    ap.add_argument("--no-remap-chains", action="store_true",
                    help="do not relabel an unambiguous ATOM/HETATM chain pair to A/B")
    ap.add_argument("--no-conect", action="store_true", help="skip ligand CONECT records")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any pair failed")
    args = ap.parse_args()

    ensure_dirs(WORK, args.struct_dir)
    manifest = pd.read_csv(args.manifest)

    rows = []
    for m in manifest.itertuples(index=False):
        bid = m.benchmark_id
        row = {"benchmark_id": bid, "status": "ok", "error": "", "chains_remapped": False}

        pdb = find_prediction(args.boltz_dir, bid)
        if pdb is None:
            row.update(status="missing", error="no *_model_0.pdb in boltz output")
            rows.append(row)
            continue

        with open(pdb) as fh:
            lines = _atom_lines(fh.read())
        if not args.no_remap_chains:
            lines, remapped = remap_chains(lines)
            row["chains_remapped"] = remapped
            if remapped:
                log(f"{bid}: relabelled chains to {PROTEIN_CHAIN}/{LIGAND_CHAIN}")

        problems = check_contract(lines, int(m.seq_len), int(m.docked_heavy_atoms))
        row["n_protein_residues"] = len(
            {_residue_key(ln) for ln in lines if _chain_of(ln) == PROTEIN_CHAIN}
        )
        row["n_ligand_atoms"] = sum(1 for ln in lines if _chain_of(ln) == LIGAND_CHAIN)
        if problems:
            row.update(status="invalid", error="; ".join(problems))
            log(f"{bid}: INVALID - {row['error']}")
            rows.append(row)
            continue

        ligand = [ln for ln in lines if _chain_of(ln) == LIGAND_CHAIN]
        conect = [] if args.no_conect else ligand_conect(ligand)
        row["n_conect_records"] = len(conect)
        out_pdb = os.path.join(args.struct_dir, f"{bid}.pdb")
        write_canonical(out_pdb, lines, conect)
        row["structure_path"] = os.path.abspath(out_pdb)
        row.update(read_scores(os.path.dirname(pdb), bid))
        rows.append(row)

    scores = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(BOLTZ_SCORES), exist_ok=True)
    scores.to_csv(BOLTZ_SCORES, index=False)

    counts = scores.status.value_counts().to_dict()
    log(f"{len(scores)} pairs: {counts}")
    log(f"structures -> {args.struct_dir}")
    log(f"scores     -> {BOLTZ_SCORES}")
    if "boltz_affinity_pred_value" in scores:
        n_aff = scores.boltz_affinity_pred_value.notna().sum()
        log(f"affinity predicted for {n_aff} pairs")
    if scores.chains_remapped.any():
        log(f"NOTE: {int(scores.chains_remapped.sum())} structures needed chain relabelling")
    failed = scores[scores.status != "ok"]
    if len(failed):
        log(f"{len(failed)} pairs unusable - see status/error in {BOLTZ_SCORES}")
        log("re-run the corresponding Boltz shards, then re-run this step")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
