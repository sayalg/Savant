"""Savant v2 command line interface.

    savant2 complex.pdb -s "CC(=O)Oc1ccccc1C(=O)O"

Takes any docked protein-ligand complex - an AlphaFold Server prediction, a
Boltz output, a Vina pose, a crystal structure - and returns a Kd with a
calibrated interval plus the interaction profile behind it, on one CPU core in
about a second.

argparse rather than click: the whole point of v2 is that a biologist can
install and run it without a cluster, so the dependency list stays at
numpy/scipy/RDKit/scikit-learn and everything else comes from the standard
library.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from savant2 import LIGAND_CHAIN, PROTEIN_CHAIN, __version__
from savant2.models import SavantBundle
from savant2.pipeline import score_complex

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def _render(name: str, result: dict) -> str:
    lines = [f"\n  {name}   ({result['meta']['seconds']}s)\n"]
    lines.append(f"  prediction   {result['classification'].upper()}"
                 f"   (P(binder) = {result['p_binder']:.2f})")
    lines.append(
        f"  affinity     pKd {result['pK']:.2f}  "
        f"[{result['pK_lower']:.2f} - {result['pK_upper']:.2f}]   "
        f"confidence: {result['confidence']}"
    )
    lines.append(
        f"               Kd  {result['Kd_nM']:,.1f} nM  "
        f"[{result['Kd_nM_lower']:,.1f} - {result['Kd_nM_upper']:,.1f}]"
    )
    lines.append("")
    lines.append(f"  interface    {result['n_interacting_residues']} residues, "
                 f"{result['buried_surface_A2']:,.0f} A^2 buried, "
                 f"ligand {result['ligand_buried_fraction'] * 100:.0f}% enclosed")
    if result["interaction_profile"]:
        lines.append("")
        lines.append("  interactions")
        for label, count in result["interaction_profile"].items():
            lines.append(f"    {label:32} {count:6.0f}")
    if result["contacting_residue_types"]:
        top = list(result["contacting_residue_types"].items())[:8]
        lines.append("")
        lines.append("  contacting residues  " + ", ".join(f"{k} x{v}" for k, v in top))
    if "note" in result:
        lines.append("")
        lines.append(f"  note: {result['note']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="savant2",
        description="Predict binding affinity for a docked protein-ligand complex.",
    )
    p.add_argument("structure", help="docked complex (.pdb)")
    p.add_argument("-s", "--ligand-smiles", required=True,
                   help="ligand SMILES; required because a docked PDB carries no bond "
                        "orders, so aromaticity, charge and H-bond roles come from here")
    p.add_argument("-p", "--protein-chain", default=PROTEIN_CHAIN)
    p.add_argument("-l", "--ligand-chain", default=LIGAND_CHAIN)
    p.add_argument("-m", "--model-dir", default=str(DEFAULT_MODEL_DIR),
                   help="directory holding classifier.pkl and regressor.pkl")
    p.add_argument("-o", "--output", default=None, help="write the full result as JSON")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print JSON instead of the report")
    p.add_argument("-V", "--version", action="version", version=f"savant2 {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model_dir = Path(args.model_dir)
    missing = [n for n in ("classifier.pkl", "regressor.pkl")
               if not (model_dir / n).exists()]
    if missing:
        raise SystemExit(f"{model_dir} is missing {', '.join(missing)}. "
                         "Train with scripts/02_train.py, or pass --model-dir.")
    bundle = SavantBundle.load(model_dir)

    try:
        result = score_complex(args.structure, args.ligand_smiles, bundle,
                               args.protein_chain, args.ligand_chain)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n\nSavant needs the protein and ligand on separate chains "
            f"(default -p {PROTEIN_CHAIN} -l {LIGAND_CHAIN}). Check the chain ids in "
            "your file and pass them explicitly if they differ."
        )

    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        print(_render(Path(args.structure).name, result))
        print()
        if not result["meta"]["ligand_typed"]:
            print("  warning: the SMILES did not match the 3D ligand, so charge and "
                  "aromaticity features are unset. Check the ligand chain.\n")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
