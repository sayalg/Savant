"""Savant command-line interface."""

from __future__ import annotations

import json
import logging

import click

from savant import __version__
from savant.models import DEFAULT_MODEL_DIR
from savant.pipeline import run_pipeline


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("structure", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--protein-chain",
    "-p",
    required=True,
    help="Chain ID of the protein in the docked structure (e.g. A).",
)
@click.option(
    "--ligand-chain",
    "-l",
    required=True,
    help="Chain ID of the ligand in the docked structure (e.g. B).",
)
@click.option(
    "--ligand-smiles",
    "-s",
    default=None,
    help="Ligand SMILES. Strongly recommended: gives correct RDKit "
    "descriptors. Without it, chemistry is guessed from 3D coordinates.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="savant_results.json",
    show_default=True,
    help="Path for the output result (JSON).",
)
@click.option(
    "--workdir",
    "-w",
    type=click.Path(file_okay=False),
    default=None,
    help="Directory for intermediate files (default: a temp directory).",
)
@click.option(
    "--cutoff",
    type=float,
    default=4.5,
    show_default=True,
    help="Distance cutoff (A) for counting interacting residues.",
)
@click.option(
    "--ncores",
    type=int,
    default=None,
    help="Cores for HADDOCK3 (default: all available).",
)
@click.option(
    "--nemsteps",
    type=int,
    default=50,
    show_default=True,
    help="Energy-minimization steps for the HADDOCK3 emref module.",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False),
    default=DEFAULT_MODEL_DIR,
    show_default=True,
    help="Directory holding the trained .pkl models (env: SAVANT_MODEL_DIR).",
)
@click.option(
    "--classifier-model",
    type=click.Path(dir_okay=False),
    default=None,
    help="Explicit path to the binder/nonbinder classifier .pkl "
    "(default: auto-discover tender_comb_*_local.pkl in --model-dir).",
)
@click.option(
    "--regressor-model",
    type=click.Path(dir_okay=False),
    default=None,
    help="Explicit path to the pK regressor .pkl "
    "(default: auto-discover mango_sponge_*_local.pkl in --model-dir).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(__version__, "-V", "--version")
def main(
    structure,
    protein_chain,
    ligand_chain,
    ligand_smiles,
    output,
    workdir,
    cutoff,
    ncores,
    nemsteps,
    model_dir,
    classifier_model,
    regressor_model,
    verbose,
):
    """Predict binding for a docked protein-ligand complex.

    STRUCTURE is a docked protein-ligand complex (.pdb/.ent or .cif/.mmcif).
    It is converted to PDB if needed, split by chain, then processed with
    RDKit (ligand descriptors) and HADDOCK3 (docking energetics). Those 18
    features feed the binder/nonbinder classifier; binders are then scored for
    pK. The result is written as JSON.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        result = run_pipeline(
            structure=structure,
            protein_chain=protein_chain,
            ligand_chain=ligand_chain,
            ligand_smiles=ligand_smiles,
            workdir=workdir,
            cutoff=cutoff,
            ncores=ncores,
            nemsteps=nemsteps,
            model_dir=model_dir,
            classifier_model=classifier_model,
            regressor_model=regressor_model,
        )
    except Exception as exc:  # surface a clean message, full trace in -v
        logging.getLogger(__name__).error("Savant failed: %s", exc, exc_info=verbose)
        raise SystemExit(1)

    payload = json.dumps(result, indent=2)
    with open(output, "w") as fh:
        fh.write(payload + "\n")
    click.echo(payload)
    click.echo(f"\nSaved result to {output}", err=True)


if __name__ == "__main__":
    main()
