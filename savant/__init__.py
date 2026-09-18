"""Savant - structure-free protein-ligand binding affinity prediction.

Predicts pKd for a protein-ligand pair from an amino-acid sequence and a SMILES
string: no experimental structure, no predicted structure, no docking, no GPU.

    from savant import Savant

    sv = Savant()                                   # esm650, the model of record
    sv.screen("MTVPDRSEI...", ["CCO", "c1ccccc1"])  # ranked best-first

    sv.predict(["MTVPDRSEI..."], ["CCO"])           # arbitrary parallel pairs

Or from the command line, which writes the JSON document described in
:mod:`savant.report`:

    savant --seq target.fasta --smiles library.smi --out screen.json

Scores rank ligands against one target. They do not rank targets against one
another, and they are not calibrated absolute affinities -- see the module
docstrings and the repository README for what that rules in and out.
"""
from __future__ import annotations

__version__ = "1.0.0"

from savant.inputs import (Ligand, Target, dedupe_ligands, read_ligands,
                           read_pdb, read_targets)
from savant.model import (BINDER_PKD, DEFAULT_MODEL, MODELS, Savant, load_bundle,
                          pkd_to_kd_nM, resolve_model_dir)
from savant.report import build_report, build_table, write_report, write_table

__all__ = [
    "__version__",
    # scoring
    "Savant", "MODELS", "DEFAULT_MODEL", "BINDER_PKD",
    "load_bundle", "resolve_model_dir", "pkd_to_kd_nM",
    # input
    "Target", "Ligand", "read_targets", "read_ligands", "dedupe_ligands", "read_pdb",
    # output
    "build_report", "write_report", "build_table", "write_table",
]
