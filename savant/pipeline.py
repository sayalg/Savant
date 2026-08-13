"""Orchestrate the Savant scoring + prediction pipeline for one docked complex."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from savant import FEATURE_COLUMNS
from savant.descriptors import ligand_descriptors
from savant.haddock import run_haddock3
from savant.identity import extract_protein_sequence, resolve_ligand_smiles
from savant.interactions import count_interacting_residues
from savant.models import (
    BINDER_LABEL,
    DEFAULT_MODEL_DIR,
    classify,
    predict_pk,
    resolve_model_paths,
)
from savant.structure import ensure_pdb, split_chains

logger = logging.getLogger(__name__)


def run_pipeline(
    structure: str | Path,
    protein_chain: str,
    ligand_chain: str,
    ligand_smiles: Optional[str] = None,
    workdir: Optional[str | Path] = None,
    cutoff: float = 4.5,
    ncores: Optional[int] = None,
    nemsteps: int = 50,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    classifier_model: Optional[str | Path] = None,
    regressor_model: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the full pipeline and return the JSON-serialisable result dict.

    Steps:
      1. Normalise input to PDB (convert mmCIF if needed).
      2. Split into protein and ligand chains.
      3. Derive protein sequence and ligand SMILES.
      4. Build the 18 model features: RDKit descriptors + HADDOCK3 energetics +
         interacting-residue count.
      5. Classify binder / nonbinder; if binder, predict pK.
    """
    # Resolve model paths up front so a bad path fails fast, before HADDOCK3.
    classifier_path, regressor_path = resolve_model_paths(
        model_dir, classifier_model, regressor_model
    )

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="savant_"))
    work.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work)

    pdb_path = ensure_pdb(structure, work)
    protein_pdb, ligand_pdb = split_chains(pdb_path, protein_chain, ligand_chain, work)

    protein_sequence = extract_protein_sequence(protein_pdb)
    smiles, smiles_source = resolve_ligand_smiles(ligand_pdb, ligand_smiles)

    # --- Build the 18 model features (kept in memory, no CSV) ---
    descriptors = ligand_descriptors(ligand_pdb, smiles=ligand_smiles)
    docking = run_haddock3(
        protein_pdb, ligand_pdb, work, ncores=ncores, nemsteps=nemsteps
    )
    num_interacting = count_interacting_residues(
        pdb_path, protein_chain, ligand_chain, cutoff=cutoff
    )
    features = {
        **descriptors,
        **docking,
        "num_interacting_residues": num_interacting,
    }
    features = {col: features[col] for col in FEATURE_COLUMNS}  # canonical order

    # --- Model inference ---
    label, probabilities = classify(features, classifier_path)
    is_binder = label == BINDER_LABEL
    pk = predict_pk(features, regressor_path) if is_binder else None
    if not is_binder:
        logger.info("Classified as '%s' — skipping pK regression.", label)

    return {
        "protein_sequence": protein_sequence,
        "ligand_smiles": smiles,
        "ligand_smiles_source": smiles_source,
        "features": features,
        "classification": label,
        "classification_probabilities": probabilities,
        "pK": pk,
    }
