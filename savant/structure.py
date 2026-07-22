"""Structure I/O: normalise the input to PDB and split it into protein/ligand chains."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.MMCIFParser import MMCIFParser

logger = logging.getLogger(__name__)

_PDB_SUFFIXES = {".pdb", ".ent"}
_CIF_SUFFIXES = {".cif", ".mmcif"}


def ensure_pdb(input_path: str | Path, workdir: Path) -> Path:
    """Return a path to a PDB version of ``input_path``, converting mmCIF if needed.

    PDB inputs are used as-is. mmCIF/CIF inputs are parsed and re-written as PDB
    into ``workdir``. Any other extension raises ``ValueError``.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix in _PDB_SUFFIXES:
        logger.info("Input is already a PDB: %s", path)
        return path

    if suffix in _CIF_SUFFIXES:
        logger.info("Converting mmCIF to PDB: %s", path)
        structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
        out_path = workdir / f"{path.stem}.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(out_path))
        logger.info("Wrote converted PDB: %s", out_path)
        return out_path

    raise ValueError(
        f"Unsupported input format '{suffix}'. "
        "Provide a .pdb/.ent or .cif/.mmcif docked complex."
    )


class _ChainSelect(Select):
    """Biopython selector that keeps a single chain id."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):  # noqa: D401 - Biopython hook
        return 1 if chain.get_id() == self.chain_id else 0


def _available_chains(pdb_path: Path) -> list[str]:
    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = next(iter(structure))
    return [chain.get_id() for chain in model]


def split_chains(
    pdb_path: Path,
    protein_chain: str,
    ligand_chain: str,
    workdir: Path,
) -> Tuple[Path, Path]:
    """Write the protein chain and ligand chain to separate PDB files.

    Returns ``(protein_pdb, ligand_pdb)``.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    model = next(iter(structure))
    present = {chain.get_id() for chain in model}

    for label, chain_id in (("protein", protein_chain), ("ligand", ligand_chain)):
        if chain_id not in present:
            raise ValueError(
                f"{label} chain '{chain_id}' not found in {pdb_path.name}. "
                f"Available chains: {sorted(present)}"
            )
    if protein_chain == ligand_chain:
        raise ValueError("protein-chain and ligand-chain must be different.")

    io = PDBIO()
    io.set_structure(structure)

    protein_pdb = workdir / "protein.pdb"
    ligand_pdb = workdir / "ligand.pdb"

    io.save(str(protein_pdb), _ChainSelect(protein_chain))
    io.save(str(ligand_pdb), _ChainSelect(ligand_chain))
    logger.info("Protein chain '%s' -> %s", protein_chain, protein_pdb)
    logger.info("Ligand chain '%s' -> %s", ligand_chain, ligand_pdb)

    return protein_pdb, ligand_pdb
