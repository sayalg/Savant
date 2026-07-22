"""Savant: score an already-docked protein-ligand complex into a feature table."""

__version__ = "0.1.0"

# The exact column order of the table Savant produces.
FEATURE_COLUMNS = [
    "molecular_weight",
    "logP",
    "num_h_donors",
    "num_h_acceptors",
    "num_rotatable_bonds",
    "tpsa",
    "num_rings",
    "num_aromatic_rings",
    "num_atoms",
    "num_heavy_atoms",
    "formal_charge",
    "score",
    "total",
    "vdw",
    "elec",
    "desolv",
    "bsa",
    "num_interacting_residues",
]
