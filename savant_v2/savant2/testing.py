"""A tiny synthetic complex, used to fix the canonical feature order.

``features.feature_names()`` needs some complex to extract from, and depending
on a real structure file would make the feature order depend on data that may
not be present. This builds a minimal but valid complex in a temp file: three
residues chosen to exercise the aromatic (PHE), anionic (ASP) and cationic (LYS)
paths, with benzoic acid placed in contact range.

The values are meaningless. Only the set and order of feature keys matters, and
those are identical for any valid complex because every feature is emitted
unconditionally, zero-valued when the interaction is absent.
"""
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

LIGAND_SMILES = "OC(=O)c1ccccc1"

# PHE ring, ASP carboxylate, LYS amine, then a benzoic acid ~3.5 A away.
_PDB = """\
ATOM      1  N   PHE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  PHE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  CB  PHE A   1       2.000   1.420   0.000  1.00  0.00           C
ATOM      4  CG  PHE A   1       3.500   1.500   0.000  1.00  0.00           C
ATOM      5  CD1 PHE A   1       4.200   2.700   0.000  1.00  0.00           C
ATOM      6  CD2 PHE A   1       4.200   0.300   0.000  1.00  0.00           C
ATOM      7  CE1 PHE A   1       5.590   2.700   0.000  1.00  0.00           C
ATOM      8  CE2 PHE A   1       5.590   0.300   0.000  1.00  0.00           C
ATOM      9  CZ  PHE A   1       6.290   1.500   0.000  1.00  0.00           C
ATOM     10  N   ASP A   2       1.900  -1.200   3.000  1.00  0.00           N
ATOM     11  CA  ASP A   2       3.300  -1.400   3.200  1.00  0.00           C
ATOM     12  CB  ASP A   2       3.800  -0.900   4.560  1.00  0.00           C
ATOM     13  CG  ASP A   2       5.300  -1.000   4.700  1.00  0.00           C
ATOM     14  OD1 ASP A   2       5.900  -1.900   4.100  1.00  0.00           O
ATOM     15  OD2 ASP A   2       5.900  -0.200   5.450  1.00  0.00           O
ATOM     16  N   LYS A   3      -1.500   2.000   4.000  1.00  0.00           N
ATOM     17  CA  LYS A   3      -0.800   2.900   4.900  1.00  0.00           C
ATOM     18  CB  LYS A   3      -1.700   3.500   5.980  1.00  0.00           C
ATOM     19  CG  LYS A   3      -1.000   4.400   6.990  1.00  0.00           C
ATOM     20  CD  LYS A   3      -1.900   5.000   8.060  1.00  0.00           C
ATOM     21  CE  LYS A   3      -1.200   5.900   9.070  1.00  0.00           C
ATOM     22  NZ  LYS A   3      -2.100   6.500  10.070  1.00  0.00           N
HETATM   23  O1  LIG B   1       4.800   1.500   3.500  1.00  0.00           O
HETATM   24  C1  LIG B   1       4.100   1.500   4.500  1.00  0.00           C
HETATM   25  O2  LIG B   1       4.600   1.500   5.650  1.00  0.00           O
HETATM   26  C2  LIG B   1       2.620   1.500   4.400  1.00  0.00           C
HETATM   27  C3  LIG B   1       1.900   0.400   3.900  1.00  0.00           C
HETATM   28  C4  LIG B   1       0.520   0.450   3.800  1.00  0.00           C
HETATM   29  C5  LIG B   1      -0.150   1.600   4.200  1.00  0.00           C
HETATM   30  C6  LIG B   1       0.560   2.700   4.700  1.00  0.00           C
HETATM   31  C7  LIG B   1       1.940   2.650   4.800  1.00  0.00           C
END
"""


@contextlib.contextmanager
def synthetic_complex() -> Iterator[Tuple[str, str]]:
    """Yield ``(pdb_path, smiles)`` for a throwaway complex."""
    with tempfile.TemporaryDirectory(prefix="savant2_") as tmp:
        path = Path(tmp) / "synthetic.pdb"
        path.write_text(_PDB)
        yield str(path), LIGAND_SMILES
