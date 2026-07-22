"""HADDOCK3 scoring of the docked protein-ligand complex.

Savant writes a minimal HADDOCK3 config (topology generation + a short energy
minimisation / refinement) and reads the resulting CAPRI single-structure
table for the energetic terms. It shells out to the ``haddock3`` executable,
which must be on PATH (typically loaded from an HPC module or its own venv).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

HADDOCK_COLUMNS = ["score", "total", "vdw", "elec", "desolv", "bsa"]

_CFG_TEMPLATE = """\
run_dir = "{run_dir}"
mode = "local"
ncores = {ncores}

molecules = [
    "{protein_pdb}",
    "{ligand_pdb}"
]

[topoaa]
autotoppar = true

[emref]
nemsteps = {nemsteps}
"""


def haddock3_available() -> bool:
    """True if the ``haddock3`` executable is on PATH."""
    return shutil.which("haddock3") is not None


def run_haddock3(
    protein_pdb: Path,
    ligand_pdb: Path,
    workdir: Path,
    ncores: int | None = None,
    nemsteps: int = 50,
) -> Dict[str, float]:
    """Run HADDOCK3 and return the {score, total, vdw, elec, desolv, bsa} dict."""
    if not haddock3_available():
        raise RuntimeError(
            "haddock3 executable not found on PATH. Install HADDOCK3 (e.g. "
            "`pip install haddock3`) or load your cluster's HADDOCK3 module."
        )

    run_dir = workdir / "haddock_run"
    # HADDOCK3 refuses to start if run_dir already exists.
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cfg = _CFG_TEMPLATE.format(
        run_dir=run_dir.as_posix(),
        ncores=ncores or os.cpu_count() or 4,
        protein_pdb=Path(protein_pdb).resolve().as_posix(),
        ligand_pdb=Path(ligand_pdb).resolve().as_posix(),
        nemsteps=nemsteps,
    )
    cfg_file = workdir / "score.cfg"
    cfg_file.write_text(cfg)
    logger.info("Running HADDOCK3 (this can take a while)...")

    proc = subprocess.run(
        ["haddock3", str(cfg_file)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("HADDOCK3 stdout:\n%s", proc.stdout)
        logger.error("HADDOCK3 stderr:\n%s", proc.stderr)
        raise RuntimeError(f"HADDOCK3 exited with code {proc.returncode}")

    capri_ss = _find_capri_ss(run_dir)
    table = pd.read_csv(capri_ss, sep="\t")
    row = table[HADDOCK_COLUMNS].iloc[0]
    return {col: float(row[col]) for col in HADDOCK_COLUMNS}


def _find_capri_ss(run_dir: Path) -> Path:
    """Locate the emref CAPRI single-structure table in a HADDOCK3 run dir."""
    preferred = sorted(run_dir.glob("analysis/*emref*/capri_ss.tsv"))
    if preferred:
        return preferred[-1]
    # Fall back to any capri_ss.tsv (highest-numbered module last).
    any_ss = sorted(run_dir.rglob("capri_ss.tsv"))
    if any_ss:
        return any_ss[-1]
    raise FileNotFoundError(
        f"No capri_ss.tsv found under {run_dir}. HADDOCK3 may have failed to "
        "produce analysis output; inspect the run directory."
    )
