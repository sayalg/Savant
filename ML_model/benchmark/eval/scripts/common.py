"""Shared paths, constants and helpers for the three-tool benchmark evaluation.

Boltz-2 predicts the complex; Savant and PRODIGY-LIG then score the structure it
produced. Everything downstream of Boltz assumes the same chain contract, which
is defined once here and enforced in 03_collect_boltz.py.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.abspath(os.path.join(HERE, ".."))          # ML_model/benchmark/eval
BENCH = os.path.abspath(os.path.join(EVAL, ".."))         # ML_model/benchmark
REPO = os.path.abspath(os.path.join(BENCH, "..", ".."))   # repo root

# --- input ----------------------------------------------------------------
BENCHMARK_CSV = os.path.join(BENCH, "output", "savant_benchmark_1000.csv")
MODEL_DIR = os.path.join(REPO, "ML_model", "train")       # savant .pkl models

# --- generated ------------------------------------------------------------
WORK = os.path.join(EVAL, "work")
OUT = os.path.join(EVAL, "output")

YAML_DIR = os.path.join(WORK, "yaml")            # yaml/shard_00/<id>.yaml
MSA_DIR = os.path.join(WORK, "msa")              # <uniprot>.a3m
BOLTZ_DIR = os.path.join(WORK, "boltz")          # raw boltz output trees
STRUCT_DIR = os.path.join(WORK, "structures")    # canonical <id>.pdb
MANIFEST = os.path.join(WORK, "manifest.csv")
TARGETS_CSV = os.path.join(WORK, "targets.csv")
SHARDS_CSV = os.path.join(WORK, "shards.csv")
BOLTZ_SCORES = os.path.join(WORK, "boltz_scores.csv")

RESULTS = os.path.join(WORK, "results")
SAVANT_RESULTS = os.path.join(RESULTS, "savant")      # <id>.json
PRODIGY_RESULTS = os.path.join(RESULTS, "prodigy")    # <id>.json

SCORED_CSV = os.path.join(OUT, "savant_benchmark_1000_scored.csv")
SCORED_SUMMARY = os.path.join(OUT, "savant_benchmark_1000_scored_summary.md")

# --- the chain contract ---------------------------------------------------
# Savant needs `-p A -l B`; PRODIGY-LIG needs `-c A B:LIG`. Both therefore
# require the protein and the ligand to sit on SEPARATE chains, and PRODIGY
# additionally requires a 3-character ligand residue name that occurs exactly
# once in its chain. Boltz honours the ids we give it in the YAML and hardcodes
# the residue name "LIG" for every non-polymer chain in PDB output, so asking
# for protein id A / ligand id B satisfies all of it.
PROTEIN_CHAIN = "A"
LIGAND_CHAIN = "B"
LIGAND_RESNAME = "LIG"

# Boltz-2's affinity head is only defined for a single ligand chain of at most
# this many heavy atoms; larger ligands get a structure but no affinity value.
MAX_AFFINITY_HEAVY_ATOMS = 128

# deltaG (kcal/mol) <-> pKd at 298 K, the same convention the benchmark's
# deltaG_kcal_per_mol column uses (deltaG = -1.364 * pKd).
RT_LN10_KCAL = 1.364


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    sys.stdout.flush()


def ensure_dirs(*dirs: str) -> None:
    for d in dirs or (WORK, OUT):
        os.makedirs(d, exist_ok=True)


def shard_name(i: int) -> str:
    return f"shard_{i:02d}"


def take_shard(items: list, shard_id: int, num_shards: int) -> list:
    """Round-robin slice of ``items`` for this array task.

    Round-robin rather than contiguous blocks: the benchmark is sorted by
    target, so contiguous blocks would put all the large proteins (or all the
    HADDOCK-expensive complexes) in the same task.
    """
    if num_shards <= 1:
        return list(items)
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} out of range for {num_shards} shards")
    return list(items)[shard_id::num_shards]


def slurm_shard(num_shards: int) -> int:
    """Shard index for this process, from SLURM_ARRAY_TASK_ID (0-based)."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return 0
    idx = int(task_id)
    if idx >= num_shards:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={idx} but only {num_shards} shards exist; "
            "use a 0-based --array matching --num-shards."
        )
    return idx


def write_json(path: str, payload: dict) -> None:
    """Write JSON atomically so an interrupted job never leaves a half file
    that a resumed run would mistake for a completed one."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def read_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def yaml_quote(value: str) -> str:
    """Single-quoted YAML scalar.

    Inside single quotes YAML takes every character literally except `'`, which
    is escaped by doubling. That matters here: SMILES are full of backslashes
    (`C/C=C\\C`) that a double-quoted scalar would eat.
    """
    return "'" + str(value).replace("'", "''") + "'"
