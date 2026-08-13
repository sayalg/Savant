"""Shared paths and helpers for the benchmark_v2 build."""
from __future__ import annotations

import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))          # ML_model/benchmark_v2
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))    # repo root

WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "output")
CACHE = os.path.join(ROOT, "cache")
REFERENCE = os.path.join(ROOT, "reference")

# --- inputs ---------------------------------------------------------------
# Only ONE file outside this directory is required: the training table, for the
# ligand-overlap gate. The two training-protein references are vendored into
# reference/ so the build survives deletion of ML_model/taxonomy/ (whose
# protein_taxonomy.csv cost ~4 h of RCSB + UniProt calls to produce).
BINDINGDB_TSV = os.path.join(WORK, "bindingdb_kd.tsv")
TRAIN_CSV = os.path.join(REPO, "ML_model", "combined_50k_filtered.csv")
TRAIN_PROTEINS = os.path.join(REFERENCE, "train_proteins.csv")
TRAIN_PROT_TAX = os.path.join(REFERENCE, "train_protein_families.csv")

# --- intermediates ---------------------------------------------------------
CANDIDATES = os.path.join(WORK, "candidates.csv")
PROT_ANNOT = os.path.join(WORK, "protein_annotation.csv")
LIG_ANNOT = os.path.join(WORK, "ligand_annotation.csv")

# --- thresholds ------------------------------------------------------------
# Ligand disjointness is a HARD gate: nothing this similar to a training
# ligand enters the benchmark at any tier.
MAX_LIGAND_TANIMOTO = 0.50
MORGAN_RADIUS, MORGAN_BITS = 2, 2048

# Protein novelty tiers, by best identity to any training chain over >= COVERAGE
# of both sequences. These label the pairs rather than gate them, so the
# benchmark can separate "new fold" failures from "new ligand" failures.
COVERAGE = 0.80
TIER_UNSEEN_MAX = 0.30      # < 0.30 identity  -> unseen_fold
TIER_FAMILY_MAX = 0.90      # 0.30-0.90        -> same_family
                            # >= 0.90          -> same_protein

# BindingDB cleaning
CENSOR_CAPS_NM = [10000.0, 100000.0, 1000000.0, 10000000.0]
MIN_CHAIN_LEN = 20
MAX_CHAIN_LEN = 2000        # AF3/HADDOCK become impractical past this

# Drug-like small molecules essentially never bind tighter than ~1 pM
# non-covalently; entries above this are unit errors in the source record
# (e.g. a 268 Da catecholamine logged at 10 fM against a beta-adrenergic
# receptor). One mislabelled point 8 log units out badly distorts RMSE and r,
# so they are dropped rather than shipped.
MAX_PKD = 12.0

N_PAIRS = 1000
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    sys.stdout.flush()


def ensure_dirs() -> None:
    for d in (WORK, OUT, CACHE):
        os.makedirs(d, exist_ok=True)


def sanitize_seq(s: str) -> str:
    """Uppercase amino-acid letters only (BindingDB targets carry stray chars)."""
    return "".join(c for c in str(s).upper() if c.isalpha())


# pKd = -log10(Kd in M). Kd arrives in nM, so the offset is 9.
#
# Verified against Input/input.csv, which carries BOTH affinity_Kd_nM and
# neg_log10_affinity_M: 199,980 / 200,000 sampled training rows satisfy
# 9 - log10(Kd_nM) and 0 satisfy 6 - log10(Kd_nM). The old benchmark_builder
# used 6, which is why its labels sat 3 log units below the training target.
# Set to 6 only if you deliberately want that older (incorrect) convention.
PKD_OFFSET = 9


def pkd_from_kd_nm(kd_nm):
    """pKd from Kd in nM, using PKD_OFFSET (9 = -log10 of molar Kd)."""
    import numpy as np
    return PKD_OFFSET - np.log10(kd_nm)
