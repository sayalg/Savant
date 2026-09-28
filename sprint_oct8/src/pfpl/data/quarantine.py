"""Benchmark quarantine (PLAN.md Section 5.3.1) - run BEFORE anything else.

Every measurement whose protein-ligand pair appears in the sealed benchmark is
removed from all training data. The "strict" variant additionally removes proteins
at >=30% sequence identity to any benchmark protein and ligands at Tanimoto >=0.9
to any benchmark ligand, and both sizes are reported.

This matters more than it sounds. The previous project's flagship result - per
target rho +0.28 on influenza A neuraminidase - turned out to be measured on a
protein that was IN the training fold (protein_id 11771), with 55 of 81 benchmark
ligands present as training pairs. The number was real; it just was not a
prediction. Quarantine keys are produced here once, frozen, and checked by
`tests/test_leakage.py` on every run.

Identity is computed by k-mer containment rather than by alignment: it is a
conservative upper bound on identity (containment >= identity for these lengths),
it runs in seconds on 6,000 sequences, and over-removing from TRAINING is the safe
direction of error. mmseqs at 30% is the reference and is used when available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 12                    # k-mer length for containment
IDENTITY_THRESHOLD = 0.30
TANIMOTO_THRESHOLD = 0.90


def kmers(seq: str, k: int = K) -> set:
    s = str(seq).upper()
    return {s[i:i + k] for i in range(max(len(s) - k + 1, 0))}


def containment(a: set, b: set) -> float:
    """|a & b| / min(|a|, |b|) - symmetric-ish and robust to length differences."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def ecfp(smiles_list, radius: int = 2, nbits: int = 2048):
    from ..preflight import require_rdkit
    require_rdkit("Tanimoto quarantine")
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out, DataStructs


def build_keys(benchmark: pd.DataFrame) -> dict:
    """Exact keys from the benchmark: (sequence_hash, inchikey_full) pairs and sets.

    Rows whose ligand failed standardisation have no InChIKey and are dropped from
    the PAIR keys - a key of "h123|None" cannot match a real corpus row, and
    counting it would overstate how much of the benchmark is quarantined. Their
    protein still goes into the protein set, so the strict rule still covers them.
    """
    ok = benchmark.dropna(subset=["sequence_hash", "inchikey_full"])
    if len(ok) < len(benchmark):
        print(f"  {len(benchmark) - len(ok)} benchmark row(s) have no usable ligand "
              "key and are excluded from the exact pair keys")
    return {
        "pairs": sorted({f"{a}|{b}" for a, b in
                         zip(ok.sequence_hash.astype(str),
                             ok.inchikey_full.astype(str))}),
        "proteins": sorted(set(benchmark.sequence_hash.dropna().astype(str))),
        "ligands_full": sorted(set(benchmark.inchikey_full.dropna().astype(str))),
        "ligands_flat": sorted(set(benchmark.inchikey_flat.dropna().astype(str))),
    }


def apply_exact(pairs: pd.DataFrame, keys: dict, verbose: bool = True) -> np.ndarray:
    """Boolean mask of rows to REMOVE under the exact (non-strict) rule."""
    pk = pairs.sequence_hash.astype(str) + "|" + pairs.inchikey_full.astype(str)
    drop = pk.isin(set(keys["pairs"])).to_numpy()
    if verbose:
        print(f"  exact quarantine: {int(drop.sum()):,} rows removed "
              f"({drop.mean():.3%} of the corpus)")
    return drop


def apply_strict(pairs: pd.DataFrame, benchmark: pd.DataFrame, keys: dict,
                 identity: float = IDENTITY_THRESHOLD,
                 tanimoto: float = TANIMOTO_THRESHOLD,
                 protein_clusters: np.ndarray | None = None,
                 verbose: bool = True) -> np.ndarray:
    """Exact, plus homologous proteins and near-identical ligands."""
    drop = apply_exact(pairs, keys, verbose=False)

    ## --- protein axis -------------------------------------------------------
    if protein_clusters is not None:
        ## Preferred route: mmseqs already grouped at 30% identity, so "homologous
        ## to a benchmark protein" is exactly "in a benchmark protein's cluster".
        bench_hashes = set(keys["proteins"])
        bench_rows = pairs.sequence_hash.astype(str).isin(bench_hashes).to_numpy()
        bad_clusters = set(np.unique(protein_clusters[bench_rows]).tolist())
        prot_drop = np.isin(protein_clusters, list(bad_clusters))
        route = "mmseqs clusters"
    else:
        uniq = pairs.drop_duplicates("sequence_hash")[["sequence_hash", "sequence"]]
        bench_km = [kmers(s) for s in benchmark.sequence.dropna().unique()]
        bad = set()
        for h, s in zip(uniq.sequence_hash, uniq.sequence):
            if not isinstance(s, str):
                continue
            km = kmers(s)
            if any(containment(km, bk) >= identity for bk in bench_km):
                bad.add(str(h))
        prot_drop = pairs.sequence_hash.astype(str).isin(bad).to_numpy()
        route = f"{K}-mer containment >= {identity}"

    ## --- ligand axis --------------------------------------------------------
    corpus_smiles = pairs.ligand_smiles_std.fillna("").tolist()
    bench_smiles = benchmark.ligand_smiles_std.dropna().tolist()
    cfp, DS = ecfp(corpus_smiles)
    bfp, _ = ecfp(bench_smiles)
    bfp = [f for f in bfp if f is not None]
    lig_drop = np.zeros(len(pairs), bool)
    ## Flat-InChIKey identity first: it is exact, free, and catches stereoisomers
    ## and tautomers that a fingerprint similarity would also catch but slower.
    lig_drop |= pairs.inchikey_flat.astype(str).isin(set(keys["ligands_flat"])).to_numpy()
    for i, f in enumerate(cfp):
        if f is None or lig_drop[i]:
            continue
        if bfp and max(DS.BulkTanimotoSimilarity(f, bfp)) >= tanimoto:
            lig_drop[i] = True

    ## Section 5.3.1: "proteins with >=30% identity AND ligands with Tanimoto >=0.9".
    ## Both axes together, so a held-out protein tested with a novel ligand survives.
    strict = drop | (prot_drop & lig_drop)
    if verbose:
        print(f"  strict quarantine ({route}):")
        print(f"    exact pairs          {int(drop.sum()):,}")
        print(f"    homologous protein   {int(prot_drop.sum()):,}")
        print(f"    near-identical ligand{int(lig_drop.sum()):,}")
        print(f"    removed (both axes)  {int(strict.sum()):,} ({strict.mean():.3%})")
    return strict


def save(path, keys: dict, stats: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"keys": keys, "stats": stats}, indent=1))
    print(f"  wrote {p}")


def load(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{p} missing. Quarantine keys must exist BEFORE any training run.\n"
            "    python scripts/phase0_verify.py --benchmark <file>")
    return json.loads(p.read_text())


def assert_clean_frame(pairs: pd.DataFrame, keys: dict) -> None:
    """Raise if any exact benchmark pair survives in `pairs`."""
    drop = apply_exact(pairs, keys, verbose=False)
    if drop.any():
        raise AssertionError(
            f"{int(drop.sum())} benchmark pair(s) are present in the training corpus.\n"
            "  The sealed benchmark is scored once per model version and must never\n"
            "  appear in training (PLAN.md Sections 0.4.3, 5.3.1).\n"
            "  The previous project's headline result was measured on a protein that\n"
            "  was in its own training fold, with 55 of 81 benchmark ligands present\n"
            "  as training pairs.")


def assert_clean(pairs: pd.DataFrame, path, mode: str = "exact") -> None:
    """The check `tests/test_leakage.py` and every trainer runs."""
    assert_clean_frame(pairs, load(path)["keys"])
