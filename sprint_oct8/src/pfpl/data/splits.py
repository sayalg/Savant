"""Splits, and the leakage guarantees they are supposed to deliver (Section 5.3).

This is where the previous attempt's headline number came from and where it went.
Two lessons are encoded here as code rather than as advice:

1. **A protein-GROUPED split is not enough.** Grouping by protein id put 21
   near-identical HIV-1 RT constructs in `fit` while another RT sat in `test` at
   0.94 k-mer containment. Clustering at 30% identity collapses those into one
   group. `protein_cluster` is the default, and `mmseqs` output is the input.
2. **A ligand can still leak.** Clustering fixes only the protein axis. The
   `double` split holds out pairs whose protein cluster AND Murcko scaffold are
   both unseen, and DISCARDS the pairs that share one axis - because a pair whose
   protein is held out but whose scaffold was trained on measures memorisation, and
   mixing the two is how an internal r of 0.580 became 0.327 externally.

Every split exposes `describe()` output with the overlap statistics, and
`tests/test_leakage.py` asserts the guarantee each one advertises. A split that
cannot prove its guarantee fails the Phase 5 gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("random", "scaffold", "protein_cluster", "double", "time",
               "leave_viral_out")


# ------------------------------------------------------------------- clustering

def clusters_from_mmseqs(tsv, n_proteins: int, header_prefix: str = "p") -> np.ndarray:
    """mmseqs createtsv output (representative<TAB>member) -> cluster id per index.

    FASTA headers are written as `p<index>` by phase1, so the mapping back to array
    positions is exact rather than inferred from order.
    """
    rep_of: dict[int, int] = {}
    reps: dict[str, int] = {}
    for ln in Path(tsv).read_text().splitlines():
        if not ln.strip():
            continue
        rep, mem = ln.split("\t")[:2]
        rep_of[int(mem.removeprefix(header_prefix))] = reps.setdefault(rep, len(reps))
    missing = [i for i in range(n_proteins) if i not in rep_of]
    if missing:
        ## mmseqs drops nothing by default, so this means the FASTA and the protein
        ## table disagree. Fail rather than quietly give the strays their own cluster,
        ## which would make them look like held-out proteins.
        raise SystemExit(f"{len(missing)} proteins absent from {tsv} "
                         f"(e.g. p{missing[0]}); FASTA and protein table are out of sync")
    out = np.empty(n_proteins, np.int32)
    for i, c in rep_of.items():
        out[i] = c
    return out


def murcko_scaffolds(smiles) -> np.ndarray:
    from ..preflight import require_rdkit
    require_rdkit("scaffold splitting")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out.append(str(s))
            continue
        try:
            out.append(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) or str(s))
        except Exception:
            out.append(str(s))
    return np.asarray(out)


# ----------------------------------------------------------------------- splits

def grouped_split(groups: np.ndarray, test_frac: float = 0.2, seed: int = 1337,
                  overshoot: float = 1.25):
    """Hold out whole groups, sized by PAIR count rather than by group count.

    Group sizes are extremely skewed here (one protein had 10,163 ligands, the
    median had 2), so drawing 20% of the *groups* yields anywhere from 2% to 60% of
    the rows. Accumulating until the pair quota is met keeps the fold size honest.

    `overshoot` handles the other end of the skew: a single group larger than the
    whole quota would otherwise be admitted on its first draw and take 90%+ of the
    rows with it. A group is skipped when it would push the test fold past
    `quota * overshoot`, unless nothing has been chosen yet - so the fold is never
    empty, and a large group still enters whenever it is drawn while there is room.

    The tradeoff is real and worth stating: capping the overshoot slightly biases
    the test fold toward smaller groups. Per-target metrics need targets with at
    least `MIN_TARGET_N` ligands, so `overshoot` should stay loose enough that the
    big targets are still reachable - which they are whenever the corpus is large
    relative to its biggest group (here, 10,163 against a 120,000-pair quota).
    """
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    quota = test_frac * len(groups)
    cap = quota * overshoot
    chosen, got = set(), 0
    for i in order:
        if got >= quota:
            break
        ## The cap applies from the first draw, including it. Exempting the first
        ## group would let a single group bigger than the whole quota in whenever it
        ## happened to be drawn first, taking 96% of the rows with it.
        if got + counts[i] > cap:
            continue
        chosen.add(uniq[i])
        got += counts[i]
    if not chosen:
        ## Every group is larger than the cap (a tiny corpus, or one group holding
        ## nearly everything). Take the smallest so the fold is never empty.
        chosen = {uniq[int(np.argmin(counts))]}
    test = np.isin(groups, list(chosen))
    return ~test, test


def three_way(groups: np.ndarray, val_frac: float = 0.1, test_frac: float = 0.2,
              seed: int = 1337):
    """train / val / test, all group-disjoint. Model selection uses VAL only."""
    fit, test = grouped_split(groups, test_frac, seed)
    sub = np.where(fit)[0]
    inner_tr, inner_va = grouped_split(groups[sub], val_frac / max(1 - test_frac, 1e-9),
                                       seed + 1)
    train = np.zeros_like(fit)
    val = np.zeros_like(fit)
    train[sub[inner_tr]] = True
    val[sub[inner_va]] = True
    return train, val, test


def double_disjoint(protein_cluster: np.ndarray, scaffold: np.ndarray,
                    val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 1337):
    """Test pairs unseen on BOTH axes; pairs sharing exactly one axis are DISCARDED.

    Discarding is the point, not an inefficiency.
    """
    p_tr, p_te = grouped_split(protein_cluster, test_frac, seed)
    s_tr, s_te = grouped_split(scaffold, test_frac, seed + 1)
    test = p_te & s_te
    fit = p_tr & s_tr
    sub = np.where(fit)[0]
    i_tr, i_va = grouped_split(protein_cluster[sub],
                               val_frac / max(1 - test_frac, 1e-9), seed + 2)
    train = np.zeros_like(fit)
    val = np.zeros_like(fit)
    train[sub[i_tr]] = True
    val[sub[i_va]] = True
    return train, val, test


def leave_viral_out(protein_cluster: np.ndarray, is_viral: np.ndarray,
                    val_frac: float = 0.1, seed: int = 1337):
    """Train on everything non-viral; test on viral clusters.

    Whole clusters move together, so a viral target cannot be "held out" while a
    95%-identical strain of the same virus stays in training.
    """
    viral_clusters = np.unique(protein_cluster[is_viral])
    test = np.isin(protein_cluster, viral_clusters)
    sub = np.where(~test)[0]
    i_tr, i_va = grouped_split(protein_cluster[sub], val_frac, seed)
    train = np.zeros_like(test)
    val = np.zeros_like(test)
    train[sub[i_tr]] = True
    val[sub[i_va]] = True
    return train, val, test


def time_split(year: np.ndarray, cutoff_val: int, cutoff_test: int):
    """Chronological: train < cutoff_val <= val < cutoff_test <= test."""
    y = np.asarray(year, float)
    return y < cutoff_val, (y >= cutoff_val) & (y < cutoff_test), y >= cutoff_test


def build(name: str, pairs: pd.DataFrame, protein_cluster: np.ndarray,
          scaffold: np.ndarray | None = None, is_viral: np.ndarray | None = None,
          val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 1337):
    """Dispatch. `protein_cluster` and `is_viral` are indexed per PAIR, not per protein."""
    if name == "random":
        return three_way(np.arange(len(pairs)), val_frac, test_frac, seed)
    if name == "scaffold":
        if scaffold is None:
            raise SystemExit("scaffold split needs scaffolds")
        return three_way(scaffold, val_frac, test_frac, seed)
    if name == "protein_cluster":
        return three_way(protein_cluster, val_frac, test_frac, seed)
    if name == "double":
        if scaffold is None:
            raise SystemExit("double split needs scaffolds")
        return double_disjoint(protein_cluster, scaffold, val_frac, test_frac, seed)
    if name == "leave_viral_out":
        if is_viral is None:
            raise SystemExit("leave_viral_out needs a per-pair viral mask")
        return leave_viral_out(protein_cluster, is_viral, val_frac, seed)
    if name == "time":
        y = pairs.get("year")
        if y is None:
            raise SystemExit("time split needs a `year` column")
        return time_split(y.to_numpy(), 2018, 2020)
    raise SystemExit(f"unknown split {name!r}; have {SPLIT_NAMES}")


# ------------------------------------------------------------------- inspection

def describe(name: str, train, val, test, protein_cluster, scaffold=None,
             y=None, strict: bool = True) -> dict:
    """The overlap statistics, and the Phase 5 gate.

    `strict` makes an overlap fatal. That is deliberate: a split that silently
    fails its own guarantee produces numbers that look fine and mean nothing.
    """
    def groups(m, g):
        return set(np.unique(np.asarray(g)[m]).tolist())

    d = {"split": name,
         "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
         "n_discarded": int(len(train) - train.sum() - val.sum() - test.sum())}
    pt, pv, pe = (groups(m, protein_cluster) for m in (train, val, test))
    d["protein_clusters_train"] = len(pt)
    d["protein_clusters_test"] = len(pe)
    d["protein_overlap_train_test"] = len(pt & pe)
    d["protein_overlap_train_val"] = len(pt & pv)
    if scaffold is not None:
        st, se = groups(train, scaffold), groups(test, scaffold)
        d["scaffolds_test"] = len(se)
        d["scaffold_overlap_train_test"] = len(st & se)
    if y is not None:
        y = np.asarray(y, float)
        for k, m in (("train", train), ("val", val), ("test", test)):
            if m.sum():
                d[f"{k}_y_mean"] = round(float(y[m].mean()), 4)
                d[f"{k}_y_std"] = round(float(y[m].std()), 4)

    promised = {
        "protein_cluster": ["protein_overlap_train_test"],
        "double": ["protein_overlap_train_test", "scaffold_overlap_train_test"],
        "leave_viral_out": ["protein_overlap_train_test"],
        "scaffold": ["scaffold_overlap_train_test"],
    }.get(name, [])
    violations = {k: d[k] for k in promised if d.get(k, 0) > 0}
    d["guarantee_ok"] = not violations
    if violations:
        msg = (f"split {name!r} promises disjoint groups but has {violations}")
        if strict:
            raise SystemExit(msg + "\n  This is the failure that produced a 0.58 internal "
                                   "score against 0.33 external. Fix the grouping.")
        print("WARNING: " + msg)
    print(json.dumps(d, indent=1))
    return d


def save(path, name: str, train, val, test, meta: dict) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p / f"{name}.npz", train=train, val=val, test=test)
    (p / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {p / f'{name}.npz'}")


def load(path, name: str):
    p = Path(path) / f"{name}.npz"
    if not p.exists():
        have = sorted(q.stem for q in Path(path).glob("*.npz"))
        raise SystemExit(
            f"split {name!r} has not been built ({p} missing).\n"
            f"  built so far: {have or 'none'}\n\n"
            "    python scripts/phase5_splits.py\n\n"
            "  The splits are frozen deliberately: rebuilding them with a different\n"
            "  seed invalidates every number already measured against the old ones.")
    z = np.load(p)
    return z["train"], z["val"], z["test"]
