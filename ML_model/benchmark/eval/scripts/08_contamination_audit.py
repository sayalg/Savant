#!/usr/bin/env python
"""
Step 08 - Is Boltz-2's lead on this benchmark real, or is it memorisation?

Boltz-2's affinity module was trained on ~3M Ki/Kd/IC50 measurements drawn from
**ChEMBL v34** (released 2024-04-15), BindingDB, PDBbind and PubChem. Our
benchmark is BindingDB Kd pulled from a pinned 2021-era Dataverse snapshot. The
two pools overlap, and nobody can say by how much: Boltz's training set and
processing pipeline were never released, so exact set subtraction is impossible.

What IS possible is a proxy. ChEMBL v34 is archived and downloadable, so for
every benchmark pair we can ask whether that pair - or that ligand, or a close
analogue of it - was sitting in v34 where Boltz would have found it. Stratify
the benchmark by that and compare each tool across the strata.

    pair_seen     (ligand skeleton, UniProt) has a binding activity in v34
    ligand_seen   the ligand is in v34, but not measured against this target
    analog_seen   ECFP4 Tanimoto >= --tanimoto to some v34 ligand
    unseen        none of the above

THE CONFOUND, AND WHY SAVANT IS THE CONTROL
-------------------------------------------
`pair_seen` pairs sit on well-studied targets, which are plausibly just *easier*
- deeper MSAs, more drug-like ligands, tighter label distributions. A raw
seen-vs-unseen gap for Boltz cannot separate "memorised it" from "easy pair".

Savant deconfounds it. Savant never saw ChEMBL v34; its training set is the
pinned BindingDB snapshot, and step 01 already gated every benchmark ligand at
ECFP4 Tanimoto < 0.50 against all 32,086 training ligands. So Savant's own
seen-vs-unseen gap measures *stratum difficulty alone*. The quantity that
isolates memorisation is the difference-in-differences:

    (boltz_seen - boltz_unseen) - (savant_seen - savant_unseen)

If that is ~0, the strata just differ in difficulty and the headline benchmark
stands as reported. If it is large and positive, Boltz's lead is inflated by
contamination and the benchmark needs rebuilding on post-v34 data.

Per-stratum label spread is reported alongside every r, because Pearson scales
with the spread of the labels: a stratum whose pKd range is compressed shows a
lower r for reasons that have nothing to do with either model.

INPUTS
------
ChEMBL v34, fetched by `fetch_chembl34.sh` into --chembl-dir (env CHEMBL34_DIR,
default work/chembl_34). Two files, two tiers - the script runs whichever the
available files allow and says so:

    chembl_34_chemreps.txt.gz   226M   -> ligand_seen, analog_seen
    chembl_34_sqlite/*.db       4.5G   -> pair_seen   (the decisive one)

Ligands are matched on the **InChIKey skeleton** (first 14 characters), so a
salt, tautomer or stereoisomer of a training ligand still counts as seen.

USAGE
-----
    python scripts/08_contamination_audit.py                 # pair + ligand, seconds
    python scripts/08_contamination_audit.py --tanimoto-scan # + analog, ~30 min
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

from common import OUT, WORK, ensure_dirs, log

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SCORED_DEFAULT = os.path.join(OUT, "savant_benchmark_scored.csv")

FLAGS_CSV = os.path.join(OUT, "contamination_flags.csv")
STRATUM_CSV = os.path.join(OUT, "contamination_by_stratum.csv")
DID_CSV = os.path.join(OUT, "contamination_did.csv")
REPORT_MD = os.path.join(OUT, "contamination_audit.md")
REPORT_PNG = os.path.join(OUT, "contamination_audit.png")

# The activity types Boltz-2 says it trained the affinity head on, plus the
# screening-style types ChEMBL files the same measurements under. Widen with
# --any-activity if you want "appears in v34 at all" rather than "appears with
# a number the affinity head could have learned from".
BINDING_TYPES = ("Kd", "Ki", "IC50", "EC50", "AC50", "Potency")

# assay_type 'B' is ChEMBL's binding assays. 'F' (functional) also carries IC50s
# but those are downstream readouts, not the direct binding data Boltz curated.
ASSAY_TYPES = ("B",)

# Ordered weakest-evidence-last so `stratum` is an ordinal contamination level.
STRATA = ("pair_seen", "ligand_seen", "analog_seen", "unseen")

# label -> prediction column in the scored CSV. Boltz and Savant are the pair the
# difference-in-differences runs on; PRODIGY rides along as a physics baseline
# that has no training set at all and so should be flat across strata.
TOOLS = {
    "Boltz-2": "boltz_affinity_pic50",
    "Savant v3": "savant_struct_ecfp_protein_grouped_pk",
    "PRODIGY-LIG": "prodigy_pkd",
}
DID_PAIR = ("Boltz-2", "Savant v3")
LABEL = "neg_log10_affinity_M"

MORGAN_RADIUS, MORGAN_BITS = 2, 1024
N_BOOT = 2000
SEED = 20260825

# Popcount of every possible byte, for Tanimoto on packed fingerprints.
_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint16)


# --------------------------------------------------------------------------- #
# ChEMBL v34 lookups
# --------------------------------------------------------------------------- #
def find_chembl_files(chembl_dir: str) -> tuple[str | None, str | None]:
    """Locate the chemreps TSV and the SQLite db, whichever are present."""
    chemreps = next(iter(sorted(glob.glob(
        os.path.join(chembl_dir, "**", "chembl_*_chemreps.txt.gz"), recursive=True))), None)
    sqlite_db = next(iter(sorted(glob.glob(
        os.path.join(chembl_dir, "**", "chembl_*.db"), recursive=True))), None)
    return chemreps, sqlite_db


def load_chemreps(path: str) -> pd.DataFrame:
    """chembl_id / canonical_smiles / standard_inchi / standard_inchi_key."""
    log(f"reading {os.path.basename(path)} (this is ~2.4M rows, give it a minute)")
    with gzip.open(path, "rt") as fh:
        d = pd.read_csv(fh, sep="\t", usecols=["chembl_id", "canonical_smiles",
                                               "standard_inchi_key"])
    d = d.dropna(subset=["standard_inchi_key"])
    d["skeleton"] = d["standard_inchi_key"].str[:14]
    log(f"  {len(d):,} v34 structures, {d.skeleton.nunique():,} unique skeletons")
    return d


def pairs_in_chembl34(db_path: str, skeletons: set[str],
                      any_activity: bool) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (ligand skeletons present in v34, (skeleton, accession) pairs).

    Two passes rather than one join. Matching on substr(inchi_key, 1, 14) cannot
    use an index, so letting SQLite drive a join off it re-scans the 2.4M-row
    structure table once per benchmark ligand. Pulling the keys out first and
    intersecting in Python scans it exactly once.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        log("  pass 1/2: scanning compound_structures for benchmark ligands")
        seen_lig: set[str] = set()
        molregnos: list[int] = []
        cur = con.execute(
            "SELECT molregno, standard_inchi_key FROM compound_structures "
            "WHERE standard_inchi_key IS NOT NULL")
        for molregno, key in cur:
            skel = key[:14]
            if skel in skeletons:
                seen_lig.add(skel)
                molregnos.append(molregno)
        log(f"    {len(seen_lig):,} of {len(skeletons):,} benchmark ligands are in v34 "
            f"({len(molregnos):,} molregnos incl. salts/stereoisomers)")

        if not molregnos:
            return seen_lig, set()

        log("  pass 2/2: joining their activities to UniProt accessions")
        con.execute("CREATE TEMP TABLE q (molregno INTEGER PRIMARY KEY)")
        con.executemany("INSERT OR IGNORE INTO q VALUES (?)", [(m,) for m in molregnos])

        where = [f"a.assay_type IN ({','.join('?' * len(ASSAY_TYPES))})"]
        params: list = list(ASSAY_TYPES)
        if not any_activity:
            where.append(f"act.standard_type IN ({','.join('?' * len(BINDING_TYPES))})")
            params += list(BINDING_TYPES)

        rows = con.execute(f"""
            SELECT DISTINCT substr(cs.standard_inchi_key, 1, 14), comp.accession
            FROM q
            JOIN compound_structures cs   ON cs.molregno = q.molregno
            JOIN activities act           ON act.molregno = q.molregno
            JOIN assays a                 ON a.assay_id = act.assay_id
            JOIN target_components tc     ON tc.tid = a.tid
            JOIN component_sequences comp ON comp.component_id = tc.component_id
            WHERE comp.accession IS NOT NULL AND {' AND '.join(where)}
        """, params).fetchall()
    finally:
        con.close()

    pairs = {(skel, acc) for skel, acc in rows}
    log(f"    {len(pairs):,} distinct (ligand, target) binding pairs in v34")
    return seen_lig, pairs


def nearest_analog_tanimoto(query_smiles: list[str], ref_smiles: pd.Series,
                            chunk: int = 100_000) -> np.ndarray:
    """Max ECFP4 Tanimoto from each query ligand to any ChEMBL v34 ligand.

    Packed-bit popcount rather than RDKit's BulkTanimotoSimilarity: 859 queries
    against 2.4M references is 2 billion comparisons, which the numpy form does
    in ~12 min. Parsing the 2.4M reference SMILES is then the larger half of the
    ~30 min total, which is why the whole scan is opt-in.
    """
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    rdBase.DisableLog("rdApp.warning")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)

    def pack(smiles_iter):
        out, ok = [], []
        for s in smiles_iter:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                ok.append(False)
                continue
            out.append(np.packbits(gen.GetFingerprintAsNumPy(m).astype(np.uint8)))
            ok.append(True)
        return (np.vstack(out) if out else np.empty((0, MORGAN_BITS // 8), np.uint8)), np.array(ok)

    log(f"  fingerprinting {len(query_smiles)} benchmark ligands")
    q_fp, q_ok = pack(query_smiles)
    q_pc = _POPCOUNT[q_fp].sum(axis=1).astype(np.int32)

    best = np.zeros(len(q_fp), dtype=np.float32)
    refs = ref_smiles.to_numpy()
    n_chunks = (len(refs) + chunk - 1) // chunk
    log(f"  scanning {len(refs):,} v34 ligands in {n_chunks} chunks")

    for ci in range(n_chunks):
        r_fp, _ = pack(refs[ci * chunk:(ci + 1) * chunk])
        if not len(r_fp):
            continue
        r_pc = _POPCOUNT[r_fp].sum(axis=1).astype(np.int32)
        for qi in range(len(q_fp)):
            inter = _POPCOUNT[np.bitwise_and(r_fp, q_fp[qi])].sum(axis=1).astype(np.int32)
            union = q_pc[qi] + r_pc - inter
            np.maximum(best[qi], (inter / np.maximum(union, 1)).max(), out=best[qi:qi + 1])
        if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            log(f"    chunk {ci + 1}/{n_chunks}, running median max-Tanimoto "
                f"{np.median(best):.3f}")

    full = np.full(len(query_smiles), np.nan, dtype=np.float32)
    full[q_ok] = best
    return full


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def boot_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
            n: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap CI for Pearson r. Wide CIs here are the point: a few
    hundred rows per stratum cannot resolve a small gap, and the report should
    say so rather than let the reader eyeball two point estimates."""
    if len(x) < 10:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    rs = np.array([pearson(x[i], y[i]) for i in idx])
    rs = rs[np.isfinite(rs)]
    if not len(rs):
        return (np.nan, np.nan)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def stratum_metrics(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for tool, col in TOOLS.items():
        for stratum in STRATA + ("ALL",):
            sub = d if stratum == "ALL" else d[d["stratum"] == stratum]
            sub = sub.dropna(subset=[col, LABEL])
            if len(sub) < 3:
                continue
            y, p = sub[LABEL].to_numpy(float), sub[col].to_numpy(float)
            lo, hi = boot_ci(y, p, rng)
            rows.append({
                "tool": tool, "stratum": stratum, "n": len(sub),
                # Reported next to every r: Pearson scales with label spread, so a
                # stratum with a compressed pKd range looks worse for free.
                "label_std": float(np.std(y, ddof=1)),
                "pearson_r": pearson(y, p), "r_lo95": lo, "r_hi95": hi,
                "spearman_rho": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae": float(np.mean(np.abs(y - p))),
                "pred_std_ratio": float(np.std(p, ddof=1) / np.std(y, ddof=1)),
            })
    return pd.DataFrame(rows)


def difference_in_differences(d: pd.DataFrame, rng: np.random.Generator,
                              seen: str = "pair_seen",
                              unseen: str = "unseen") -> pd.DataFrame:
    """(boltz_seen - boltz_unseen) - (savant_seen - savant_unseen), bootstrapped.

    Restricted to rows where both tools produced a prediction, and both r's are
    recomputed from the *same* resample each iteration, so the CI accounts for
    the two tools being scored on identical pairs.
    """
    a_col, b_col = TOOLS[DID_PAIR[0]], TOOLS[DID_PAIR[1]]
    d = d.dropna(subset=[a_col, b_col, LABEL])
    s, u = d[d["stratum"] == seen], d[d["stratum"] == unseen]
    if len(s) < 10 or len(u) < 10:
        log(f"  skipping DiD: n={len(s)} {seen} / n={len(u)} {unseen}, too few to bootstrap")
        return pd.DataFrame()

    def gaps(ss, uu):
        return (pearson(ss[LABEL].to_numpy(float), ss[a_col].to_numpy(float))
                - pearson(uu[LABEL].to_numpy(float), uu[a_col].to_numpy(float)),
                pearson(ss[LABEL].to_numpy(float), ss[b_col].to_numpy(float))
                - pearson(uu[LABEL].to_numpy(float), uu[b_col].to_numpy(float)))

    a_gap, b_gap = gaps(s, u)
    boots = []
    for _ in range(N_BOOT):
        ag, bg = gaps(s.iloc[rng.integers(0, len(s), len(s))],
                      u.iloc[rng.integers(0, len(u), len(u))])
        if np.isfinite(ag) and np.isfinite(bg):
            boots.append(ag - bg)
    boots = np.asarray(boots)

    return pd.DataFrame([{
        "seen_stratum": seen, "unseen_stratum": unseen,
        "n_seen": len(s), "n_unseen": len(u),
        f"{DID_PAIR[0]}_gap": a_gap, f"{DID_PAIR[1]}_gap": b_gap,
        "did": a_gap - b_gap,
        "did_lo95": float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
        "did_hi95": float(np.percentile(boots, 97.5)) if len(boots) else np.nan,
        # Fraction of resamples where Boltz's gap does not exceed Savant's, i.e.
        # a one-sided bootstrap p for "contamination inflates Boltz".
        "p_one_sided": float((boots <= 0).mean()) if len(boots) else np.nan,
    }])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(flags: pd.DataFrame, strat: pd.DataFrame, did: pd.DataFrame,
                 modes: list[str], args) -> None:
    counts = flags["stratum"].value_counts().reindex(STRATA).fillna(0).astype(int)
    lines = [
        "# Contamination audit - is Boltz-2's lead memorisation?",
        "",
        f"Benchmark: `{os.path.basename(args.scored)}` ({len(flags)} pairs)  ",
        f"Reference: ChEMBL v34 (2024-04-15), the affinity-module training source  ",
        f"Evidence available: {', '.join(modes)}  ",
        f"Ligand matching: InChIKey skeleton (first 14 chars)"
        + (f"; analogues at Tanimoto >= {args.tanimoto}" if "analog_seen" in modes else ""),
        "",
        "## Strata", "",
        "| stratum | n | share |", "|---|---:|---:|",
    ]
    for s in STRATA:
        lines.append(f"| `{s}` | {counts[s]} | {counts[s] / len(flags):.1%} |")

    lines += ["", "## Accuracy by stratum", "",
              "`label_std` is the pKd spread within the stratum. Pearson r scales with "
              "it, so compare r only between strata with similar spread.", "",
              "| tool | stratum | n | label_std | Pearson r | 95% CI | RMSE | pred_std_ratio |",
              "|---|---|---:|---:|---:|---|---:|---:|"]
    for _, r in strat.iterrows():
        ci = ("-" if not np.isfinite(r.r_lo95)
              else f"[{r.r_lo95:+.3f}, {r.r_hi95:+.3f}]")
        lines.append(f"| {r.tool} | `{r.stratum}` | {int(r.n)} | {r.label_std:.2f} | "
                     f"{r.pearson_r:+.3f} | {ci} | {r.rmse:.3f} | {r.pred_std_ratio:.3f} |")

    lines += ["", "## Difference-in-differences", "",
              f"`({DID_PAIR[0]} seen - unseen) - ({DID_PAIR[1]} seen - unseen)`. "
              f"{DID_PAIR[1]} never saw ChEMBL v34, so its own gap is stratum difficulty; "
              "subtracting it leaves the part attributable to contamination.", ""]
    if did.empty:
        lines.append("_Not computed - too few rows in one stratum._")
    else:
        r = did.iloc[0]
        lines += [
            f"| quantity | value |", "|---|---:|",
            f"| {DID_PAIR[0]} gap | {r[f'{DID_PAIR[0]}_gap']:+.3f} |",
            f"| {DID_PAIR[1]} gap | {r[f'{DID_PAIR[1]}_gap']:+.3f} |",
            f"| **difference-in-differences** | **{r.did:+.3f}** |",
            f"| 95% CI | [{r.did_lo95:+.3f}, {r.did_hi95:+.3f}] |",
            f"| one-sided bootstrap p | {r.p_one_sided:.3f} |",
            "",
            "**Reading it.** A CI spanning 0 means this benchmark cannot detect "
            "contamination-driven inflation at this sample size - the headline "
            "comparison stands as reported, and you can say so. A CI clearly above 0 "
            "means Boltz's lead is partly memorisation and the benchmark should be "
            "rebuilt from post-v34 data (ChEMBL 35+, publication year >= 2024).",
        ]
    lines += ["", "---", "",
              "Caveat that belongs in the paper: Boltz-2's training set was never "
              "released, so ChEMBL v34 membership is a *proxy* for what the model saw, "
              "not set subtraction. It also drew on BindingDB and PubChem, so `unseen` "
              "here means 'not in v34', not 'not in training'. This bounds the "
              "contamination; it does not eliminate it.", ""]

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log(f"wrote {REPORT_MD}")


def write_plot(strat: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not available, skipping plot")
        return

    order = [s for s in STRATA if (strat["stratum"] == s).any()]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(order) + 3, 4.6))
    width = 0.8 / len(TOOLS)

    for i, tool in enumerate(TOOLS):
        sub = strat[strat.tool == tool].set_index("stratum").reindex(order)
        x = np.arange(len(order)) + (i - (len(TOOLS) - 1) / 2) * width
        err = np.vstack([(sub.pearson_r - sub.r_lo95).to_numpy(),
                         (sub.r_hi95 - sub.pearson_r).to_numpy()])
        ax.bar(x, sub.pearson_r, width, label=tool,
               yerr=np.nan_to_num(err), capsize=3, alpha=0.9)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{s}\nn={int(strat[strat.stratum == s].n.max())}" for s in order])
    ax.set_ylabel("Pearson r vs experimental pKd")
    ax.set_title("Accuracy by ChEMBL v34 contamination stratum\n"
                 "(Savant never saw v34 - its profile is stratum difficulty)",
                 fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_PNG, dpi=150)
    log(f"wrote {REPORT_PNG}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default=SCORED_DEFAULT, help="scored benchmark CSV")
    ap.add_argument("--chembl-dir", default=os.environ.get(
        "CHEMBL34_DIR", os.path.join(WORK, "chembl_34")))
    ap.add_argument("--tanimoto-scan", action="store_true",
                    help="also compute analog_seen (~30 min, needs chemreps)")
    ap.add_argument("--tanimoto", type=float, default=0.50,
                    help="analogue threshold, matching step 01's training gate")
    ap.add_argument("--any-activity", action="store_true",
                    help="count any binding-assay record, not just Kd/Ki/IC50/EC50")
    args = ap.parse_args()

    ensure_dirs(OUT)
    d = pd.read_csv(args.scored)
    log(f"{len(d)} benchmark pairs, {d.inchikey.nunique()} ligands, "
        f"{d.uniprot_id.nunique()} targets")

    chemreps_path, sqlite_path = find_chembl_files(args.chembl_dir)
    if not chemreps_path and not sqlite_path:
        log(f"ERROR: no ChEMBL v34 files under {args.chembl_dir}")
        log("       run: bash scripts/fetch_chembl34.sh")
        return 1

    d["skeleton"] = d["inchikey"].astype(str).str[:14]
    skeletons = set(d["skeleton"])
    modes: list[str] = []

    # --- pair_seen / ligand_seen ------------------------------------------- #
    d["pair_seen"] = False
    d["ligand_seen"] = False
    if sqlite_path:
        log(f"querying {os.path.basename(sqlite_path)}")
        seen_lig, pairs = pairs_in_chembl34(sqlite_path, skeletons, args.any_activity)
        d["ligand_seen"] = d["skeleton"].isin(seen_lig)
        d["pair_seen"] = [(s, a) in pairs for s, a in zip(d["skeleton"], d["uniprot_id"])]
        modes += ["pair_seen", "ligand_seen"]
    elif chemreps_path:
        log("no SQLite db - ligand_seen only, pair_seen cannot be determined")
        reps = load_chemreps(chemreps_path)
        d["ligand_seen"] = d["skeleton"].isin(set(reps["skeleton"]))
        modes += ["ligand_seen"]

    # --- analog_seen -------------------------------------------------------- #
    d["max_chembl34_tanimoto"] = np.nan
    if args.tanimoto_scan:
        if not chemreps_path:
            log("WARNING: --tanimoto-scan needs chembl_34_chemreps.txt.gz, skipping")
        else:
            reps = load_chemreps(chemreps_path)
            uniq = d.drop_duplicates("inchikey")[["inchikey", "ligand_smiles"]]
            best = nearest_analog_tanimoto(uniq["ligand_smiles"].tolist(),
                                           reps["canonical_smiles"])
            d["max_chembl34_tanimoto"] = d["inchikey"].map(
                dict(zip(uniq["inchikey"], best)))
            modes += ["analog_seen"]
    d["analog_seen"] = d["max_chembl34_tanimoto"].ge(args.tanimoto).fillna(False)

    # --- stratum ------------------------------------------------------------ #
    d["stratum"] = np.select(
        [d["pair_seen"], d["ligand_seen"], d["analog_seen"]],
        ["pair_seen", "ligand_seen", "analog_seen"], default="unseen")
    log("strata: " + ", ".join(
        f"{k}={v}" for k, v in d["stratum"].value_counts().items()))

    # --- metrics ------------------------------------------------------------ #
    rng = np.random.default_rng(SEED)
    log(f"bootstrapping ({N_BOOT} resamples)")
    strat = stratum_metrics(d, rng)
    did = difference_in_differences(d, rng)

    keep = ["benchmark_id", "uniprot_id", "inchikey", "skeleton", "novelty_tier", LABEL,
            "pair_seen", "ligand_seen", "analog_seen", "max_chembl34_tanimoto", "stratum"]
    d[[c for c in keep if c in d]].to_csv(FLAGS_CSV, index=False)
    strat.to_csv(STRATUM_CSV, index=False)
    if not did.empty:
        did.to_csv(DID_CSV, index=False)
    log(f"wrote {FLAGS_CSV}")
    log(f"wrote {STRATUM_CSV}")

    write_report(d, strat, did, modes, args)
    write_plot(strat)

    if not did.empty:
        r = did.iloc[0]
        log("")
        log(f"  {DID_PAIR[0]} seen-unseen gap : {r[f'{DID_PAIR[0]}_gap']:+.3f}")
        log(f"  {DID_PAIR[1]} seen-unseen gap : {r[f'{DID_PAIR[1]}_gap']:+.3f}")
        log(f"  difference-in-differences  : {r.did:+.3f} "
            f"[{r.did_lo95:+.3f}, {r.did_hi95:+.3f}]  p={r.p_one_sided:.3f}")
        log("")
        if np.isfinite(r.did_lo95) and r.did_lo95 > 0:
            log("  -> CI excludes 0: Boltz's lead is partly contamination. Rebuild the")
            log("     benchmark on ChEMBL 35+ records with publication year >= 2024.")
        else:
            log("  -> CI spans 0: no detectable contamination-driven inflation at this")
            log("     sample size. The headline comparison stands; report this as the")
            log("     leakage control rather than rebuilding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
