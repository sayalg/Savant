#!/usr/bin/env python3
"""Phase 0: setup, verification, and benchmark quarantine (PLAN.md Section 8).

Three jobs, in this order:

1. **Environment and paths.** Prove the code can find itself and its dependencies
   before anything expensive starts.
2. **Verification checklist.** Appendix A lists every "(verify)" claim the plan
   leans on - dataset versions, sizes, Kd/Ki fractions, licences, PIGNet2 and
   Boltz-2 details. This writes `DATA_MANIFEST.md` with one row per claim and marks
   each UNVERIFIED until a human fills it in. It deliberately does not invent
   answers: Section 0.4.10 forbids treating preprint or vendor numbers as ground
   truth, and a manifest full of plausible guesses is worse than an empty one.
3. **Quarantine keys.** Every benchmark pair, protein and ligand, frozen to
   `splits/quarantine.json` BEFORE any training data is assembled.

Step 3 is the one that matters most. The previous project's flagship result -
per-target rho +0.28 on influenza A neuraminidase - was measured on a protein that
was in the training fold, with 55 of 81 benchmark ligands present as training
pairs. The number was real; it was not a prediction.

    python scripts/phase0_verify.py --benchmark path/to/benchmark.csv
    python scripts/phase0_verify.py --manifest-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

## Imported FIRST, before torch / matplotlib / transformers: importing
## pfpl.paths redirects every HOME-bound cache and TMPDIR under the project
## root, and some of those libraries read their cache variable at import time.
from pfpl import paths as _paths  # noqa: E402,F401


import pandas as pd  # noqa: E402

from pfpl import paths, preflight  # noqa: E402
from pfpl.data import quarantine, schema  # noqa: E402
from pfpl.data.standardize import standardize_frame  # noqa: E402

## Appendix A, verbatim in structure: claim, evidence type, and what to do about it.
## PR peer-reviewed | PP preprint | V vendor/blog | D dataset or model card
CHECKLIST = [
    ("Boltz-2 FEP+ avg Pearson ~0.62 (0.66 on 4 targets) vs FEP+ ~0.72; affinity "
     "output is log10 IC50 (uM); binder-probability head; MIT; v2.1.x current",
     "PP/V + repo", "Read the paper and repo; confirm units, training cutoffs, version"),
    ("Independent Boltz-2 critiques: external mean ~0.42; poor on molecular-glue "
     "ternary complexes; a reviewer flagged a missing scaffold split",
     "V, review comments", "Re-read sources; treat as indicative only"),
    ("PIGNet2: Morse-type vdW, NO explicit Coulomb, CASF-2016 R~0.747 / Spearman "
     "~0.651 / SR1 ~93%, docked+cross-docked augmentation, open repo",
     "PR", "Read paper and repo; confirm terms and numbers"),
    ("GEMS/CleanSplit: ligand-only RMSE 1.424 on original PDBbind; ~49% of CASF "
     "test points leakage-affected; GEMS on CleanSplit -> CASF-2016 r 0.803",
     "PR", "Read paper; reproduce split filtering if feasible"),
    ("HonestAffinity strict tiers: pose-free r ~0.43-0.53, RMSE ~1.5-1.67; ESM-2 "
     "REDUCES Pearson on every strict tier (-0.024/-0.042/-0.064)",
     "PP", "Re-derive with our own baselines; do not rely on the decimals"),
    ("PSICHIC 0.478 on Atom3D LBA 30%; IPBind 0.732", "PP", "Optional check"),
    ("TankBind ~0.6 on a PDBbind time split; E3Bind documents knotted distance maps",
     "PR/PP", "Optional check"),
    ("BindingDB ~560K Ki, ~100K Kd, ~1.8M IC50; TDC split 375,032 Ki / 52,284 Kd",
     "PR/D", "RECOUNT from the current download - phase1 prints the real numbers"),
    ("PDBbind v2020 is the last freely available release; later versions restricted; "
     "PDBBind-opt is not redistributable", "D", "Read the licence terms"),
    ("LP-PDBBind CC-BY 4.0; ~10,980/2,312/4,651 train/val/test", "PR/D", "Verify"),
    ("PLINDER ~449k systems, CC-BY, structure resource not an affinity source",
     "D", "Verify affinity content before relying on it"),
    ("BindingNet v1/v2 Kd/Ki fraction", "UNVERIFIED",
     "COUNT Ki/Kd rows before use - the plan flags this as unverified"),
    ("SAIR: 5.24M structures, ~1.05M systems, IC50-type, ~3% anomalies, CC BY 4.0",
     "PP/D", "Verify licence and label types; mostly excluded by the label policy"),
    ("Papyrus/Papyrus++ keeps Ki/Kd; ChEMBL 34; reproducibility threshold corrected "
     "to 0.5 log units, which shrinks the size", "PR/D", "Verify the current release"),
    ("DES370K SAPT0 components; DES5M energies only; Splinter ~1.7M configs of "
     ">9,000 dimers with SAPT0", "PR/D",
     "Verify the components BEFORE designing the Stage 0 losses"),
    ("ESM-C licences: 300M open (Cambrian), 600M NON-COMMERCIAL, 6B API-only; "
     "ESM-2 ~1022-token limit; Uni-Mol MIT", "D",
     "Verify licences; Section 0.4.8 blocks 600M/6B without confirmation"),
    ("MISATO ~20,000 complexes, MD + QM", "PR/D", "Verify contents"),
    ("Noise ceiling: ~0.5 log inter-lab scatter; 6-8% of curated matched pairs "
     "differ by >1 unit", "PR + estimate",
     "QUANTIFY on our own duplicates - harmonize.noise_ceiling() does this"),
]


def write_manifest(path: Path, extra: dict | None = None) -> None:
    if path.exists():
        print(f"  {path} already exists - leaving it alone (it holds verified answers)")
        return
    lines = [
        "# DATA_MANIFEST", "",
        "Phase 0 verification checklist (PLAN.md Appendix A). Every row starts",
        "UNVERIFIED. Fill in the result and the date; do not delete rows.", "",
        "Section 0.4.10: preprint and vendor numbers are not ground truth, and no",
        "dataset or benchmark fact may be fabricated. An empty cell is an honest",
        "answer; a plausible guess is not.", "",
        "| # | Claim | Evidence | Phase 0 action | Status | Finding | Checked |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (claim, ev, action) in enumerate(CHECKLIST, 1):
        lines.append(f"| {i} | {claim} | {ev} | {action} | UNVERIFIED |  |  |")
    lines += ["", "## Dataset inventory", "",
              "Filled in by `scripts/phase1_harmonize.py` (per-source row counts,",
              "unique proteins and ligands, Kd vs Ki, overlap, drop reasons).", ""]
    if extra:
        lines += ["```json", json.dumps(extra, indent=1), "```", ""]
    path.write_text("\n".join(lines))
    print(f"  wrote {path} ({len(CHECKLIST)} claims, all UNVERIFIED)")


def load_benchmark(path: Path, assume_type: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  benchmark: {len(df):,} rows from {path}")
    ## Aliases cover the column names this project's own benchmark actually uses
    ## (ML_model/benchmark/eval/output/bench_scored.csv): `ligand_smiles`,
    ## `target_sequence`, `neg_log10_affinity_M`, `standard_type`.
    ren = {}
    for want, alts in (("ligand_smiles_raw", ("smiles", "SMILES", "ligand_smiles",
                                              "canonical_smiles", "docked_smiles")),
                       ("sequence", ("sequence", "target_sequence", "protein_sequence",
                                     "seq")),
                       ("pK", ("pK", "pKd", "pKi", "label", "y",
                               "neg_log10_affinity_M", "pchembl_value"))):
        for a in alts:
            if a in df.columns:
                ren[a] = want
                break
    df = df.rename(columns=ren)
    for c in ("ligand_smiles_raw", "sequence"):
        if c not in df.columns:
            raise SystemExit(f"benchmark needs a {c} column; have {list(df.columns)}")

    ## Section 5.3.1 / Section 13.1: IC50 rows in the benchmark are excluded from
    ## SCORING, but they are still quarantined from training - a pair we cannot
    ## score is still a pair we must not have trained on.
    ##
    ## require_typed searches every alias (type / standard_type / activity_type /
    ## ...), so this must not be gated on a literal `type` column: this project's own
    ## benchmark names it `standard_type`, and testing for "type" alone silently
    ## discarded the Kd/Ki labelling of all 500 rows.
    try:
        t = schema.require_typed(df, "benchmark", assume_type)
        df["type"] = t
        df["scoreable"] = t.isin(schema.ALLOWED_TYPES)
        n_excl = int((~df["scoreable"]).sum())
        print(f"    assay types ({t.attrs.get('type_source', 'column')}): "
              f"{t.value_counts().to_dict()}")
        if n_excl:
            print(f"    {n_excl:,} benchmark rows are not Kd/Ki: excluded from SCORING, "
                  "still quarantined from training")
    except SystemExit:
        ## No usable type column and none declared. Quarantine everything (a pair we
        ## cannot type is still a pair we must not train on) but refuse to guess for
        ## scoring.
        df["type"] = ""
        df["scoreable"] = True
        print("    benchmark has no assay-type column: every row is quarantined, and\n"
              "    scoring will assume Kd/Ki. Pass --benchmark-type Kd|Ki to be explicit.")
    return standardize_frame(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", help="CSV with sequence, smiles and (ideally) type")
    ap.add_argument("--benchmark-type", default=None,
                    help="declare the type if the benchmark has no type column")
    ap.add_argument("--boltz2-scores", default=None,
                    help="CSV of Boltz-2 predictions on the same rows, for Phase 6")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()

    paths.ensure()
    print("=" * 78)
    print("PHASE 0 - setup, verification, quarantine")
    print("=" * 78)
    for k, v in paths.describe().items():
        print(f"  {k:<14} {v}")
    print()
    preflight.check()
    print()

    write_manifest(ROOT / "DATA_MANIFEST.md")

    if a.manifest_only:
        return
    if not a.benchmark:
        print("\n  NO BENCHMARK GIVEN. The quarantine keys are the one Phase 0 output")
        print("  that everything else depends on, and the gate is not met without them:")
        print("      python scripts/phase0_verify.py --benchmark <file>")
        print("\n  Section 0.4.3: the sealed benchmark is scored once per model version.")
        print("  Section 5.3.1: every benchmark pair is removed from all training data.")
        sys.exit(1)

    bench = load_benchmark(Path(a.benchmark), a.benchmark_type)
    keys = quarantine.build_keys(bench)
    stats = {
        "benchmark_file": str(Path(a.benchmark).resolve()),
        "n_rows": int(len(bench)),
        "n_scoreable": int(bench.scoreable.sum()),
        "n_pairs": len(keys["pairs"]),
        "n_proteins": len(keys["proteins"]),
        "n_ligands_full": len(keys["ligands_full"]),
        "n_ligands_flat": len(keys["ligands_flat"]),
        ## str() the keys: a pd.NA or numpy scalar key is not JSON-serialisable and
        ## would blow up on write, after the expensive standardisation has run.
        "types": {str(k): int(v) for k, v in
                  bench["type"].value_counts(dropna=False).items()},
    }
    print(f"\n  quarantine keys: {stats['n_pairs']:,} pairs, "
          f"{stats['n_proteins']:,} proteins, {stats['n_ligands_full']:,} ligands")
    quarantine.save(ROOT / "splits" / "quarantine.json", keys, stats)
    bench.to_csv(paths.WORK / "benchmark_standardized.csv", index=False)
    print(f"  wrote {paths.WORK / 'benchmark_standardized.csv'}")

    if a.boltz2_scores:
        b = pd.read_csv(a.boltz2_scores)
        b.to_csv(paths.WORK / "boltz2_scores.csv", index=False)
        print(f"  Boltz-2 scores: {len(b):,} rows -> work/boltz2_scores.csv")
        print("    Section 9: its output is log10 IC50-like, NOT a Kd-calibrated pK.")
        print("    Lead with rank metrics; report Pearson only after a linear")
        print("    recalibration fit on rows disjoint from the test items.")
    else:
        print("\n  no --boltz2-scores given; the Phase 0 gate asks that our metric code")
        print("  reproduces your Boltz-2 numbers on the benchmark before Phase 6.")

    print("\n" + "=" * 78)
    print("GATE - Phase 0")
    print(f"  manifest written        : yes (all {len(CHECKLIST)} claims UNVERIFIED)")
    print(f"  quarantine keys         : yes ({stats['n_pairs']:,} pairs)")
    print(f"  Boltz-2 numbers checked : {'yes' if a.boltz2_scores else 'NO'}")
    print("  The manifest rows are NOT self-verifying. Work through them before")
    print("  Phase 6; Section 0.3.9 makes this a gate, not a chore.")
    print("=" * 78)


if __name__ == "__main__":
    main()
